"""Связь стадий по HTTP: что едет по проводу и что происходит при рассинхроне.

Стадии конвейера живут на разных машинах, и единственное, что их связывает, —
остаток вперёд и выбранный токен назад. Здесь проверяется провод: формат,
очередь, счёт шагов и отказы. Посредник, который разносит сообщения по
рангам, подменён простым перенаправителем — маршрутизация не его дело.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.stage_http import (
    BUSY, DTYPE, KIND, SHAPE, STEP, TO, HttpStageLink, StageProtocolError, pack, unpack,
)

CPU = torch.device("cpu")


# ------------------------------------------------------------------ формат


@pytest.mark.parametrize("tensor", [
    torch.randn(3, 10240).to(torch.bfloat16),     # остаток, как он есть на декоде
    torch.randn(1, 2560),
    torch.tensor([7, 11, 13], dtype=torch.int32),  # выбранные токены
])
def test_a_tensor_survives_the_wire(tensor):
    body, shape, dtype = pack(tensor)

    assert torch.equal(unpack(body, shape, dtype, device=CPU), tensor)


def test_the_remainder_is_not_widened_on_the_way():
    """bf16 остаётся bf16: расширение удвоило бы каждый байт и не добавило
    ничего — принимающая сторона всё равно сузит обратно."""
    tensor = torch.randn(4, 10240).to(torch.bfloat16)

    body, _shape, dtype = pack(tensor)

    assert dtype == "bfloat16"
    assert len(body) == 4 * 10240 * 2


def test_an_unknown_dtype_is_refused_not_guessed():
    """Угадав, конвейер из разных сборок выдал бы уверенную чушь."""
    with pytest.raises(StageProtocolError, match="одной сборки"):
        unpack(b"\x00\x00", "1", "float8_e4m3", device=CPU)


def test_a_dtype_that_cannot_travel_is_refused_on_send():
    with pytest.raises(StageProtocolError, match="не ездит"):
        pack(torch.zeros(2, dtype=torch.bool))


# ------------------------------------------------------------------ провод


class Relay:
    """Посредник: принимает от стадии и перекладывает в указанную стадию.

    Настоящий разносит по рангам через агента; здесь достаточно словаря
    «ранг -> порт», потому что проверяется провод, а не маршрутизация.
    """

    def __init__(self):
        self.ports: dict[str, int] = {}
        self.seen: list[tuple[str, str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                to, kind = self.headers.get(TO, ""), self.headers.get(KIND, "")
                outer.seen.append((kind, to))
                targets = (list(outer.ports) if to == "*" else [to])
                for rank in targets:
                    port = outer.ports.get(rank)
                    if port is None:
                        continue
                    import http.client

                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    conn.request("POST", "/", body=body, headers={
                        KIND: kind, SHAPE: self.headers.get(SHAPE, ""),
                        DTYPE: self.headers.get(DTYPE, ""), STEP: self.headers.get(STEP, ""),
                        BUSY: self.headers.get(BUSY, ""),
                        "Content-Length": str(len(body)),
                    })
                    conn.getresponse().read()
                    conn.close()
                self.send_response(202)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def pair():
    """Две стадии, связанные через посредника."""
    relay = Relay()
    head = HttpStageLink(send_url=relay.url, listen_port=0, rank=0, size=2,
                         device=CPU, timeout_s=5)
    tail = HttpStageLink(send_url=relay.url, listen_port=0, rank=1, size=2,
                         device=CPU, timeout_s=5)
    relay.ports["0"] = head._server.server_address[1]
    relay.ports["1"] = tail._server.server_address[1]
    yield head, tail, relay
    head.close()
    tail.close()
    relay.close()


def test_the_remainder_reaches_the_next_stage(pair):
    head, tail, relay = pair
    batch = SimpleNamespace()
    remainder = torch.randn(3, 10240).to(torch.bfloat16)

    head.give(batch, remainder)

    assert torch.equal(tail.take(batch), remainder)
    assert relay.seen == [("remainder", "1")], "остаток поехал не следующей стадии"


def test_the_token_comes_back_to_everyone(pair):
    head, tail, relay = pair
    batch = SimpleNamespace()
    chosen = torch.tensor([42], dtype=torch.int32)

    tail.publish(batch, chosen)

    assert torch.equal(head.tokens(batch), chosen)
    assert relay.seen[-1] == ("tokens", "*"), "токен разослан не всем"


def test_a_full_step_round_trip(pair):
    """Круг целиком, дважды: счёт шагов не должен разойтись."""
    head, tail, relay = pair
    batch = SimpleNamespace()

    for step in range(2):
        remainder = torch.full((1, 10240), float(step)).to(torch.bfloat16)
        head.give(batch, remainder)
        assert torch.equal(tail.take(batch), remainder)
        chosen = torch.tensor([step], dtype=torch.int32)
        tail.publish(batch, chosen)
        assert torch.equal(head.tokens(batch), chosen)

    assert head._step == 2 and tail._step == 2


def _step(head, tail, rows: int = 1, *, busy_s: float = 0.0) -> None:
    """Один шаг конвейера в одном потоке: остаток вперёд, токен назад."""
    batch = SimpleNamespace()
    head.give(batch, torch.zeros((rows, 10240), dtype=torch.bfloat16))
    tail.take(batch)
    if busy_s:
        time.sleep(busy_s)
    tail.publish(batch, torch.tensor([1], dtype=torch.int32))
    head.tokens(batch)


def test_the_seam_reports_the_step_and_the_rate(pair, caplog):
    """Отчёт печатается раз в N шагов декода, называет ток/с и обнуляет счёт.

    Стенд 2026-09-27: 3 ток/с, а по первой версии секундомера делить 333 мс
    между сетью и счётом не выходило — она не мерила ни период шага, ни счёт.
    """
    import logging

    head, tail, relay = pair
    head._report_every = tail._report_every = 2
    with caplog.at_level(logging.INFO):
        for _ in range(3):          # первый шаг не в счёт: у него нет периода
            _step(head, tail)
    lines = [r.message for r in caplog.records if "шагов декода" in r.message]
    assert len(lines) == 2, lines                   # по строке на стадию
    assert all("ток/с" in line and "вне шва" in line for line in lines)
    assert head._counted == 0 and head._period == 0.0
    assert set(head._spent.values()) == {0.0}


def test_unloading_from_the_card_is_not_the_network(pair, monkeypatch):
    """Снятие выхода с карты — счёт стадии, а не сеть.

    `pack` копирует с карты синхронно и потому ждёт всю недосчитанную работу
    GPU. Первая версия секундомера писала это время в сеть, и на стенде
    отправка токена хвостом (45.6 мс) выглядела медленнее отправки остатка
    головой (2.3 мс) без всякой сетевой причины. Медленная карта здесь
    изображена медленной упаковкой.
    """
    import freetoken.engine.stage_http as wire

    real = wire.pack

    def slow_pack(tensor):
        time.sleep(0.2)
        return real(tensor)

    monkeypatch.setattr(wire, "pack", slow_pack)
    head, tail, relay = pair
    for _ in range(3):
        _step(head, tail)

    for stage in (head, tail):
        assert stage._counted == 2
        assert stage._spent["выгрузка"] / 2 >= 0.2, "ожидание карты не попало в выгрузку"
        assert stage._spent["сеть"] / 2 < 0.1, "ожидание карты утекло в сеть"


def test_the_head_step_is_the_sum_of_its_parts(pair):
    """У головы `шаг = счёт + выгрузка + сеть + ожидание`, и счёт хвоста — в ожидании.

    Хвост считает в своём потоке, как на стенде, и голова в это время
    действительно ждёт. Если бы что-то выпадало из учёта, у головы появился
    бы «вне шва» — остаток периода, которого не объясняет ни одна корзина.
    """
    head, tail, relay = pair
    steps, busy = 5, 0.05

    def tail_loop():
        batch = SimpleNamespace()
        for _ in range(steps):
            tail.take(batch)
            time.sleep(busy)                   # счёт хвоста
            tail.publish(batch, torch.tensor([1], dtype=torch.int32))

    worker = threading.Thread(target=tail_loop)
    worker.start()
    batch = SimpleNamespace()
    for _ in range(steps):
        head.give(batch, torch.zeros((1, 10240), dtype=torch.bfloat16))
        head.tokens(batch)
    worker.join(timeout=10)

    n = head._counted
    assert n == steps - 1
    parts = sum(head._spent.values())
    assert head._period - parts < 0.005 * n, "у головы часть шага не объяснена"
    assert head._spent["ожидание"] / n >= busy, "счёт хвоста не виден в ожидании головы"
    assert tail._spent["счёт"] / (tail._counted or 1) >= busy


def test_prefill_does_not_enter_the_average(pair):
    """Шаг, у которого через шов прошло больше одной строки, — префилл.

    Он в разы дольше декода, а шаг после простоя несёт в периоде сам простой.
    В среднее не попадает ни то, ни другое.
    """
    head, tail, relay = pair
    _step(head, tail)                       # первый: без периода
    _step(head, tail, rows=5, busy_s=0.3)   # префилл, и долгий
    _step(head, tail)                       # декод

    for stage in (head, tail):
        assert stage._counted == 1
        assert stage._period < 0.3, "время префилла попало в средний шаг"


def _run_pipeline(head, tail, steps, *, head_busy, tail_busy, prefill_first=False):
    """Хвост в своём потоке, голова в этом, как на стенде: голова реально ждёт."""
    def tail_loop():
        batch = SimpleNamespace()
        for _ in range(steps):
            tail.take(batch)
            time.sleep(tail_busy)
            tail.publish(batch, torch.tensor([1], dtype=torch.int32))

    worker = threading.Thread(target=tail_loop)
    worker.start()
    batch = SimpleNamespace()
    for i in range(steps):
        head.step_started()
        time.sleep(head_busy)
        rows = 5 if prefill_first and i == 0 else 1
        head.give(batch, torch.zeros((rows, 10240), dtype=torch.bfloat16))
        head.tokens(batch)
    worker.join(timeout=10)


def test_the_head_splits_every_step_into_head_peer_and_wire(pair):
    """Окно замера рисует токен как голова + хвост + дорога; сумма обязана сойтись.

    Хвост присылает своё время с токеном, голова знает своё, а дорога — остаток
    её ожидания. Браузер видит только сумму и поделить её не может.
    """
    head, tail, relay = pair
    _run_pipeline(head, tail, 5, head_busy=0.02, tail_busy=0.05)

    rows = head.series_since(0.0)
    assert len(rows) == 5 and not any(r["prefill"] for r in rows)
    for previous, row in zip(rows, rows[1:]):
        assert row["head_ms"] >= 20 and row["peer_ms"] >= 50, row
        period = 1e3 * (row["end"] - previous["end"])
        parts = row["head_ms"] + row["peer_ms"] + row["wire_ms"]
        assert abs(period - parts) < 5, (period, row)
    assert not tail.series_since(0.0), "ряд пишет только голова"


def test_prefill_on_the_head_is_timed_from_the_start_of_its_forward(pair):
    """Вход префилла у головы приходит не по связи.

    Без начала forward её счёт тянулся бы от последнего токена прошлого
    запроса, и нулевая точка ряда показала бы простой вместо префилла.
    """
    head, tail, relay = pair
    _run_pipeline(head, tail, 1, head_busy=0.0, tail_busy=0.0)   # прошлый запрос
    time.sleep(0.3)                                              # простой
    since = time.time()
    _run_pipeline(head, tail, 2, head_busy=0.05, tail_busy=0.0, prefill_first=True)

    rows = head.series_since(since)
    assert [r["prefill"] for r in rows] == [True, False]
    assert 50 <= rows[0]["head_ms"] < 250, rows[0]


def test_the_timings_endpoint_serves_the_steps_since_a_moment(pair):
    """Посредник забирает ряд по HTTP: он другой процесс, и знает только время."""
    import json
    import urllib.request

    head, tail, relay = pair
    _run_pipeline(head, tail, 2, head_busy=0.0, tail_busy=0.0)
    since = time.time()
    _run_pipeline(head, tail, 3, head_busy=0.0, tail_busy=0.0)

    port = head._server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/timings?since={since}") as answer:
        steps = json.loads(answer.read())["steps"]
    assert len(steps) == 3
    assert {"end", "head_ms", "peer_ms", "wire_ms", "prefill"} <= set(steps[0])


def test_the_seam_stopwatch_can_be_silenced(pair):
    """Нулевая частота — молчать и ничего не копить."""
    head, tail, relay = pair
    head._report_every = tail._report_every = 0
    for _ in range(3):
        _step(head, tail)
    assert head._counted == 0 and set(head._spent.values()) == {0.0}


def test_a_message_from_the_wrong_step_is_refused(pair):
    """Разойдясь на шаг, стадии считали бы разные токены и не заметили."""
    head, tail, _relay = pair
    batch = SimpleNamespace()

    tail._step = 3                       # хвост «убежал» вперёд
    head.give(batch, torch.zeros(1, 10240).to(torch.bfloat16))

    with pytest.raises(StageProtocolError, match="разошлись"):
        tail.take(batch)


def test_silence_from_a_peer_is_an_error_not_a_hang(pair):
    head, _tail, _relay = pair

    with pytest.raises(StageProtocolError, match="конвейер встал"):
        head.tokens(SimpleNamespace())


def test_a_message_that_arrives_early_waits_its_turn(pair):
    """Сосед может прислать раньше, чем стадия спросит: очередь на то и есть."""
    head, tail, _relay = pair
    batch = SimpleNamespace()
    first = torch.full((1, 10240), 1.0).to(torch.bfloat16)

    head.give(batch, first)
    import time

    time.sleep(0.2)                      # пришло, лежит в очереди
    assert torch.equal(tail.take(batch), first)


# ------------------------------------------------------------------ настройка


def _cfg(**fields):
    base = dict(layer_range="", stage_send_url="", stage_listen_port=0,
                stage_rank=0, stage_size=1)
    base.update(fields)
    return SimpleNamespace(**base)


def test_a_whole_model_gets_no_link():
    from freetoken.engine.stage import attach_stage_link

    engine = SimpleNamespace(stage_link=None, device=CPU)
    attach_stage_link(engine, _cfg())

    assert engine.stage_link is None


def test_a_stage_without_a_link_is_refused_at_startup():
    """Иначе отказ придёт на первом токене — через минуты и после того, как
    веса уже в памяти."""
    from freetoken.engine.stage import attach_stage_link

    with pytest.raises(ValueError, match="не с кем"):
        attach_stage_link(SimpleNamespace(stage_link=None, device=CPU),
                          _cfg(layer_range="0:24"))


def test_a_link_without_a_stage_is_refused_too():
    """Целая модель с соседями — это почти наверняка забытый --layer-range."""
    from freetoken.engine.stage import attach_stage_link

    with pytest.raises(ValueError, match="ни к чему"):
        attach_stage_link(SimpleNamespace(stage_link=None, device=CPU),
                          _cfg(stage_send_url="http://127.0.0.1:1/"))


def test_a_configured_stage_gets_a_link():
    from freetoken.engine.stage import attach_stage_link

    relay = Relay()
    engine = SimpleNamespace(stage_link=None, device=CPU)
    try:
        attach_stage_link(engine, _cfg(layer_range="0:24", stage_send_url=relay.url,
                                       stage_rank=0, stage_size=2))
        assert isinstance(engine.stage_link, HttpStageLink)
        assert engine.stage_link.rank == 0 and engine.stage_link.size == 2
    finally:
        if engine.stage_link is not None:
            engine.stage_link.close()
        relay.close()


# ------------------------------------------------------------------ графы
#
# Стенд 2026-09-27: `graphs_allowed` стояла только в `forward_batch`, а захват
# идёт при СБОРКЕ движка — стадия падала на захвате, не дожив до первого
# запроса: `expanded size (248320) must match existing size (10240)`, ширина
# словаря против ширины остатка. Охранник должен быть один.


def test_the_whole_model_keeps_its_graphs():
    from freetoken.engine.stage import graph_batch_sizes, graphs_allowed

    whole = SimpleNamespace(layer_range=None)

    assert graphs_allowed(whole)
    assert graph_batch_sizes(whole, [1, 2, 4]) == [1, 2, 4]
    assert graph_batch_sizes(whole, None) is None      # пусть решает движок


def test_a_stage_captures_no_graphs():
    """Пустой список — штатный путь «графы выключены» в GraphRunner."""
    from freetoken.engine.stage import graph_batch_sizes, graphs_allowed

    stage = SimpleNamespace(layer_range=(24, 48))

    assert not graphs_allowed(stage)
    assert graph_batch_sizes(stage, [1, 2, 4]) == []
    assert graph_batch_sizes(stage, None) == []


def test_an_empty_list_really_means_no_capture():
    """Проверяется не наше намерение, а то, как его понимает GraphRunner:
    пустой список -> max_graph_bs 0 -> ранний возврат из захвата."""
    from freetoken.engine.graph import _determine_cuda_graph_bs

    chosen = _determine_cuda_graph_bs(cuda_graph_bs=[], cuda_graph_max_bs=None,
                                      free_memory=64 << 30)

    assert chosen == []
    assert max(chosen, default=0) == 0


def test_capture_refuses_a_stage_at_the_capture_itself():
    """Второй охранник, у самого захвата: забыть его на месте вызова нельзя."""
    from freetoken.engine.graph import capture_refusal

    whole = SimpleNamespace(produces_logits=True, stage_input_width=0)
    head = SimpleNamespace(produces_logits=False, stage_input_width=0)
    tail = SimpleNamespace(produces_logits=True, stage_input_width=10240)

    assert capture_refusal(whole) == ""
    assert "остаток другой ширины" in capture_refusal(head)
    assert "фиксированному адресу" in capture_refusal(tail)
