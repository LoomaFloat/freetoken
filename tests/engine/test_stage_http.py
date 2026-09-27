"""Связь стадий по HTTP: что едет по проводу и что происходит при рассинхроне.

Стадии конвейера живут на разных машинах, и единственное, что их связывает, —
остаток вперёд и выбранный токен назад. Здесь проверяется провод: формат,
очередь, счёт шагов и отказы. Посредник, который разносит сообщения по
рангам, подменён простым перенаправителем — маршрутизация не его дело.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.stage_http import (
    DTYPE, KIND, SHAPE, STEP, TO, HttpStageLink, StageProtocolError, pack, unpack,
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
