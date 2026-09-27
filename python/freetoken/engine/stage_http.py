"""Связь стадий поверх HTTP: остаток вперёд, выбранные токены назад.

Почему HTTP и почему один адрес. Стадии живут на разных машинах за NAT и
друг к другу не ходят: рядом с каждой есть посредник, который знает, где
сосед, и умеет до него добраться. Поэтому здесь ровно два адреса — куда
отдавать и на каком порту принимать, — а маршрутизацию по рангам делает
посредник, читая заголовок назначения.

Формат тела — сырые байты тензора; форма и тип едут заголовками. Никакого
base64: остаток на декоде это 20 КБ, и удваивать их кодированием незачем.

Неизвестный тип НЕ угадывается. Угадывание — это как конвейер из разных
сборок превращается в уверенно работающую чушь, а это худшее, что такая
система может выдать.
"""

from __future__ import annotations

import http.client
import os
import queue
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict

import torch

from freetoken.core import Batch
from freetoken.utils import init_logger

from .stage import StageLink

#: Через сколько шагов конвейера писать средние времена шва. Ноль — молчать.
#: Печатать каждый шаг бессмысленно: на 3 ток/с это три строки в секунду шума,
#: а нужен порядок величины, а не отдельный токен.
_REPORT_EVERY = int(os.environ.get("FREETOKEN_STAGE_REPORT_EVERY", "16") or 0)
#: Из чего складывается шаг стадии; смысл каждой — в `HttpStageLink._report`.
BUCKETS = ("счёт", "выгрузка", "сеть", "ожидание")

logger = init_logger(__name__)

#: Имена типов — договор между процессами, которые могут быть разных сборок.
#: Добавить имя сюда значит поменять протокол.
DTYPES: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
}
NAMES = {value: key for key, value in DTYPES.items()}

KIND = "X-Looma-Stage-Kind"        # remainder | tokens
SHAPE = "X-Looma-Stage-Shape"      # "3,10240"
DTYPE = "X-Looma-Stage-Dtype"
STEP = "X-Looma-Stage-Step"        # номер шага: рассинхрон должен быть слышен
TO = "X-Looma-Stage-To"            # ранг назначения или "*" — всем остальным

REMAINDER, TOKENS = "remainder", "tokens"


class StageProtocolError(RuntimeError):
    """Сосед прислал то, чего эта стадия не понимает."""


def pack(tensor: torch.Tensor) -> tuple[bytes, str, str]:
    """Тензор -> (байты, форма, имя типа).

    Читается через uint8, а не через `.numpy()`: у numpy нет bfloat16 и он бы
    отказался. Байты при этом те же — на той стороне их соберёт torch,
    который тип знает.
    """
    flat = tensor.detach().to("cpu").contiguous()
    name = NAMES.get(flat.dtype)
    if name is None:
        raise StageProtocolError(f"тип {flat.dtype} не ездит по этому протоколу")
    return flat.view(torch.uint8).numpy().tobytes(), ",".join(map(str, flat.shape)), name


def unpack(data: bytes, shape: str, dtype: str, *, device: torch.device) -> torch.Tensor:
    """(байты, форма, имя типа) -> тензор, ровно такой, каким его отправили."""
    name = (dtype or "").strip()
    if name not in DTYPES:
        raise StageProtocolError(
            f"сосед прислал {name!r}, а эта стадия знает только "
            f"{', '.join(sorted(DTYPES))}. Стадии конвейера должны быть одной сборки"
        )
    dims = tuple(int(x) for x in shape.split(",") if x != "")
    flat = torch.frombuffer(bytearray(data), dtype=DTYPES[name])
    return flat.reshape(dims).to(device)


class HttpStageLink(StageLink):
    """Соседи по конвейеру через локального посредника.

    Приём асинхронный: сообщение может прийти раньше, чем стадия его
    попросит, поэтому оно кладётся в очередь по виду. Виды разделены, потому
    что средняя стадия ждёт и остаток от предыдущей, и токены с хвоста.

    Номер шага проверяется на приёме. Рассинхрон стадий — это молча неверный
    ответ, а не отказ, поэтому он должен быть слышен на первом же
    несовпадении.
    """

    def __init__(self, *, send_url: str, listen_port: int, rank: int, size: int,
                 device: torch.device, timeout_s: float = 300.0) -> None:
        self.send_url = send_url.rstrip("/")
        self.rank, self.size = rank, size
        self.device = device
        self.timeout_s = timeout_s
        # ОДИН счётчик на обе стороны. Шаг конвейера общий: на шаге k голова
        # отдаёт остаток(k), хвост принимает(k) и публикует токены(k), и
        # только после этого все переходят к k+1. Два счётчика разошлись бы
        # на первой же стадии, которая не публикует.
        self._step = 0
        self._inbox: Dict[str, queue.Queue] = {REMAINDER: queue.Queue(), TOKENS: queue.Queue()}
        # Секундомер шага (см. `_step_done`). Текущий шаг копится отдельно и
        # попадает в среднее, только если это был шаг ДЕКОДА: префилл и
        # простой между запросами исказили бы число в разы.
        self._now_spent: Dict[str, float] = dict.fromkeys(BUCKETS, 0.0)
        self._spent: Dict[str, float] = dict.fromkeys(BUCKETS, 0.0)
        self._period = 0.0
        self._counted = 0
        self._input_at: float | None = None
        self._ended_at: float | None = None
        self._prefill = False
        self._report_every = _REPORT_EVERY
        self._server = self._listen(listen_port)

    # ------------------------------------------------------------ приём
    def _listen(self, port: int) -> ThreadingHTTPServer:
        inbox = self._inbox

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b""
                kind = self.headers.get(KIND, "")
                if kind not in inbox:
                    self.send_response(400)
                    self.end_headers()
                    return
                inbox[kind].put((body, self.headers.get(SHAPE, ""),
                                 self.headers.get(DTYPE, ""), self.headers.get(STEP, "")))
                self.send_response(202)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=server.serve_forever, name="stage-link", daemon=True).start()
        logger.info_rank0(f"связь стадии слушает 127.0.0.1:{server.server_address[1]}")
        return server

    def _await(self, kind: str) -> torch.Tensor:
        started = time.perf_counter()
        try:
            body, shape, dtype, step = self._inbox[kind].get(timeout=self.timeout_s)
        except queue.Empty:
            raise StageProtocolError(
                f"сосед не прислал {kind} за {self.timeout_s:g} с; конвейер встал"
            ) from None
        if step != "" and int(step) != self._step:
            raise StageProtocolError(
                f"{kind} пришёл с шага {step}, а эта стадия на {self._step}: стадии разошлись"
            )
        self._input_at = time.perf_counter()
        self._now_spent["ожидание"] += self._input_at - started
        got = unpack(body, shape, dtype, device=self.device)
        if kind == REMAINDER and got.dim() and got.shape[0] > 1:
            self._prefill = True
        return got

    # ------------------------------------------------------------ отправка
    def _post(self, kind: str, tensor: torch.Tensor, to: str) -> None:
        started = time.perf_counter()
        if self._input_at is not None:
            self._now_spent["счёт"] += started - self._input_at
            self._input_at = None
        if kind == REMAINDER and tensor.dim() and tensor.shape[0] > 1:
            self._prefill = True
        # `pack` снимает тензор с карты синхронной копией, то есть ЖДЁТ всю
        # недосчитанную работу GPU этой стадии. Это счёт, а не сеть: первая
        # версия секундомера записывала его в сеть и путала одно с другим.
        body, shape, dtype = pack(tensor)
        packed = time.perf_counter()
        self._now_spent["выгрузка"] += packed - started
        parts = urllib.parse.urlsplit(self.send_url)
        connection = http.client.HTTPConnection(parts.netloc, timeout=self.timeout_s)
        try:
            connection.request("POST", parts.path or "/", body=body, headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(body)),
                KIND: kind, SHAPE: shape, DTYPE: dtype,
                STEP: str(self._step), TO: to,
            })
            answer = connection.getresponse()
            answer.read()
            if answer.status >= 300:
                raise StageProtocolError(f"посредник отказал на {kind}: HTTP {answer.status}")
        finally:
            connection.close()
            self._now_spent["сеть"] += time.perf_counter() - packed

    # ------------------------------------------------------------ StageLink
    def take(self, batch: Batch) -> torch.Tensor:
        return self._await(REMAINDER)

    def give(self, batch: Batch, remainder: torch.Tensor) -> None:
        self._post(REMAINDER, remainder, to=str(self.rank + 1))

    def tokens(self, batch: Batch) -> torch.Tensor:
        got = self._await(TOKENS)
        self._step += 1      # шаг этой стадии закончился
        self._step_done()
        return got

    def publish(self, batch: Batch, tokens: torch.Tensor) -> None:
        # Всем остальным: токен решает, что подать на вход следующим шагом, и
        # не узнав его, любая стадия разойдётся с остальными.
        self._post(TOKENS, tokens, to="*")
        self._step += 1      # шаг этой стадии закончился
        self._step_done()

    def _step_done(self) -> None:
        """Сложить шаг в среднее — если это был шаг декода.

        Префилл везёт через шов `[T, ширина]` с T > 1 и идёт в разы дольше, а
        шаг после простоя между запросами несёт в периоде сам простой. Оба
        сдвинули бы среднее, и всё, что с T > 1, в него не попадает.
        """
        now = time.perf_counter()
        period = None if self._ended_at is None else now - self._ended_at
        self._ended_at = now
        spent, self._now_spent = self._now_spent, dict.fromkeys(BUCKETS, 0.0)
        prefill, self._prefill = self._prefill, False
        if prefill or period is None or not self._report_every:
            return
        for name, value in spent.items():
            self._spent[name] += value
        self._period += period
        self._counted += 1
        if self._counted >= self._report_every:
            self._report()

    def _report(self) -> None:
        """Средний шаг декода этой стадии и из чего он сложен.

        - `счёт` — от момента, когда вход стадии на руках, до отправки выхода:
          своя работа стадии на хосте, с планировщиком;
        - `выгрузка` — снятие выхода с карты. Ждёт всю недосчитанную работу
          GPU этой стадии, так что это тоже счёт, только асинхронный;
        - `сеть` — от готовых байтов до подтверждения посредника;
        - `ожидание` — пока сосед пришлёт своё: сеть в пути плюс его счёт;
        - `вне шва` — остаток периода. У головы ≈ 0; у хвоста это работа
          после отправки токена, которая идёт параллельно с головой.

        У головы `шаг = счёт + выгрузка + сеть + ожидание`, и это время токена.
        """
        n = self._counted
        ms = {name: 1e3 * total / n for name, total in self._spent.items()}
        period = 1e3 * self._period / n
        logger.info(
            "шов за %d шагов декода, мс на шаг: шаг %.1f (%.2f ток/с) = "
            "счёт %.1f + выгрузка %.1f + сеть %.1f + ожидание %.1f + вне шва %.1f",
            n, period, 1e3 / period if period else 0.0,
            ms["счёт"], ms["выгрузка"], ms["сеть"], ms["ожидание"],
            period - sum(ms.values()),
        )
        self._counted = 0
        self._period = 0.0
        for name in self._spent:
            self._spent[name] = 0.0

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


__all__ = ["DTYPES", "HttpStageLink", "StageProtocolError", "pack", "unpack"]
