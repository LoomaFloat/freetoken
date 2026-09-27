"""Связь между стадиями конвейера: остаток вперёд, выбранный токен назад.

Зачем конвейер вообще. Пул экспертов Flash-Next в bf16 — 225 ГиБ, и живёт он
в памяти ХОСТА. На одном хосте он не помещается (стенд 2026-09-26: узел с
251.5 ГиБ умер на 45-м слое из 48). Делит пул только другой хост, то есть
другой отрезок слоёв; TP не делит — он режет по устройствам, а хост один.

Как устроен шов. Стадии гоняют ОБЫЧНЫЙ движок целиком: свой планировщик, свой
KV, свои рекуррентные состояния. Связывает их ровно две вещи:

* **остаток** ``R [T, hc_count*hidden]`` едет со стадии на следующую. Это
  единственное, что пересекает границу: между слоями больше не ездит ничего,
  и на декоде это 20 КБ на токен против 4.7 ГБ, которые пришлось бы возить,
  если бы по сети ездили веса;
* **токен**, выбранный ПОСЛЕДНЕЙ стадией, возвращается всем. Только он решает,
  что подать на вход следующим шагом; без него стадии разойдутся уже на
  втором токене, и разойдутся молча.

Здесь описан только шов. Чем возить — петлёй в одном процессе, сокетом, нашим
релеем — дело того, кто подставит реализацию.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

from freetoken.core import Batch


class StageLink(ABC):
    """Соседи этой стадии по конвейеру.

    Все четыре действия синхронные: стадия не может считать следующий шаг, не
    зная токена предыдущего. Это и есть цена конвейера — один круговой обмен
    на токен, порядка RTT.
    """

    @abstractmethod
    def take(self, batch: Batch) -> torch.Tensor:
        """Остаток предыдущей стадии для этого шага, ``[T, hc_count*hidden]``."""

    @abstractmethod
    def give(self, batch: Batch, remainder: torch.Tensor) -> None:
        """Отдать свой остаток следующей стадии."""

    @abstractmethod
    def tokens(self, batch: Batch) -> torch.Tensor:
        """Дождаться токенов, выбранных последней стадией: ``[batch.size]`` int32."""

    @abstractmethod
    def publish(self, batch: Batch, tokens: torch.Tensor) -> None:
        """Разослать выбранные токены назад по цепочке (делает последняя)."""


def needs_incoming(model_config) -> bool:
    """Ждёт ли эта стадия остаток от предыдущей."""
    return not model_config.owns_first_layer


def graphs_allowed(model_config) -> bool:
    """Можно ли захватывать CUDA-графы.

    На стадии — пока нет, и причин две. Вход не-первой стадии приезжает
    снаружи, а граф требует буфера по фиксированному адресу. А выход
    не-последней — это остаток, а захват пишет выход в буфер ЛОГИТОВ: стенд
    2026-09-27, `expanded size (248320) must match existing size (10240)` —
    ширина словаря против ширины остатка.

    Сделать графы на стадии можно (дисковая таблица PLE ровно так и
    синхронизируется через флаг и `memop_wait`), но это отдельная работа. До
    неё стадия считает в eager: медленнее, но не молча неправильно.
    """
    return getattr(model_config, "layer_range", None) is None


def graph_batch_sizes(model_config, requested):
    """Размеры батча для захвата графов; пустой список — захвата нет.

    Отдельной функцией, потому что охранник должен быть ОДИН. Со стенда
    2026-09-27: `graphs_allowed` стояла только в `forward_batch`, а захват
    идёт при сборке движка — и стадия падала на захвате, не дожив до первого
    запроса. Решение о графах принимается здесь, а зовётся там, где
    строится GraphRunner.
    """
    return requested if graphs_allowed(model_config) else []


def stage_input(batch: Batch, *, link: StageLink | None, model_config) -> None:
    """Положить в батч остаток предыдущей стадии, если он этой стадии нужен."""
    if not needs_incoming(model_config):
        return
    if link is None:
        raise RuntimeError(
            "стадия начинается не с нулевого слоя, но связи с предыдущей нет: "
            "считать было бы не из чего"
        )
    batch.stage_hidden = link.take(batch)


def stage_output(
    batch: Batch,
    out: torch.Tensor,
    *,
    link: StageLink | None,
    produces_logits: bool,
    sample: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Что стадия делает с результатом своего forward.

    Последняя сэмплирует сама и рассылает токены назад. Остальные отдают
    остаток дальше и ждут, что выберет хвост: сэмплировать из остатка —
    молча выдавать мусор, поэтому решает именно `produces_logits`, а не догадка
    по форме тензора.
    """
    if not produces_logits:
        if link is None:
            raise RuntimeError(
                "стадия не считает логиты, но отдать остаток некому: "
                "без связи с следующей стадией ответа не будет"
            )
        link.give(batch, out)
        return link.tokens(batch)
    tokens = sample(out)
    if link is not None:
        link.publish(batch, tokens)
    return tokens


__all__ = ["StageLink", "attach_stage_link", "graph_batch_sizes", "graphs_allowed",
           "needs_incoming", "stage_input", "stage_output"]


def attach_stage_link(engine, config) -> None:
    """Дать движку связь с соседями, если он стадия конвейера.

    Отказ здесь, а не на первом токене. Стадия с отрезком слоёв, но без связи
    доходит до первого forward и падает там — через минуты после старта и
    после того, как веса уже в памяти. Дешевле сказать сразу.
    """
    span = getattr(config, "layer_range", "")
    url = getattr(config, "stage_send_url", "")
    if not url:
        if span:
            raise ValueError(
                "--layer-range задан, но --stage-send-url нет: стадии не с кем "
                "обмениваться остатком, и ответа не будет"
            )
        return
    if not span:
        raise ValueError(
            "--stage-send-url задан, но --layer-range нет: целая модель "
            "соседей не имеет, и связь была бы ни к чему"
        )
    from .stage_http import HttpStageLink

    engine.stage_link = HttpStageLink(
        send_url=url,
        listen_port=int(getattr(config, "stage_listen_port", 0) or 0),
        rank=int(getattr(config, "stage_rank", 0) or 0),
        size=int(getattr(config, "stage_size", 1) or 1),
        device=engine.device,
    )
