"""Веса стадии конвейера: какие тензоры ей нужны и под какими номерами слоёв.

Общее для всех семейств, которые умеют стадию (`ModelSpec.supports_stages`).
Семейства отличаются только именами краёв: что строит лишь первая стадия
(эмбеддинги, у мультимодальных — и башня, которая их кормит) и что лишь
последняя (финальная норма или смеситель, `lm_head`).
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from freetoken.distributed import get_stage_info

#: Номер слоя в УЖЕ переименованном ключе (`model.layers.7.…`).
LAYER_KEY_RE = re.compile(r"(?P<head>(?:^|\.)layers\.)(?P<id>\d+)\.")


def stage_keeps(
    total_layers: int,
    *,
    first_only: Sequence[str] = (),
    last_only: Sequence[str] = (),
) -> Optional[Callable[[str], bool]]:
    """Фильтр «это моё» по переименованным ключам; None — стадия одна.

    Слой берётся, если он из отрезка стадии. Ключ без номера слоя с префиксом из
    `first_only` нужен только первой стадии, из `last_only` — только последней;
    остальное без номера — всем. Если движок построил то, чей вес здесь
    отброшен, `load_state_dict` скажет об этом вслух, а не соберёт молча
    полумодель.
    """
    stage = get_stage_info(total_layers)
    if stage.whole:
        return None
    first, last = tuple(first_only), tuple(last_only)

    def keep(name: str) -> bool:
        match = LAYER_KEY_RE.search(name)
        if match is not None:
            return stage.owns(int(match["id"]))
        if first and name.startswith(first):
            return stage.is_first
        if last and name.startswith(last):
            return stage.is_last
        return True

    return keep


def stage_renumber(total_layers: int) -> Optional[Callable[[str], str]]:
    """Глобальный номер слоя -> номер внутри стадии; None — стадия одна.

    Стадия держит свои слои подряд с нуля: ``OPList`` нумерует их по порядку,
    и у стадии со слоями 2..3 в state dict лежат ``layers.0`` и ``layers.1``.
    Переименовывать надо в самом конце, после слияния проекций: квантовые
    схемы ищутся по номеру ИЗ ЧЕКПОИНТА (у смешанной точности они заданы
    послойно). А читалке банков экспертов номер нужен глобальный — банк ей
    назовёт `bank_layer_of`.
    """
    stage = get_stage_info(total_layers)
    if stage.whole:
        return None

    def local(name: str) -> str:
        return LAYER_KEY_RE.sub(
            lambda m: f"{m['head']}{int(m['id']) - stage.first}.", name, count=1
        )

    return local


__all__ = ["LAYER_KEY_RE", "stage_keeps", "stage_renumber"]
