from __future__ import annotations

import re
from dataclasses import dataclass

#: Номер слоя в имени тензора чекпоинта: `model.language_model.layers.7.…`,
#: `model.layers.7.…`, `mtp.layers.0.…` — все пишут его одинаково.
_LAYER_IN_KEY = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


@dataclass(frozen=True)
class StageInfo:
    """Какой отрезок слоёв держит этот процесс: ``[first, last)`` из ``total``.

    Окружающая настройка, а не параметр: диапазон нужен и построителю модели,
    и читалке весов, и читалке экспертов — тем же способом, каким план
    резидентности доходит до всех загрузчиков, «without a new parameter in
    each signature».
    """

    first: int
    last: int
    total: int

    def __post_init__(self):
        assert 0 <= self.first < self.last <= self.total, (self.first, self.last, self.total)

    @property
    def whole(self) -> bool:
        return self.first == 0 and self.last == self.total

    @property
    def is_first(self) -> bool:
        return self.first == 0

    @property
    def is_last(self) -> bool:
        return self.last == self.total

    def owns(self, layer_id: int) -> bool:
        return self.first <= layer_id < self.last

    def owns_key(self, name: str) -> bool:
        """Нужен ли этой стадии тензор с таким именем в чекпоинте.

        Правило нарочно осторожное: отбрасывается только то, что ТОЧНО
        принадлежит слоям другой стадии. Всё, чей слой из имени не
        определяется — эмбеддинги, голова, башня зрения, — остаётся. Они
        малы рядом со слоями, а ошибка в другую сторону — это модель,
        которая молча стартует на половине весов.
        """
        found = _LAYER_IN_KEY.search(name)
        return self.owns(int(found.group(1))) if found else True


_STAGE: StageInfo | None = None


def set_stage_info(first: int, last: int, total: int) -> None:
    """Задать отрезок слоёв процесса. Повторный вызов с тем же отрезком — не ошибка."""
    global _STAGE
    stage = StageInfo(first, last, total)
    if _STAGE is not None and _STAGE != stage:
        raise RuntimeError(f"stage info has been set to {_STAGE}, refusing {stage}")
    _STAGE = stage


def get_stage_info(total: int) -> StageInfo:
    """Отрезок процесса, а без него — вся модель из ``total`` слоёв."""
    return _STAGE if _STAGE is not None else StageInfo(0, total, total)


def try_get_stage_info() -> StageInfo | None:
    return _STAGE


def clear_stage_info() -> None:
    """Только для тестов: настройка окружающая, и между прогонами течёт."""
    global _STAGE
    _STAGE = None


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info",
           "StageInfo", "set_stage_info", "get_stage_info", "try_get_stage_info",
           "clear_stage_info"]
