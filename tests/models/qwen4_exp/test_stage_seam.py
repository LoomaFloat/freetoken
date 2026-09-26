"""Шов между стадиями: остаток вместо входных токенов, остаток вместо логитов.

Через границу стадии едет ровно одно — остаток ``R [T, hc_count*hidden]``;
между слоями больше не ездит ничего (докстрока модуля модели). Поэтому здесь
проверяется не арифметика слоёв, а МЕСТО РАЗРЕЗА: если разрезать стек в любой
точке и склеить обратно через остаток, ответ обязан совпасть до бита.

Слои для этого подменены детерминированной чистой функцией от входа и номера
слоя: настоящие требуют карты, а шов от того, что они считают, не зависит.
Зато порядок зависит — функция некоммутативна, и перепутанные местами стадии
дали бы другой результат.
"""

from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from freetoken.distributed import clear_stage_info, set_tp_info, try_get_tp_info
from freetoken.engine.config import _stage_view
from freetoken.layers.quantization.configs.base import NoQuantConfig
from freetoken.layers.quantization import set_quant_config
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.model import Qwen4ExpDecoderLayer, Qwen4ExpModel

from .common import fill_weights, toy_hf_config

LAYERS = 4
DIM = 128      # hidden_size у toy_hf_config


@pytest.fixture(autouse=True)
def _runtime():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_quant_config(NoQuantConfig())
    clear_stage_info()
    yield
    clear_stage_info()


@pytest.fixture(autouse=True)
def _no_nvtx(monkeypatch):
    """NVTX-разметка требует сборки torch с CUDA. К месту разреза она
    отношения не имеет, а без неё форвард на CPU не доходит до эмбеддингов."""
    import contextlib

    import torch.cuda.nvtx as nvtx

    monkeypatch.setattr(nvtx, "range", lambda *a, **k: contextlib.nullcontext())


@pytest.fixture(autouse=True)
def _stub_layers(monkeypatch):
    """Слой — чистая функция от входа и своего номера, эмбеддинг — от токена.

    Настоящие требуют карты: слои считают на Triton, а таблица эмбеддингов
    ходит за `tvm_ffi`. Гиперсоединения и смеситель при этом остаются
    настоящими — именно они решают, что везти через границу.
    """
    import freetoken.models.qwen4_exp.model as module

    def layer(self, hidden, batch):
        return torch.tanh(hidden * (1.0 + 0.25 * self._layer_id) + 0.5 * self._layer_id)

    def embed(embed_tokens, input_ids, batch):
        dim = embed_tokens.num_embeddings and DIM
        return torch.sin(input_ids.float()[:, None] * 0.7 + torch.arange(dim, dtype=torch.float32))

    monkeypatch.setattr(Qwen4ExpDecoderLayer, "forward", layer, raising=True)
    monkeypatch.setattr(module, "embed_input_ids", embed, raising=True)


def _config(span: str = ""):
    cfg = parse_config(toy_hf_config(LAYERS, ple_layer_ids=[], num_experts=0))
    return _stage_view(cfg, span)


def _model(span: str = "") -> Qwen4ExpModel:
    """Стадия с заполненными весами смесителя.

    `create_weights` оставляет тензоры неинициализированными, и смеситель на
    мусоре выдаёт NaN — а NaN не равен сам себе, так что сравнение стадий с
    целой моделью «проходило» бы случайно. Заполняется только смеситель: он
    единственный настоящий счётчик в этом тесте, слои подменены.
    """
    model = Qwen4ExpModel(_config(span))
    if model.hyper_connection_mixer is not None:
        fill_weights(model.hyper_connection_mixer, seed=7, device=torch.device("cpu"))
    return model


def _batch(tokens: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.arange(tokens, dtype=torch.long),
        mm_embeds=None, mm_rows=None, fla_metadata=None, stage_hidden=None,
    )


# ------------------------------------------------------------------ шов


def test_two_stages_equal_the_whole_model():
    """Главное утверждение фазы: разрез посередине ничего не меняет."""
    batch = _batch()
    whole = _model().forward(batch.input_ids, batch)

    clear_stage_info()
    head = _model("0:2").forward(batch.input_ids, batch)
    clear_stage_info()
    tail = _model("2:4").forward(batch.input_ids, batch, hidden=head)

    assert torch.equal(tail, whole)


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_the_cut_may_go_anywhere(cut):
    batch = _batch()
    whole = _model().forward(batch.input_ids, batch)

    clear_stage_info()
    head = _model(f"0:{cut}").forward(batch.input_ids, batch)
    clear_stage_info()
    tail = _model(f"{cut}:{LAYERS}").forward(batch.input_ids, batch, hidden=head)

    assert torch.equal(tail, whole)


def test_three_stages_chain():
    """Не только пополам: конвейер длиннее двух звеньев складывается так же."""
    batch = _batch()
    whole = _model().forward(batch.input_ids, batch)

    hidden = None
    for span in ("0:1", "1:3", "3:4"):
        clear_stage_info()
        model = _model(span)
        hidden = model.forward(batch.input_ids, batch, hidden=hidden)

    assert torch.equal(hidden, whole)


def test_the_order_of_stages_matters():
    """Защита от теста, который прошёл бы и на перепутанных стадиях."""
    batch = _batch()
    whole = _model().forward(batch.input_ids, batch)

    clear_stage_info()
    head = _model("0:2").forward(batch.input_ids, batch)
    clear_stage_info()
    # Тот же остаток, но во вторую стадию отданы её же слои дважды.
    wrong = _model("0:2").forward(batch.input_ids, batch, hidden=head)

    assert not torch.equal(wrong, whole)


# ------------------------------------------------------------------ форма остатка


def test_the_intermediate_stage_does_not_collapse_the_streams():
    """Смеситель собирает четыре потока в один и стоит только у последней.
    Собрав их раньше, следующая стадия получила бы вход вчетверо уже."""
    batch = _batch()

    clear_stage_info()
    head = _model("0:2").forward(batch.input_ids, batch)
    clear_stage_info()
    whole = _model().forward(batch.input_ids, batch)

    config = _config()
    assert head.shape[-1] == config.qwen4_args.hc_count * config.hidden_size
    assert whole.shape[-1] == config.hidden_size


# ------------------------------------------------------------------ отказы


def test_a_middle_stage_without_the_remainder_refuses():
    """Иначе стадия без эмбеддингов упала бы где-то внутри, с невнятным
    AttributeError вместо причины."""
    batch = _batch()
    clear_stage_info()
    model = _model("2:4")

    with pytest.raises(ValueError, match="stage_hidden"):
        model.forward(batch.input_ids, batch)


def test_a_remainder_of_the_wrong_width_refuses():
    """Разойдясь в ширине потока, стадии молча испортили бы ответ."""
    batch = _batch()
    clear_stage_info()
    model = _model("2:4")

    with pytest.raises(ValueError, match="ожидалось"):
        model.forward(batch.input_ids, batch, hidden=torch.zeros(3, 7))


# ------------------------------------------------------------------ PLE


def _ple_model(span: str) -> Qwen4ExpModel:
    """Стадия модели, у которой таблица n-грамм висит на слое 1."""
    cfg = parse_config(toy_hf_config(LAYERS, ple_layer_ids=[1], num_experts=0))
    return Qwen4ExpModel(_stage_view(cfg, span))


def test_the_ngram_table_goes_to_the_stage_that_owns_its_layer():
    """Таблица n-грамм в bf16 — 95.4 ГиБ, и висит она на слое 1. Достаться
    она должна ровно той стадии, у которой этот слой: `load_host_tables`
    смотрит на `ple_layers`, и у остальных стадий их быть не должно."""
    head = _ple_model("0:2")
    clear_stage_info()
    tail = _ple_model("2:4")

    assert [ple.args is not None for ple in head.ple_layers] == [True]
    assert tail.ple_layers == []
