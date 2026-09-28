"""Стадия конвейера: модель, веса и банки экспертов по своему отрезку слоёв.

Зачем это вообще. Пул экспертов Flash-Next в bf16 — 225 ГиБ, и живёт он в
памяти ХОСТА. На одном хосте он не помещается (стенд 2026-09-26: узел с
251.5 ГиБ умер на 45-м слое из 48). Единственное, что делит пул по-настоящему,
— разные хосты, то есть разные отрезки слоёв. TP не делит: он режет по
устройствам, а хост остаётся один.

Здесь проверяется ровно первый шаг: стадия СОБИРАЕТСЯ на своём отрезке и берёт
пропорциональную долю. Шов между стадиями (передача остатка) — следующий шаг.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import clear_stage_info, set_tp_info, try_get_tp_info
from freetoken.engine.config import _parse_layer_range, _stage_view
from freetoken.layers.quantization import QuantKind
from freetoken.models.qwen4_exp import iter_expert_pieces
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.weight import iter_weights
from freetoken.utils import cached_load_hf_config

from .common import QWEN_FP8, hf_config, install_quant_config, meta_state_dict

H, E, I = 128, 3, 6
LAYERS = 4
LM = "model.language_model"


@pytest.fixture(autouse=True)
def _runtime():
    """Отрезок — окружающая настройка, и между тестами она течёт."""
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    clear_stage_info()
    yield
    clear_stage_info()


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Четыре слоя со склеенными bf16-экспертами: два на стадию."""
    folder = tmp_path_factory.mktemp("qwen4_exp_stage")
    raw: dict[str, torch.Tensor] = {
        f"{LM}.embed_tokens.weight": _bf16(11, H),
        "lm_head.weight": _bf16(11, H),
    }
    for layer in range(LAYERS):
        mlp = f"{LM}.layers.{layer}.mlp"
        raw[f"{mlp}.experts.gate_up_proj"] = _bf16(E, 2 * I, H)
        raw[f"{mlp}.experts.down_proj"] = _bf16(E, H, I)
        raw[f"{mlp}.gate.weight"] = _bf16(E, H)
        raw[f"{mlp}.shared_expert.gate_proj.weight"] = _bf16(I, H)
        raw[f"{mlp}.shared_expert.up_proj.weight"] = _bf16(I, H)
        raw[f"{mlp}.shared_expert.down_proj.weight"] = _bf16(H, I)
        raw[f"{mlp}.shared_expert_gate.weight"] = _bf16(1, H)
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    cfg = hf_config(
        num_layers=LAYERS, head_dim=64, num_q=4, num_kv=2, index_head_dim=64, index_heads=2,
        budget=16, hidden=H, max_position=4096, rope_theta=10000.0,
        layer_types=["linear_attention", "full_attention"] * (LAYERS // 2),
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32,
        hc_lowrank=320, ple_layer_ids=[],
        num_experts=E, moe_intermediate_size=I, shared_expert_intermediate_size=I,
    )
    (folder / "config.json").write_text(
        json.dumps({**vars(cfg), "text_config": vars(cfg.text_config), "quantization_config": None})
    )
    return str(folder), raw


def _config(path: str, span: str = ""):
    install_quant_config(path)
    return _stage_view(parse_config(cached_load_hf_config(path)), span)


# ------------------------------------------------------------------ разбор


@pytest.mark.parametrize("text", ["", "0:4"])
def test_the_whole_model_is_the_default(checkpoint, text):
    """Без отрезка и с полным отрезком стадия — это вся модель."""
    path, _raw = checkpoint
    config = _config(path, text)

    assert config.local_layer_ids == (0, 1, 2, 3)
    assert config.owns_first_layer and config.owns_last_layer
    assert config.num_moe_layers == LAYERS


@pytest.mark.parametrize("text", ["2", "a:b", "2:2", "0:5", "-1:2"])
def test_a_bad_range_is_refused(checkpoint, text):
    """Ошибка здесь — это молча собранная полумодель, так что отказ громкий."""
    path, _raw = checkpoint

    with pytest.raises(ValueError, match="layer-range"):
        _config(path, text)


# ------------------------------------------------------------------ конфиг стадии


def test_a_stage_owns_its_layers_only(checkpoint):
    path, _raw = checkpoint

    config = _config(path, "2:4")

    assert config.local_layer_ids == (2, 3)
    assert config.num_layers == LAYERS, "номера слоёв остаются глобальными"
    assert not config.owns_first_layer
    assert config.owns_last_layer


def test_the_expert_pool_splits_by_stage(checkpoint):
    """Ровно та цифра, ради которой всё и делается: банков экспертов у стадии
    столько, сколько у неё слоёв, и память хоста делится вместе с ними."""
    path, _raw = checkpoint

    # Отрезок — окружающая настройка на процесс, так что между замерами её
    # надо забыть: повторная установка ДРУГОГО отрезка это ошибка, и пусть ею
    # и остаётся.
    assert _config(path, "0:2").num_moe_layers == 2
    clear_stage_info()
    assert _config(path, "2:4").num_moe_layers == 2
    clear_stage_info()
    assert _config(path).num_moe_layers == 4


def test_kv_is_sized_for_the_stage(checkpoint):
    """Группы внимания сужаются до своих слоёв, иначе стадия просит KV на всю
    модель — то есть вчетверо больше, чем ей нужно."""
    path, _raw = checkpoint

    whole = {g.name: g.layer_ids for g in _config(path).attention_groups}
    clear_stage_info()
    half = {g.name: g.layer_ids for g in _config(path, "2:4").attention_groups}

    assert sum(len(v) for v in whole.values()) == LAYERS
    assert sum(len(v) for v in half.values()) == 2
    assert all(set(v) <= {2, 3} for v in half.values())


# ------------------------------------------------------------------ сама модель


def test_a_stage_builds_only_its_layers(checkpoint):
    path, _raw = checkpoint
    install_quant_config(path)
    from freetoken.engine.config import EngineConfig
    from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig

    config = EngineConfig(model_path=path, tp_info=try_get_tp_info(), dtype=torch.bfloat16,
                          moe_strategy="offload", layer_range="2:4",
                          mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)))

    assert config.model_config.local_layer_ids == (2, 3)


def test_the_edges_go_to_the_edges(checkpoint):
    """Эмбеддинги у первой стадии, смеситель и lm_head у последней. Иначе
    средняя стадия тащит таблицу словаря, которой не пользуется."""
    path, _raw = checkpoint
    first = {k for k in _stage_dict(path, "0:2")}
    last = {k for k in _stage_dict(path, "2:4")}

    assert any(k.startswith("model.embed_tokens") for k in first)
    assert not any(k.startswith("model.embed_tokens") for k in last)
    assert not any(k.startswith("lm_head") for k in first)
    assert any(k.startswith("lm_head") for k in last)
    assert not any("hyper_connection_mixer" in k for k in first)
    assert any("hyper_connection_mixer" in k for k in last)


def _stage_dict(path: str, span: str) -> dict:
    """State dict модели, собранной движком для этого отрезка."""
    return _stage_model(path, span).state_dict()


def _stage_model(path: str, span: str):
    """Сама модель, собранная движком для этого отрезка, на meta-устройстве."""
    clear_stage_info()
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    install_quant_config(path)
    config = EngineConfig(model_path=path, tp_info=try_get_tp_info(), dtype=torch.bfloat16,
                          moe_strategy="offload", layer_range=span,
                          mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)))
    object.__setattr__(config.model_config, "moe_strategy", "offload")
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            return create_model(config.model_config)
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


@pytest.mark.parametrize("span, first", [("0:2", 0), ("2:4", 2), ("1:3", 1)])
def test_moe_addresses_its_banks_from_the_stage(checkpoint, span, first):
    """Кэш экспертов адресуется индексом банка, а банки нумерованы от стадии.

    Стенд 2026-09-27: оба ранга поднялись, а на первом же запросе хвост
    (24..47) умер в `wait_prefill_layer`. Он просил банк 24 из 24: модель
    отдавала в MoE глобальный номер слоя, а префетч на выходе за набор
    МОЛЧА ничего не делал, и падало сотней строк дальше.
    """
    path, _raw = checkpoint
    model = _stage_model(path, span)
    layers = model.model.layers.op_list
    assert [layer.mlp.experts.layer_id for layer in layers] == list(range(len(layers)))
    # И глобальный номер у слоя при этом свой: банк — не замена ему.
    assert [layer._layer_id for layer in layers] == [first, first + 1]


def test_the_whole_model_numbers_banks_as_before(checkpoint):
    """Без отрезка нумерация та же, что до конвейера: правка не трогает рабочий путь."""
    path, _raw = checkpoint
    layers = _stage_model(path, "").model.layers.op_list
    assert [layer.mlp.experts.layer_id for layer in layers] == [layer._layer_id for layer in layers]
    assert len(layers) == LAYERS


def test_two_stages_are_the_model_cut_not_rebuilt(checkpoint):
    """Ни одного потерянного веса, ни одного лишнего.

    Слои стадии лежат подряд с нуля, поэтому сравнивать можно только вернув
    им глобальные номера. Если после этого объединение равно целой модели и
    пересечение пусто — стадии именно разрезали её.
    """
    from freetoken.models.qwen4_exp.weight import _RENAMED_LAYER_RE

    path, _raw = checkpoint
    whole = set(meta_state_dict(path))
    clear_stage_info()
    first = set(_stage_dict(path, "0:2"))
    clear_stage_info()
    last = set(_stage_dict(path, "2:4"))

    def globally(keys: set[str], by: int) -> set[str]:
        return {
            _RENAMED_LAYER_RE.sub(lambda m: f"{m['head']}{int(m['id']) + by}.", key, count=1)
            for key in keys
        }

    head, tail = globally(first, 0), globally(last, 2)

    assert head & tail == set(), sorted(head & tail)[:8]
    assert head | tail == whole, sorted(whole ^ (head | tail))[:8]


# ------------------------------------------------------------------ веса и эксперты


def test_the_reader_yields_only_this_stage(checkpoint):
    path, _raw = checkpoint
    config = _config(path, "2:4")
    names = {n for n, _ in iter_weights(path, torch.device("cpu"),
                                        include_moe_experts=False, include_non_moe=True)}

    assert names, "стадия осталась без весов вовсе"
    # Ключи приходят с ЛОКАЛЬНЫМ номером: у стадии 2..3 это layers.0 и
    # layers.1, потому что столько их у неё и построено.
    assert any(".layers.0." in n for n in names)
    assert any(".layers.1." in n for n in names)
    assert not any(".layers.2." in n or ".layers.3." in n for n in names)
    assert not any(n.startswith("model.embed_tokens") for n in names)
    assert config.local_layer_ids == (2, 3)


def test_experts_come_from_this_stage_layers(checkpoint):
    """Банк 0 у второй стадии — это глобальный слой 2. Взять по локальному
    счёту значит прочитать чужую половину модели и не заметить."""
    path, raw = checkpoint
    config = _config(path, "2:4")

    pieces = list(iter_expert_pieces(path, config, QuantKind.NONE))

    assert [p[0] for p in pieces] == [0, 1], "банки нумеруются от стадии"
    assert torch.equal(pieces[0][3]["gate_up"], raw[f"{LM}.layers.2.mlp.experts.gate_up_proj"])
    assert torch.equal(pieces[1][3]["gate_up"], raw[f"{LM}.layers.3.mlp.experts.gate_up_proj"])


def test_the_whole_model_still_reads_every_layer(checkpoint):
    path, raw = checkpoint
    config = _config(path)

    pieces = list(iter_expert_pieces(path, config, QuantKind.NONE))

    assert [p[0] for p in pieces] == [0, 1, 2, 3]
    assert torch.equal(pieces[0][3]["gate_up"], raw[f"{LM}.layers.0.mlp.experts.gate_up_proj"])


# ------------------------------------------------------------------ fp8-эксперты

FP8 = torch.float8_e4m3fn


@pytest.fixture(scope="module")
def fp8_checkpoint(tmp_path_factory, checkpoint):
    """Тот же конфиг, но эксперты по одному, как в Qwen/Qwen3.8-Flash-Next-FP8.

    Стенд 2026-09-28: эта модель шла целиком на одной ноде и падала нулевым
    рангом конвейера на собственном слое 0 — FP8-читалка (qwen3_5_moe) считала
    плотный префикс как ``num_layers - num_moe_layers``, а у стадии второе —
    это её 2 слоя из 4, и слой 0 выходил банком −2.
    """
    src, _ = checkpoint
    folder = tmp_path_factory.mktemp("qwen4_exp_stage_fp8")
    raw: dict[str, torch.Tensor] = {}
    for layer in range(LAYERS):
        for e in range(E):
            base = f"{LM}.layers.{layer}.mlp.experts.{e}"
            for proj, shape in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I))):
                raw[f"{base}.{proj}_proj.weight"] = torch.randn(*shape).to(FP8)
                raw[f"{base}.{proj}_proj.weight_scale_inv"] = _bf16(1, 1)
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-fp8-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-fp8-00002.safetensors"))
    cfg = json.loads((__import__("pathlib").Path(src) / "config.json").read_text())
    cfg["quantization_config"] = QWEN_FP8
    (folder / "config.json").write_text(json.dumps(cfg))
    return str(folder), raw


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.parametrize("span, layers", [("0:2", (0, 1)), ("2:4", (2, 3)), ("", (0, 1, 2, 3))],
                         ids=["head", "tail", "whole"])
def test_fp8_experts_come_from_this_stage_layers(fp8_checkpoint, span, layers, parallel):
    """Через точку входа движка: банк от стадии, чужие слои не читаются."""
    from freetoken.moe.expert_pieces import iter_expert_pieces as engine_pieces

    path, raw = fp8_checkpoint
    config = _config(path, span)

    pieces = list(engine_pieces(path, config, QuantKind.FP8_BLOCK, parallel=parallel, workers=2))

    got = sorted((p[0], p[1]) for p in pieces)
    assert got == [(bank, e) for bank in range(len(layers)) for e in range(E)]
    for bank, e, _e1, piece in pieces:
        name = f"{LM}.layers.{layers[bank]}.mlp.experts.{e}.gate_proj.weight"
        assert torch.equal(piece["gate"][0].view(torch.uint8), raw[name].view(torch.uint8))
