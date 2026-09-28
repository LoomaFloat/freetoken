"""Стадия конвейера у Qwen3-MoE: свой отрезок слоёв, свои края, свои веса.

Стенд 2026-09-28: Qwen3-30B-A3B на двух узлах упал на обоих. Движок уже был
стадийным (KV, число банков, скачивание), а семейство строило все слои: хвост
не нашёл в скачанном свой же слой 2, голова споткнулась о чужой слой 24 в
пограничном шарде экспертов. Здесь — что стадия строит, что читает и что везёт
через шов, и что семейства без поддержки отказывают сразу.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import clear_stage_info, set_tp_info, try_get_stage_info, try_get_tp_info

H, HEADS, KV, HD, VOCAB, E, I = 64, 4, 2, 64, 128, 4, 32
LAYERS = 4


@pytest.fixture(autouse=True)
def _runtime():
    """Отрезок — окружающая настройка, и между тестами она течёт."""
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    clear_stage_info()
    yield
    clear_stage_info()


def _config(**overrides) -> dict:
    config = {
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "hidden_size": H, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV, "head_dim": HD, "vocab_size": VOCAB,
        "intermediate_size": 128, "moe_intermediate_size": I, "num_experts": E,
        "num_experts_per_tok": 2, "norm_topk_prob": True, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "rope_theta": 1_000_000.0,
        "tie_word_embeddings": False, "torch_dtype": "bfloat16",
    }
    config.update(overrides)
    return config


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _layer(n: int) -> dict[str, torch.Tensor]:
    p = f"model.layers.{n}"
    out = {
        f"{p}.self_attn.q_proj.weight": _bf16(HEADS * HD, H),
        f"{p}.self_attn.k_proj.weight": _bf16(KV * HD, H),
        f"{p}.self_attn.v_proj.weight": _bf16(KV * HD, H),
        f"{p}.self_attn.o_proj.weight": _bf16(H, HEADS * HD),
        f"{p}.self_attn.q_norm.weight": _bf16(HD),
        f"{p}.self_attn.k_norm.weight": _bf16(HD),
        f"{p}.input_layernorm.weight": _bf16(H),
        f"{p}.post_attention_layernorm.weight": _bf16(H),
        f"{p}.mlp.gate.weight": _bf16(E, H),
    }
    for e in range(E):
        out[f"{p}.mlp.experts.{e}.gate_proj.weight"] = _bf16(I, H)
        out[f"{p}.mlp.experts.{e}.up_proj.weight"] = _bf16(I, H)
        out[f"{p}.mlp.experts.{e}.down_proj.weight"] = _bf16(H, I)
    return out


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Четыре слоя в двух шардах, и первый ПЕРЕСЕКАЕТ границу стадий 0:2 | 2:4.

    Так и было на стенде: шард со слоями по обе стороны разреза скачивают обе
    стадии, и каждая обязана взять из него только своё.
    """
    folder = tmp_path_factory.mktemp("qwen3_moe_stage")
    first = {"model.embed_tokens.weight": _bf16(VOCAB, H), **_layer(0), **_layer(1), **_layer(2)}
    second = {**_layer(3), "model.norm.weight": _bf16(H), "lm_head.weight": _bf16(VOCAB, H)}
    save_file(first, str(folder / "model-00001-of-00002.safetensors"))
    save_file(second, str(folder / "model-00002-of-00002.safetensors"))
    weight_map = {name: "model-00001-of-00002.safetensors" for name in first}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in second})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (folder / "config.json").write_text(json.dumps(_config()))
    return str(folder)


def _engine(path: str, span: str = ""):
    from freetoken.engine.config import EngineConfig

    return EngineConfig(model_path=path, tp_info=try_get_tp_info(), dtype=torch.bfloat16,
                        moe_strategy="offload", layer_range=span)


def _model(path: str, span: str = ""):
    """Модель, собранная движком для этого отрезка, на meta-устройстве."""
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    clear_stage_info()
    config = _engine(path, span)
    object.__setattr__(config.model_config, "moe_strategy", "offload")
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            return create_model(config.model_config), config.model_config
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


def _layer_ids(keys) -> set[int]:
    return {int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")}


# ------------------------------------------------------------------ охранник


def test_a_family_without_stages_is_refused_before_anything_loads(tmp_path):
    """Семейство, не объявившее стадию, отказывает на конфиге — до весов.

    Иначе движок сузит KV, банки и скачивание, а модель соберёт целиком, и
    упадёт это через минуты и гигабайты, в непонятном месте.
    """
    (tmp_path / "config.json").write_text(json.dumps(_config(
        architectures=["Qwen3ForCausalLM"], model_type="qwen3")))
    with pytest.raises(ValueError, match="не умеет стадию"):
        _engine(str(tmp_path), "0:2").model_config
    assert try_get_stage_info() is None, "отказ должен прийти раньше, чем стадия задана"


def test_the_same_family_without_a_range_is_not_refused(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_config(
        architectures=["Qwen3ForCausalLM"], model_type="qwen3")))
    assert _engine(str(tmp_path)).model_config.num_layers == LAYERS


# ------------------------------------------------------------------ модель


def test_the_whole_model_builds_as_before(checkpoint):
    model, _ = _model(checkpoint)
    keys = set(model.state_dict())
    assert _layer_ids(keys) == set(range(LAYERS))
    assert {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} <= keys
    layers = model.model.layers.op_list
    assert [layer.mlp.experts.layer_id for layer in layers] == list(range(LAYERS))
    assert model.produces_logits and model.stage_input_width == 0


@pytest.mark.parametrize("span, embed, head", [("0:2", True, False), ("2:4", False, True), ("1:3", False, False)])
def test_a_stage_builds_its_layers_and_its_edges(checkpoint, span, embed, head):
    model, _ = _model(checkpoint, span)
    keys = set(model.state_dict())
    assert _layer_ids(keys) == {0, 1}, "слои стадии лежат подряд с нуля"
    assert ("model.embed_tokens.weight" in keys) is embed
    assert ("model.norm.weight" in keys) is head and ("lm_head.weight" in keys) is head
    assert model.produces_logits is head
    assert model.stage_input_width == (0 if embed else 2 * H)


def test_two_stages_are_the_model_cut_not_rebuilt(checkpoint):
    """Ни одного потерянного веса, ни одного лишнего."""
    from freetoken.models.stage_weights import LAYER_KEY_RE

    whole = set(_model(checkpoint)[0].state_dict())
    first = set(_model(checkpoint, "0:2")[0].state_dict())
    last = set(_model(checkpoint, "2:4")[0].state_dict())

    def globally(keys: set[str], by: int) -> set[str]:
        return {LAYER_KEY_RE.sub(lambda m: f"{m['head']}{int(m['id']) + by}.", k, count=1) for k in keys}

    first, last = globally(first, 0), globally(last, 2)
    assert first | last == whole and not first & last


@pytest.mark.parametrize("span", ["0:2", "2:4"])
def test_moe_addresses_its_banks_from_the_stage(checkpoint, span):
    """Кэш экспертов адресуется индексом банка; тот же баг стоил Flash-Next стенда."""
    model, _ = _model(checkpoint, span)
    layers = model.model.layers.op_list
    assert [layer.mlp.experts.layer_id for layer in layers] == [0, 1]
    first = int(span.split(":")[0])
    assert [layer._layer_id for layer in layers] == [first, first + 1]


def test_tied_embeddings_are_refused_on_a_later_last_stage(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_config(tie_word_embeddings=True)))
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        _model(str(tmp_path), "2:4")
    _model(str(tmp_path), "0:2")          # у первой стадии lm_head нет — отказывать не в чем


def test_kv_covers_only_the_stage_layers(checkpoint):
    """У модели без групп внимания KV считался бы на все слои на каждой стадии."""
    assert _engine(checkpoint, "2:4").model_config.kv_cache_group_specs()[0].layer_ids == (2, 3)
    clear_stage_info()
    assert _engine(checkpoint).model_config.kv_cache_group_specs()[0].layer_ids == (0, 1, 2, 3)


# ------------------------------------------------------------------ веса


@pytest.mark.parametrize("span", ["", "0:2", "2:4", "1:3"])
def test_the_reader_feeds_exactly_what_the_stage_built(checkpoint, span):
    """Плотная читалка отдаёт ровно ключи построенной модели — ни больше, ни меньше.

    Хвост на стенде упал ровно на этом: модель ждала слой, которого в
    скачанном не было. Лишний ключ `load_state_dict` тоже не простит.
    """
    from freetoken.models.qwen3_moe.weight import iter_weights

    model, _ = _model(checkpoint, span)
    built = set(model.state_dict())
    read = {name for name, _ in iter_weights(
        checkpoint, torch.device("cpu"), include_moe_experts=False, include_non_moe=True)}
    assert read == built


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("span, layers", [("0:2", {0, 1}), ("2:4", {2, 3})])
def test_the_expert_reader_takes_only_this_stage(checkpoint, span, layers, parallel):
    """Эксперты — только своих слоёв, под глобальными номерами, в банки стадии.

    Голова на стенде упала здесь: пограничный шард отдал слой 24, и общая
    читалка банков не нашла ему места.
    """
    from freetoken.models.qwen3_moe import weight
    from freetoken.moe.expert_pieces import stacked_expert_pieces

    _, config = _model(checkpoint, span)
    read = weight.iter_weights_parallel if parallel else weight.iter_weights
    tensors = list(read(checkpoint, torch.device("cpu"), include_moe_experts=True, include_non_moe=False))
    assert _layer_ids(name for name, _ in tensors) == layers, "номера у банков — глобальные"
    banks = sorted(bank for bank, _, _, _ in stacked_expert_pieces(iter(tensors), config))
    assert banks == [0, 1]


# ------------------------------------------------------------------ шов


class _Layer:
    """Слой-заменитель с тем же контрактом, что у настоящего: отложенный остаток.

    Настоящие слои требуют ядер; шов от того, что они считают, не зависит.
    Сложение `x + residual` — то, что делает слитая норма следующего слоя.
    """

    def __init__(self, global_id: int):
        self.scale = 1.0 + 0.1 * global_id

    def forward(self, x, residual=None):
        stream = x if residual is None else x + residual
        return torch.tanh(stream * self.scale), stream


class _Norm:
    def forward(self, x, residual):
        return ((x + residual) * 2.0,)


class _Embed:
    def __init__(self, table):
        self.table = table

    def forward(self, input_ids):
        return self.table[input_ids]


def _stand_in(model, first: int, table):
    inner = model.model
    inner.layers.op_list[:] = [_Layer(first + i) for i in range(len(inner.layers.op_list))]
    if inner.embed_tokens is not None:
        inner.embed_tokens = _Embed(table)
    if inner.norm is not None:
        inner.norm = _Norm()
    return inner


def test_the_seam_carries_x_and_residual_and_the_cut_changes_nothing(checkpoint):
    """Голова и хвост через шов дают побитово то же, что целая модель."""
    table = torch.randn(VOCAB, H)
    ids = torch.tensor([3, 17, 42, 99])

    whole = _stand_in(_model(checkpoint)[0], 0, table).forward(ids)
    head = _stand_in(_model(checkpoint, "0:2")[0], 0, table)
    tail = _stand_in(_model(checkpoint, "2:4")[0], 2, table)

    seam = head.forward(ids)
    assert seam.shape == (4, 2 * H), "через шов едет пара (x, residual)"
    assert torch.equal(tail.forward(ids, hidden=seam), whole)


def test_the_seam_refuses_what_it_cannot_continue(checkpoint):
    tail = _stand_in(_model(checkpoint, "2:4")[0], 2, torch.randn(VOCAB, H))
    ids = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="не получила остаток"):
        tail.forward(ids)
    with pytest.raises(ValueError, match="шириной"):
        tail.forward(ids, hidden=torch.zeros(2, H))


def test_a_model_without_a_hidden_argument_still_runs_whole(checkpoint, monkeypatch):
    """Qwen3-VL подменяет текстовую модель на свою с `forward(input_ids)`.

    Целая модель должна звать её прежним вызовом, без аргумента для остатка,
    иначе правка ради конвейера сломает соседнее семейство.
    """
    import freetoken.models.qwen3_moe.model as family

    model, _ = _model(checkpoint)

    class OldStyle:
        def forward(self, input_ids):
            return torch.ones(len(input_ids), H)

    model.model = OldStyle()

    class Head:
        def forward(self, x):
            return x.sum(-1)

    model.lm_head = Head()
    batch = SimpleNamespace(input_ids=torch.tensor([1, 2, 3]), stage_hidden=None)
    monkeypatch.setattr(family, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    assert torch.equal(model.forward(), torch.full((3,), float(H)))
