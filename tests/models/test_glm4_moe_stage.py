"""GLM-4 MoE (glm4_moe): the bf16 release (GLM-4.5-Air) and pipeline stages.

The family was written for nvidia/GLM-4.7-NVFP4: NVFP4 routed experts, NVFP4 dense MLP
and shared experts. GLM-4.5-Air ships all of it in bf16, so the reader has to take plain
weights too, and the routed experts come in one tensor per projection per expert. On top
of that a stage builds and reads only its own layers; the seam is the deferred
(x, residual) pair, like Qwen3-MoE's.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import clear_stage_info, set_tp_info, try_get_tp_info
from freetoken.utils.hf import RawConfigShim

LAYERS, HIDDEN, HEADS, KV, HEAD_DIM = 6, 64, 4, 2, 64
VOCAB, EXPERTS, INTER, MOE_INTER = 128, 4, 96, 32
FIRST_DENSE = 1


def _hf_config() -> RawConfigShim:
    return RawConfigShim({
        "architectures": ["Glm4MoeForCausalLM"], "model_type": "glm4_moe",
        "hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV, "head_dim": HEAD_DIM, "vocab_size": VOCAB,
        "intermediate_size": INTER, "moe_intermediate_size": MOE_INTER,
        "n_routed_experts": EXPERTS, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "first_k_dense_replace": FIRST_DENSE, "norm_topk_prob": True,
        "routed_scaling_factor": 1.0, "n_group": 1, "topk_group": 1,
        "hidden_act": "silu", "rms_norm_eps": 1e-5, "max_position_embeddings": 4096,
        "partial_rotary_factor": 0.5, "rope_theta": 1_000_000.0, "attention_bias": True,
        "use_qk_norm": False, "tie_word_embeddings": False, "num_nextn_predict_layers": 1,
    })


def _bf16(*shape):
    return torch.randn(*shape).to(torch.bfloat16)


def _layer(n: int) -> dict[str, torch.Tensor]:
    p = f"model.layers.{n}"
    out = {
        f"{p}.self_attn.q_proj.weight": _bf16(HEADS * HEAD_DIM, HIDDEN),
        f"{p}.self_attn.q_proj.bias": _bf16(HEADS * HEAD_DIM),
        f"{p}.self_attn.k_proj.weight": _bf16(KV * HEAD_DIM, HIDDEN),
        f"{p}.self_attn.k_proj.bias": _bf16(KV * HEAD_DIM),
        f"{p}.self_attn.v_proj.weight": _bf16(KV * HEAD_DIM, HIDDEN),
        f"{p}.self_attn.v_proj.bias": _bf16(KV * HEAD_DIM),
        f"{p}.self_attn.o_proj.weight": _bf16(HIDDEN, HEADS * HEAD_DIM),
        f"{p}.input_layernorm.weight": _bf16(HIDDEN),
        f"{p}.post_attention_layernorm.weight": _bf16(HIDDEN),
    }
    if n < FIRST_DENSE:
        out.update({f"{p}.mlp.gate_proj.weight": _bf16(INTER, HIDDEN),
                    f"{p}.mlp.up_proj.weight": _bf16(INTER, HIDDEN),
                    f"{p}.mlp.down_proj.weight": _bf16(HIDDEN, INTER)})
        return out
    out.update({f"{p}.mlp.gate.weight": _bf16(EXPERTS, HIDDEN),
                f"{p}.mlp.gate.e_score_correction_bias": torch.randn(EXPERTS),
                f"{p}.mlp.shared_experts.gate_proj.weight": _bf16(MOE_INTER, HIDDEN),
                f"{p}.mlp.shared_experts.up_proj.weight": _bf16(MOE_INTER, HIDDEN),
                f"{p}.mlp.shared_experts.down_proj.weight": _bf16(HIDDEN, MOE_INTER)})
    for e in range(EXPERTS):
        out[f"{p}.mlp.experts.{e}.gate_proj.weight"] = _bf16(MOE_INTER, HIDDEN)
        out[f"{p}.mlp.experts.{e}.up_proj.weight"] = _bf16(MOE_INTER, HIDDEN)
        out[f"{p}.mlp.experts.{e}.down_proj.weight"] = _bf16(HIDDEN, MOE_INTER)
    return out


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Air-shaped bf16 checkpoint, two shards, the second one starting mid-model, plus
    the MTP layer (index num_layers) that must never be read."""
    folder = tmp_path_factory.mktemp("glm4_moe_bf16")
    first = {"model.embed_tokens.weight": _bf16(VOCAB, HIDDEN)}
    for n in range(3):
        first.update(_layer(n))
    second = {"model.norm.weight": _bf16(HIDDEN), "lm_head.weight": _bf16(VOCAB, HIDDEN)}
    for n in range(3, LAYERS + 1):
        second.update(_layer(n))
    save_file(first, str(folder / "model-00001-of-00002.safetensors"))
    save_file(second, str(folder / "model-00002-of-00002.safetensors"))
    weight_map = {k: "model-00001-of-00002.safetensors" for k in first}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in second})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return str(folder), {**first, **second}


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    import freetoken.engine.config as engine_config
    import freetoken.models.glm4_moe.weight as weight

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    shim = _hf_config()
    monkeypatch.setattr(engine_config, "cached_load_hf_config", lambda path: shim)
    monkeypatch.setattr(engine_config, "checkpoint_quant_config", lambda *args: None)
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda path: shim)
    clear_stage_info()
    yield
    clear_stage_info()


def _model(span: str = "", path: str = "/fake"):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    clear_stage_info()
    config = EngineConfig(model_path=path, tp_info=DistributedInfo(rank=0, size=1),
                          dtype=torch.bfloat16, moe_strategy="offload", layer_range=span)
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


def _banks(model) -> list[int]:
    return [layer.mlp.experts.layer_id for layer in model.model.layers.op_list
            if hasattr(layer.mlp, "experts")]


# ------------------------------------------------------------------ model


def test_the_whole_model_builds_as_before():
    model, _ = _model()
    keys = set(model.state_dict())
    assert _layer_ids(keys) == set(range(LAYERS))
    assert "model.norm.weight" in keys and "lm_head.weight" in keys
    assert any(k.startswith("model.embed_tokens.") for k in keys)
    assert _banks(model) == list(range(LAYERS - FIRST_DENSE))
    assert model.produces_logits and model.stage_input_width == 0


@pytest.mark.parametrize("span, embed, head, banks", [
    ("0:3", True, False, [0, 1]),        # layer 0 is dense
    ("3:6", False, True, [0, 1, 2]),
    ("2:4", False, False, [0, 1]),
])
def test_a_stage_builds_its_layers_and_its_edges(span, embed, head, banks):
    model, config = _model(span)
    keys = set(model.state_dict())
    first, last = map(int, span.split(":"))
    assert _layer_ids(keys) == set(range(last - first))
    assert any(k.startswith("model.embed_tokens.") for k in keys) is embed
    assert ("model.norm.weight" in keys) is head and ("lm_head.weight" in keys) is head
    assert model.produces_logits is head
    assert model.stage_input_width == (0 if embed else 2 * HIDDEN)
    assert _banks(model) == banks and config.num_moe_layers == len(banks)


def test_two_stages_are_the_model_cut_not_rebuilt():
    from freetoken.models.stage_weights import LAYER_KEY_RE

    whole = set(_model()[0].state_dict())
    first = set(_model("0:3")[0].state_dict())
    last = set(_model("3:6")[0].state_dict())
    shift = lambda keys, by: {LAYER_KEY_RE.sub(lambda m: f"{m['head']}{int(m['id']) + by}.", k, count=1)
                              for k in keys}
    assert shift(first, 0) | shift(last, 3) == whole and not shift(first, 0) & shift(last, 3)


def test_kv_covers_only_the_stage_layers():
    assert _model("3:6")[1].kv_cache_group_specs()[0].layer_ids == (3, 4, 5)


# ------------------------------------------------------------------ weights


@pytest.mark.parametrize("span", ["", "0:3", "3:6", "2:4"])
def test_the_bf16_dense_reader_feeds_exactly_what_the_stage_built(checkpoint, span):
    """GLM-4.5-Air's dense MLP and shared experts are plain bf16; the reader used to ask
    for NVFP4 scales there. A stage reads only its own layers and edges."""
    from freetoken.models.glm4_moe.weight import iter_weights

    path, _ = checkpoint
    model, _ = _model(span, path)
    built = set(model.state_dict())
    read = {name for name, _ in iter_weights(path, torch.device("cpu"),
                                             include_moe_experts=False, include_non_moe=True)}
    assert read == built


@pytest.mark.parametrize("span, layers", [("", {1, 2, 3, 4, 5}), ("0:3", {1, 2}), ("3:6", {3, 4, 5})])
def test_the_bf16_expert_reader_takes_this_stage_and_never_the_mtp_layer(checkpoint, span, layers):
    import freetoken.models.glm4_moe.weight as weight
    from freetoken.layers.quantization import QuantKind

    path, tensors = checkpoint
    _, config = _model(span, path)
    pieces = list(weight.iter_expert_pieces(path, config, QuantKind.NONE, parallel=False))
    first_moe = min(layers)
    assert {(bank, e0) for bank, e0, e1, _ in pieces} == {
        (layer - first_moe, e) for layer in layers for e in range(EXPERTS)}
    bank, e0, e1, piece = next(p for p in pieces if p[0] == 0 and p[1] == 1)
    for proj in ("gate", "up", "down"):
        assert torch.equal(piece[proj][0], tensors[f"model.layers.{first_moe}.mlp.experts.1.{proj}_proj.weight"])


@pytest.mark.parametrize("parallel", [False, None])
def test_the_engine_entry_reaches_the_family_reader(checkpoint, parallel):
    """The engine looks the hook up on the family PACKAGE, not on weight.py. Stand
    2026-09-28: it was not exported, the engine fell through to the generic bf16 path and
    died on iter_weights' assert, after half an hour of loading. Called directly the
    reader worked, which is why only this entry point catches it."""
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_pieces import iter_expert_pieces

    path, _ = checkpoint
    _, config = _model("", path)
    pieces = list(iter_expert_pieces(path, config, QuantKind.NONE, parallel=parallel))
    assert len(pieces) == (LAYERS - FIRST_DENSE) * EXPERTS


def test_a_bf16_piece_fills_the_bank_gate_first():
    """The unquantized bank is [gate; up] per expert; the reader hands gate and up apart."""
    from freetoken.layers.quantization.moe.base import fused_piece

    gate, up = _bf16(1, MOE_INTER, HIDDEN), _bf16(1, MOE_INTER, HIDDEN)
    fused = fused_piece({"gate": gate, "up": up, "down": _bf16(1, HIDDEN, MOE_INTER)}, "gate_up")
    assert torch.equal(fused[0, :MOE_INTER], gate[0]) and torch.equal(fused[0, MOE_INTER:], up[0])


def test_other_expert_kinds_go_their_own_way():
    import freetoken.models.glm4_moe.weight as weight
    from freetoken.layers.quantization import QuantKind

    _, config = _model()
    assert weight.iter_expert_pieces("/fake", config, QuantKind.NVFP4) is None


def test_the_nvfp4_spec_numbers_banks_from_the_stage():
    from freetoken.models.glm4_moe.weight import _NVFP4_SOURCE_SPEC as spec

    _, whole = _model()
    assert [spec.layer_to_bank(lid, whole) for lid in range(LAYERS + 1)] == [None, 0, 1, 2, 3, 4, None]
    _, tail = _model("3:6")
    assert [spec.layer_to_bank(lid, tail) for lid in range(LAYERS + 1)] == [None, None, None, 0, 1, 2, None]


class _Reader:
    def __init__(self, names):
        self.names = set(names)

    def has(self, name):
        return name in self.names

    def get(self, name):
        if name.endswith("weight_scale_2"):
            return torch.ones(())
        return torch.zeros(2, 2, dtype=torch.bfloat16)


def test_a_resident_linear_is_whatever_the_checkpoint_stores():
    from freetoken.models.glm4_moe.weight import _iter_resident_linear

    bf16 = [n for n, _ in _iter_resident_linear(_Reader({"x.weight"}), "x", "y")]
    assert bf16 == ["y.weight"]
    nvfp4 = [n for n, _ in _iter_resident_linear(
        _Reader({"x.weight", "x.weight_scale", "x.weight_scale_2"}), "x", "y")]
    assert nvfp4 == ["y.weight", "y.weight_scale", "y.weight_global"]


# ------------------------------------------------------------------ seam


class _Layer:
    """Same contract as the real layer: the deferred residual; the add is the fused norm's."""

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

    def forward(self, ids):
        return self.table[ids]


def _stand_in(model, first: int, table):
    inner = model.model
    inner.layers.op_list[:] = [_Layer(first + i) for i in range(len(inner.layers.op_list))]
    if inner.embed_tokens is not None:
        inner.embed_tokens = _Embed(table)
    if inner.norm is not None:
        inner.norm = _Norm()
    return inner


def test_the_seam_carries_the_pair_and_the_cut_changes_nothing():
    table = torch.randn(VOCAB, HIDDEN).to(torch.bfloat16)
    ids = torch.tensor([3, 17, 42, 99])
    whole = _stand_in(_model()[0], 0, table).forward(ids)
    head = _stand_in(_model("0:3")[0], 0, table)
    tail = _stand_in(_model("3:6")[0], 3, table)
    seam = head.forward(ids)
    assert seam.shape == (4, 2 * HIDDEN)
    assert torch.equal(tail.forward(ids, hidden=seam), whole)


def test_the_seam_refuses_what_it_cannot_continue():
    tail = _stand_in(_model("3:6")[0], 3, torch.zeros(VOCAB, HIDDEN))
    ids = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="no remainder"):
        tail.forward(ids)
    with pytest.raises(ValueError, match="width"):
        tail.forward(ids, hidden=torch.zeros(2, HIDDEN))
