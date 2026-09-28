"""gpt-oss (gpt_oss): pipeline stages.

A stage builds and reads only its own layers; the seam is the deferred (x, residual)
pair, like Qwen3-MoE's. The family is the first stage-capable one with sliding-window
attention, so the hybrid full/SWA KV pool has to address a stage's layers by their
global ids too.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import clear_stage_info, set_tp_info, try_get_tp_info
from freetoken.utils.hf import RawConfigShim

LAYERS, HIDDEN, HEADS, KV, HEAD_DIM = 6, 64, 4, 2, 64
VOCAB, EXPERTS, INTER = 128, 4, 64
SPLIT = 3


def _hf_config() -> RawConfigShim:
    return RawConfigShim({
        "architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss",
        "hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV, "head_dim": HEAD_DIM, "vocab_size": VOCAB,
        "intermediate_size": INTER, "num_local_experts": EXPERTS, "num_experts_per_tok": 2,
        "experts_per_token": 2, "hidden_act": "silu", "rms_norm_eps": 1e-5,
        "max_position_embeddings": 4096, "rope_theta": 150000, "attention_bias": True,
        "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                         "original_max_position_embeddings": 4096, "truncate": False},
        "sliding_window": 128, "swiglu_limit": 7.0, "tie_word_embeddings": False,
        "layer_types": ["sliding_attention", "full_attention"] * (LAYERS // 2),
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router",
                                       "model.embed_tokens", "lm_head"],
        },
    })


def _bf16(*shape):
    return torch.randn(*shape).to(torch.bfloat16)


def _u8(*shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8)


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
        f"{p}.self_attn.o_proj.bias": _bf16(HIDDEN),
        f"{p}.self_attn.sinks": _bf16(HEADS),
        f"{p}.input_layernorm.weight": _bf16(HIDDEN),
        f"{p}.post_attention_layernorm.weight": _bf16(HIDDEN),
        f"{p}.mlp.router.weight": _bf16(EXPERTS, HIDDEN),
        f"{p}.mlp.router.bias": _bf16(EXPERTS),
        f"{p}.mlp.experts.gate_up_proj_blocks": _u8(EXPERTS, 2 * INTER, HIDDEN // 32, 16),
        f"{p}.mlp.experts.gate_up_proj_scales": _u8(EXPERTS, 2 * INTER, HIDDEN // 32),
        f"{p}.mlp.experts.gate_up_proj_bias": _bf16(EXPERTS, 2 * INTER),
        f"{p}.mlp.experts.down_proj_blocks": _u8(EXPERTS, HIDDEN, INTER // 32, 16),
        f"{p}.mlp.experts.down_proj_scales": _u8(EXPERTS, HIDDEN, INTER // 32),
        f"{p}.mlp.experts.down_proj_bias": _bf16(EXPERTS, HIDDEN),
    }
    return out


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """gpt-oss-shaped MXFP4 checkpoint, two shards, the second one starting mid-model."""
    folder = tmp_path_factory.mktemp("gpt_oss_mxfp4")
    first = {"model.embed_tokens.weight": _bf16(VOCAB, HIDDEN)}
    for n in range(SPLIT - 1):
        first.update(_layer(n))
    second = {"model.norm.weight": _bf16(HIDDEN), "lm_head.weight": _bf16(VOCAB, HIDDEN)}
    for n in range(SPLIT - 1, LAYERS):
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
    import freetoken.models.gpt_oss.weight as weight

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    shim = _hf_config()
    monkeypatch.setattr(engine_config, "cached_load_hf_config", lambda path: shim)
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
    return [layer.mlp.experts.layer_id for layer in model.model.layers.op_list]


# ------------------------------------------------------------------ model


def test_the_whole_model_builds_as_before():
    model, config = _model()
    keys = set(model.state_dict())
    assert _layer_ids(keys) == set(range(LAYERS))
    assert "model.norm.weight" in keys and "lm_head.weight" in keys
    assert any(k.startswith("model.embed_tokens.") for k in keys)
    assert _banks(model) == list(range(LAYERS))
    assert model.produces_logits and model.stage_input_width == 0
    assert config.moe_weight_format == "mxfp4"


@pytest.mark.parametrize("span, embed, head", [
    ("0:3", True, False),
    ("3:6", False, True),
    ("2:4", False, False),
])
def test_a_stage_builds_its_layers_and_its_edges(span, embed, head):
    model, config = _model(span)
    keys = set(model.state_dict())
    first, last = map(int, span.split(":"))
    assert _layer_ids(keys) == set(range(last - first))
    assert any(k.startswith("model.embed_tokens.") for k in keys) is embed
    assert ("model.norm.weight" in keys) is head and ("lm_head.weight" in keys) is head
    assert model.produces_logits is head
    assert model.stage_input_width == (0 if embed else 2 * HIDDEN)
    assert _banks(model) == list(range(last - first)) and config.num_moe_layers == last - first


def test_two_stages_are_the_model_cut_not_rebuilt():
    from freetoken.models.stage_weights import LAYER_KEY_RE

    whole = set(_model()[0].state_dict())
    first = set(_model("0:3")[0].state_dict())
    last = set(_model("3:6")[0].state_dict())
    shift = lambda keys, by: {LAYER_KEY_RE.sub(lambda m: f"{m['head']}{int(m['id']) + by}.", k, count=1)
                              for k in keys}
    assert shift(first, 0) | shift(last, 3) == whole and not shift(first, 0) & shift(last, 3)


def test_the_family_is_allowed_to_be_a_stage():
    from freetoken.models.register import get_model_spec

    assert get_model_spec("GptOssForCausalLM").supports_stages


# ------------------------------------------------------------------ KV


def test_kv_groups_cover_only_the_stage_layers():
    specs = {s.name: s.layer_ids for s in _model("3:6")[1].kv_cache_group_specs()}
    assert specs == {"swa": (4,), "full": (3, 5)}


def _pool(config):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    return HybridSWAKVCache(groups=config.kv_cache_group_specs(), num_layers=config.num_layers,
                            num_full_pages=4, page_size=16, num_swa_tokens=32,
                            dtype=torch.bfloat16, device=torch.device("cpu"))


def test_the_swa_pool_addresses_a_stage_by_global_layer_id():
    """The pool used to map layers in a list over the WHOLE model: a stage's missing
    layers raised 'missing from full/swa groups', and a compacted list would have handed
    layer 3 the slab of whatever sat at position 3."""
    pool = _pool(_model("3:6")[1])
    assert [pool.group_of(i) for i in (3, 4, 5)] == ["full", "swa", "full"]
    assert pool.k_cache(3).data_ptr() == pool.full_kv_pool.k_buffer[0].data_ptr()
    assert pool.k_cache(5).data_ptr() == pool.full_kv_pool.k_buffer[1].data_ptr()
    assert pool.k_cache(4).data_ptr() == pool.swa_kv_pool.k_buffer[0].data_ptr()
    assert pool.full_kv_pool.k_buffer.shape[0] == 2 and pool.swa_kv_pool.k_buffer.shape[0] == 1
    with pytest.raises(KeyError):
        pool.group_of(0)


def test_the_swa_pool_for_the_whole_model_is_as_before():
    pool = _pool(_model()[1])
    assert [pool.group_of(i) for i in range(LAYERS)] == ["swa", "full"] * (LAYERS // 2)
    assert pool.k_cache(5).data_ptr() == pool.full_kv_pool.k_buffer[2].data_ptr()


def test_the_swa_pool_still_refuses_a_hole():
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    specs = (KVCacheGroupSpec(name="full", layer_ids=(1, 5), num_kv_heads=KV, head_dim=HEAD_DIM, sliding_window=None),
             KVCacheGroupSpec(name="swa", layer_ids=(0, 2), num_kv_heads=KV, head_dim=HEAD_DIM, sliding_window=128))
    with pytest.raises(ValueError, match="missing"):
        HybridSWAKVCache(groups=specs, num_layers=LAYERS, num_full_pages=4, page_size=16,
                         num_swa_tokens=32, dtype=torch.bfloat16, device=torch.device("cpu"))


# ------------------------------------------------------------------ weights


@pytest.mark.parametrize("span", ["", "0:3", "3:6", "2:4"])
def test_the_dense_reader_feeds_exactly_what_the_stage_built(checkpoint, span):
    from freetoken.models.gpt_oss.weight import iter_weights

    path, _ = checkpoint
    model, _ = _model(span, path)
    built = set(model.state_dict())
    read = {name for name, _ in iter_weights(path, torch.device("cpu"),
                                             include_moe_experts=False, include_non_moe=True)}
    assert read == built


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.parametrize("span, layers", [("", (0, 1, 2, 3, 4, 5)), ("0:3", (0, 1, 2)), ("3:6", (3, 4, 5))],
                         ids=["whole", "head", "tail"])
def test_the_expert_reader_takes_this_stage_by_bank(checkpoint, span, layers, parallel):
    """Through the engine entry point, which looks the hook up on the family package."""
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_pieces import iter_expert_pieces

    path, tensors = checkpoint
    _, config = _model(span, path)
    pieces = list(iter_expert_pieces(path, config, QuantKind.MXFP4, parallel=parallel, workers=2))
    assert sorted(p[0] for p in pieces) == list(range(len(layers)))
    for bank, e0, e1, piece in pieces:
        assert (e0, e1) == (0, EXPERTS)
        src = f"model.layers.{layers[bank]}.mlp.experts"
        assert torch.equal(piece["gate_up"], tensors[f"{src}.gate_up_proj_blocks"])
        assert torch.equal(piece["down_bias"], tensors[f"{src}.down_proj_bias"])


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
    with pytest.raises(ValueError, match="no stream"):
        tail.forward(ids)
    with pytest.raises(ValueError, match="wide"):
        tail.forward(ids, hidden=torch.zeros(2, HIDDEN))
