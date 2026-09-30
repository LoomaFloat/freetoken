"""MiniMax-M2 (minimax_m2): the block-fp8 release (MiniMaxAI/MiniMax-M2.7).

The family was written for the NVFP4 export: its only expert reader was the ModelOpt spec,
so the fp8 original built its model and dense weights and then had nothing to fill the expert
banks with (``no expert reader for fp8_block experts``). The checkpoint stores per-expert
``w1`` (gate) / ``w3`` (up) / ``w2`` (down) fp8 codes with 128x128 ``weight_scale_inv``;
attention is block-fp8 as well, router, selection bias and lm_head are bf16.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import clear_stage_info, set_tp_info, try_get_tp_info
from freetoken.utils.hf import RawConfigShim

LAYERS, HIDDEN, HEADS, KV, HEAD_DIM = 3, 256, 4, 2, 64
VOCAB, EXPERTS, INTER = 128, 4, 256
B = 128
FP8 = torch.float8_e4m3fn


def _hf_config() -> RawConfigShim:
    return RawConfigShim({
        "architectures": ["MiniMaxM2ForCausalLM"], "model_type": "minimax_m2",
        "hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV, "head_dim": HEAD_DIM, "vocab_size": VOCAB,
        "intermediate_size": INTER, "num_local_experts": EXPERTS, "num_experts_per_tok": 2,
        "hidden_act": "silu", "rms_norm_eps": 1e-6, "max_position_embeddings": 4096,
        "rope_theta": 5_000_000, "rotary_dim": 32, "use_qk_norm": True, "qk_norm_type": "per_layer",
        "scoring_func": "sigmoid", "use_routing_bias": True, "tie_word_embeddings": False,
        "quantization_config": {
            "quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "float8_e4m3fn",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["gate", "e_score_correction_bias", "lm_head"],
        },
    })


def _fp8(out: int, inp: int, name: str) -> dict[str, torch.Tensor]:
    return {f"{name}.weight": torch.randn(out, inp).to(FP8),
            f"{name}.weight_scale_inv": torch.rand(out // B, inp // B) + 0.5}


def _bf16(*shape):
    return torch.randn(*shape).to(torch.bfloat16)


def _layer(n: int) -> dict[str, torch.Tensor]:
    p = f"model.layers.{n}"
    out = {
        **_fp8(HEADS * HEAD_DIM, HIDDEN, f"{p}.self_attn.q_proj"),
        **_fp8(KV * HEAD_DIM, HIDDEN, f"{p}.self_attn.k_proj"),
        **_fp8(KV * HEAD_DIM, HIDDEN, f"{p}.self_attn.v_proj"),
        **_fp8(HIDDEN, HEADS * HEAD_DIM, f"{p}.self_attn.o_proj"),
        f"{p}.self_attn.q_norm.weight": _bf16(HEADS * HEAD_DIM),
        f"{p}.self_attn.k_norm.weight": _bf16(KV * HEAD_DIM),
        f"{p}.input_layernorm.weight": _bf16(HIDDEN),
        f"{p}.post_attention_layernorm.weight": _bf16(HIDDEN),
        f"{p}.block_sparse_moe.gate.weight": _bf16(EXPERTS, HIDDEN),
        f"{p}.block_sparse_moe.e_score_correction_bias": torch.randn(EXPERTS),
    }
    for e in range(EXPERTS):
        for proj, (o, i) in {"w1": (INTER, HIDDEN), "w3": (INTER, HIDDEN), "w2": (HIDDEN, INTER)}.items():
            out.update(_fp8(o, i, f"{p}.block_sparse_moe.experts.{e}.{proj}"))
    return out


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("minimax_m2_fp8")
    first = {"model.embed_tokens.weight": _bf16(VOCAB, HIDDEN), **_layer(0), **_layer(1)}
    second = {"model.norm.weight": _bf16(HIDDEN), "lm_head.weight": _bf16(VOCAB, HIDDEN), **_layer(2)}
    save_file(first, str(folder / "model-00001-of-00002.safetensors"))
    save_file(second, str(folder / "model-00002-of-00002.safetensors"))
    weight_map = {k: "model-00001-of-00002.safetensors" for k in first}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in second})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return str(folder), {**first, **second}


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    import freetoken.engine.config as engine_config
    import freetoken.models.minimax_m2.weight as weight

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    shim = _hf_config()
    monkeypatch.setattr(engine_config, "cached_load_hf_config", lambda path: shim)
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda path: shim)
    clear_stage_info()
    yield
    clear_stage_info()


def _model(path: str = "/fake"):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.layers.quantization import set_quant_config
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    config = EngineConfig(model_path=path, tp_info=DistributedInfo(rank=0, size=1),
                          dtype=torch.bfloat16, moe_strategy="offload")
    object.__setattr__(config.model_config, "moe_strategy", "offload")
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    set_quant_config(config.model_config.quant)
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            return create_model(config.model_config), config.model_config
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


def _method(model):
    return model.model.layers.op_list[0].block_sparse_moe.experts.quant_method


# ------------------------------------------------------------------ model


def test_the_experts_get_the_block_fp8_method():
    from freetoken.layers.quantization.moe import Fp8BlockMoEMethod

    model, _ = _model()
    assert isinstance(_method(model), Fp8BlockMoEMethod)


# ------------------------------------------------------------------ weights


def test_the_dense_reader_feeds_exactly_what_the_model_built(checkpoint):
    """fp8 attention: q|k|v merge into qkv_proj with their block scales; the router stays bf16."""
    from freetoken.models.minimax_m2.weight import iter_weights

    path, _ = checkpoint
    model, _ = _model(path)
    built = {n: (tuple(t.shape), t.dtype) for n, t in model.state_dict().items()}
    read = {n: (tuple(t.shape), t.dtype) for n, t in iter_weights(
        path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True)}
    assert set(read) == set(built)
    qkv = "model.layers.0.self_attn.qkv_proj"
    assert read[f"{qkv}.weight"] == (((HEADS + 2 * KV) * HEAD_DIM, HIDDEN), FP8) == built[f"{qkv}.weight"]
    assert read[f"{qkv}.weight_scale_inv"][0] == built[f"{qkv}.weight_scale_inv"][0]


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_the_expert_reader_through_the_engine_entry(checkpoint, parallel):
    """The engine looks the hook up on the family package (GLM-4.5-Air died on exactly that)."""
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_pieces import iter_expert_pieces

    path, tensors = checkpoint
    _, config = _model(path)
    pieces = list(iter_expert_pieces(path, config, QuantKind.FP8_BLOCK, parallel=parallel, workers=2))
    assert sorted((p[0], p[1]) for p in pieces) == [(l, e) for l in range(LAYERS) for e in range(EXPERTS)]
    layer, e, _, piece = next(p for p in pieces if p[0] == 2 and p[1] == 3)
    assert set(piece) == {"gate", "gate_scale", "up", "up_scale", "down", "down_scale"}
    src = "model.layers.2.block_sparse_moe.experts.3"
    for role, proj in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
        assert torch.equal(piece[role][0].view(torch.uint8), tensors[f"{src}.{proj}.weight"].view(torch.uint8))
        assert torch.equal(piece[f"{role}_scale"][0], tensors[f"{src}.{proj}.weight_scale_inv"])


def test_a_piece_packs_into_the_block_fp8_bank_gate_first(checkpoint):
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_pieces import iter_expert_pieces

    path, tensors = checkpoint
    model, config = _model(path)
    method = _method(model)
    _, e, e1, piece = next(p for p in iter_expert_pieces(path, config, QuantKind.FP8_BLOCK) if p[0] == 1 and p[1] == 0)
    out = {role: torch.zeros(1, *spec.shape, dtype=spec.dtype) for role, spec in method.layout().items()}
    method.pack(piece, out)
    src = "model.layers.1.block_sparse_moe.experts.0"
    assert torch.equal(out["gate_up"][0, :INTER].view(torch.uint8), tensors[f"{src}.w1.weight"].view(torch.uint8))
    assert torch.equal(out["gate_up"][0, INTER:].view(torch.uint8), tensors[f"{src}.w3.weight"].view(torch.uint8))
    gs = out["gate_up_scale"][0, :, : HIDDEN // B].float()
    want = torch.cat([tensors[f"{src}.w1.weight_scale_inv"], tensors[f"{src}.w3.weight_scale_inv"]]).to(gs.dtype)
    assert torch.equal(gs, want.to(out["gate_up_scale"].dtype).float())


def test_other_expert_kinds_go_their_own_way():
    import freetoken.models.minimax_m2 as family
    from freetoken.layers.quantization import QuantKind

    _, config = _model()
    assert family.iter_expert_pieces("/fake", config, QuantKind.NVFP4) is None
