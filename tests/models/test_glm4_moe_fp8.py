"""GLM-4 MoE (glm4_moe): the channel-fp8 release (zai-org/GLM-4.5-Air-FP8).

llm-compressor quantized every Linear to fp8 e4m3 with one fp32 scale per output row
(``strategy: channel``, dynamic per-token activations), routers and norms excepted. Stand
2026-09-28: the model died at build, ``no quant method for moe.fp8_tensor`` -- the experts had
no method for that kind. Behind it the reader had two silent faults: the dense MLP took
``weight_scale`` for an NVFP4 export, and the bf16-only attention would have loaded the raw
e4m3 codes as weights.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import clear_stage_info, set_tp_info, try_get_tp_info
from freetoken.utils.hf import RawConfigShim

LAYERS, HIDDEN, HEADS, KV, HEAD_DIM = 6, 128, 4, 2, 64
VOCAB, EXPERTS, INTER, MOE_INTER = 128, 4, 256, 128
FIRST_DENSE = 1
FP8 = torch.float8_e4m3fn

QUANT = {
    "quant_method": "compressed-tensors", "format": "float-quantized",
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True,
                    "dynamic": False, "group_size": None, "block_structure": None},
        "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True},
    }},
    "ignore": ["lm_head", *[f"model.layers.{n}.mlp.gate" for n in range(LAYERS + 1)]],
}


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
        "quantization_config": QUANT,
    })


def _bf16(*shape):
    return torch.randn(*shape).to(torch.bfloat16)


def _fp8(out: int, inp: int, name: str) -> dict[str, torch.Tensor]:
    """llm-compressor channel fp8: e4m3 weight + fp32 [out, 1] scale."""
    return {f"{name}.weight": torch.randn(out, inp).to(FP8),
            f"{name}.weight_scale": torch.rand(out, 1) + 0.5}


def _layer(n: int) -> dict[str, torch.Tensor]:
    p = f"model.layers.{n}"
    out = {
        **_fp8(HEADS * HEAD_DIM, HIDDEN, f"{p}.self_attn.q_proj"),
        **_fp8(KV * HEAD_DIM, HIDDEN, f"{p}.self_attn.k_proj"),
        **_fp8(KV * HEAD_DIM, HIDDEN, f"{p}.self_attn.v_proj"),
        **_fp8(HIDDEN, HEADS * HEAD_DIM, f"{p}.self_attn.o_proj"),
        f"{p}.self_attn.q_proj.bias": _bf16(HEADS * HEAD_DIM),
        f"{p}.self_attn.k_proj.bias": _bf16(KV * HEAD_DIM),
        f"{p}.self_attn.v_proj.bias": _bf16(KV * HEAD_DIM),
        f"{p}.input_layernorm.weight": _bf16(HIDDEN),
        f"{p}.post_attention_layernorm.weight": _bf16(HIDDEN),
    }
    if n < FIRST_DENSE:
        for proj, (o, i) in {"gate_proj": (INTER, HIDDEN), "up_proj": (INTER, HIDDEN), "down_proj": (HIDDEN, INTER)}.items():
            out.update(_fp8(o, i, f"{p}.mlp.{proj}"))
        return out
    out.update({f"{p}.mlp.gate.weight": _bf16(EXPERTS, HIDDEN),
                f"{p}.mlp.gate.e_score_correction_bias": torch.randn(EXPERTS)})
    for proj, (o, i) in {"gate_proj": (MOE_INTER, HIDDEN), "up_proj": (MOE_INTER, HIDDEN), "down_proj": (HIDDEN, MOE_INTER)}.items():
        out.update(_fp8(o, i, f"{p}.mlp.shared_experts.{proj}"))
        for e in range(EXPERTS):
            out.update(_fp8(o, i, f"{p}.mlp.experts.{e}.{proj}"))
    return out


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Two shards, the second one starting mid-model, plus the MTP layer (index num_layers)."""
    folder = tmp_path_factory.mktemp("glm4_moe_fp8")
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
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda path: shim)
    clear_stage_info()
    yield
    clear_stage_info()


def _model(span: str = "", path: str = "/fake"):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.layers.quantization import set_quant_config
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    clear_stage_info()
    config = EngineConfig(model_path=path, tp_info=DistributedInfo(rank=0, size=1),
                          dtype=torch.bfloat16, moe_strategy="offload", layer_range=span)
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


# ------------------------------------------------------------------ model


def test_the_experts_get_the_channel_fp8_method():
    from freetoken.layers.quantization import QuantKind
    from freetoken.layers.quantization.moe import Fp8ChannelMoEMethod

    model, config = _model()
    moe = model.model.layers.op_list[FIRST_DENSE].mlp
    assert isinstance(moe.experts.quant_method, Fp8ChannelMoEMethod)
    assert moe.shared_experts.gate_proj.quant_method.kind is QuantKind.FP8_TENSOR
    assert model.model.layers.op_list[0].mlp.gate_proj.quant_method.kind is QuantKind.FP8_TENSOR


def test_the_bank_is_fp8_codes_and_one_fp32_scale_per_row():
    model, _ = _model()
    method = model.model.layers.op_list[FIRST_DENSE].mlp.experts.quant_method
    layout = {role: (spec.shape, spec.dtype) for role, spec in method.layout().items()}
    assert layout == {
        "gate_up": ((2 * MOE_INTER, HIDDEN), FP8), "gate_up_scale": ((2 * MOE_INTER,), torch.float32),
        "down": ((HIDDEN, MOE_INTER), FP8), "down_scale": ((HIDDEN,), torch.float32),
    }


# ------------------------------------------------------------------ weights


@pytest.mark.parametrize("span", ["", "0:3", "3:6"])
def test_the_dense_reader_feeds_exactly_what_the_model_built(checkpoint, span):
    from freetoken.models.glm4_moe.weight import iter_weights

    path, _ = checkpoint
    model, _ = _model(span, path)
    built = {name: (tuple(t.shape), t.dtype) for name, t in model.state_dict().items()}
    read = {name: (tuple(t.shape), t.dtype) for name, t in iter_weights(
        path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True)}
    assert set(read) == set(built)
    fp8 = {n for n in read if n.endswith(".weight") and read[n][1] == FP8}
    assert fp8 and all(".mlp." in n and ".experts." not in n for n in fp8)
    assert all(read[n] == built[n] for n in fp8)
    assert all(read[n.replace(".weight", ".weight_scale")] == built[n.replace(".weight", ".weight_scale")] for n in fp8)


def test_attention_is_dequantized_before_df11(checkpoint, monkeypatch):
    """DF11 attention is bf16-only; the e4m3 codes times their row scale go in, not the codes."""
    import freetoken.models.glm4_moe.weight as weight

    path, tensors = checkpoint
    seen: list[torch.Tensor] = []
    real = weight.compress_df11_weight
    monkeypatch.setattr(weight, "compress_df11_weight", lambda w: seen.append(w) or real(w))
    _model("", path)
    for _ in weight.iter_weights(path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True):
        if len(seen) >= 4:
            break
    q = tensors["model.layers.0.self_attn.q_proj.weight"].float() * tensors["model.layers.0.self_attn.q_proj.weight_scale"]
    assert seen[0].dtype == torch.bfloat16
    assert torch.equal(seen[0], q.to(torch.bfloat16))


def test_a_dense_linear_keeps_its_codes_and_row_scale(checkpoint):
    from freetoken.models.glm4_moe.weight import iter_weights

    path, tensors = checkpoint
    _model("", path)
    read = dict(iter_weights(path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    src = "model.layers.1.mlp.shared_experts.down_proj"
    assert torch.equal(read[f"{src}.weight"].view(torch.uint8), tensors[f"{src}.weight"].view(torch.uint8))
    assert torch.equal(read[f"{src}.weight_scale"], tensors[f"{src}.weight_scale"].reshape(-1))


@pytest.mark.parametrize("parallel", [False, None])
@pytest.mark.parametrize("span, layers", [("", (1, 2, 3, 4, 5)), ("0:3", (1, 2)), ("3:6", (3, 4, 5))],
                         ids=["whole", "head", "tail"])
def test_the_expert_reader_takes_codes_and_scales_by_bank(checkpoint, span, layers, parallel):
    """Through the engine entry point; the MTP layer (index num_layers) is never read."""
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_pieces import iter_expert_pieces

    path, tensors = checkpoint
    _, config = _model(span, path)
    pieces = list(iter_expert_pieces(path, config, QuantKind.FP8_TENSOR, parallel=parallel))
    assert sorted((p[0], p[1]) for p in pieces) == [(b, e) for b in range(len(layers)) for e in range(EXPERTS)]
    bank, e0, _, piece = next(p for p in pieces if p[0] == len(layers) - 1 and p[1] == 2)
    assert set(piece) == {"gate", "gate_scale", "up", "up_scale", "down", "down_scale"}
    src = f"model.layers.{layers[bank]}.mlp.experts.2"
    assert torch.equal(piece["up"][0].view(torch.uint8), tensors[f"{src}.up_proj.weight"].view(torch.uint8))
    assert torch.equal(piece["down_scale"][0], tensors[f"{src}.down_proj.weight_scale"])


def test_the_pack_puts_gate_scales_before_up_scales():
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.fp8_channel import TritonFp8ChannelMoEKernel

    kernel = TritonFp8ChannelMoEKernel()
    cfg = MoEConfig(num_experts=2, hidden=HIDDEN, intermediate=MOE_INTER, top_k=1, strategy="offload")
    pieces = {
        "gate": torch.randn(2, MOE_INTER, HIDDEN).to(FP8), "up": torch.randn(2, MOE_INTER, HIDDEN).to(FP8),
        "down": torch.randn(2, HIDDEN, MOE_INTER).to(FP8),
        "gate_scale": torch.rand(2, MOE_INTER, 1), "up_scale": torch.rand(2, MOE_INTER, 1),
        "down_scale": torch.rand(2, HIDDEN, 1),
    }
    out = {role: torch.empty(2, *spec.shape, dtype=spec.dtype) for role, spec in kernel.layout(cfg).items()}
    kernel.pack(pieces, cfg, out)
    assert torch.equal(out["gate_up_scale"][:, :MOE_INTER], pieces["gate_scale"][..., 0])
    assert torch.equal(out["gate_up_scale"][:, MOE_INTER:], pieces["up_scale"][..., 0])
    assert torch.equal(out["gate_up"][:, MOE_INTER:].view(torch.uint8), pieces["up"].view(torch.uint8))
    assert torch.equal(out["down_scale"], pieces["down_scale"][..., 0])


def test_a_tensor_scale_is_broadcast_to_rows():
    from freetoken.layers.quantization.moe.fp8_channel import _rows

    assert torch.equal(_rows(torch.tensor([2.0, 3.0]), 4), torch.tensor([[2.0] * 4, [3.0] * 4]))


def test_the_bank_format_has_a_name():
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.legacy_format import legacy_bank_names, legacy_format_for

    fmt = legacy_format_for(QuantKind.FP8_TENSOR, "triton")
    assert fmt == "fp8_channel"
    assert set(legacy_bank_names(fmt)) == {"gate_up", "gate_up_scale", "down", "down_scale"}
