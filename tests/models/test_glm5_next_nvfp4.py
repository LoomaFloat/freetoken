"""GLM-5.3-Flash, the ModelOpt NVFP4 release by NVIDIA (nvidia/GLM-5.3-Flash-NVFP4).

Unlike the LibertAIDAI / RedHatAI exports the reader was written for, NVIDIA quantized the
dense-prefix MLP (layers 0..2) to NVFP4 as well; attention, routers and shared experts stay
bf16. Stand 2026-09-29: the head stage died on ``KeyError:
'model.layers.0.mlp.gate_proj.weight_scale'`` -- the model built an NVFP4 linear there, the
reader cast the packed e2m1 codes to bf16 and gave no scales. The tail stage has no dense
layer and came up; a single node would have died the same way as the head.
"""

from __future__ import annotations

import json

import pytest
import torch

from freetoken.distributed import clear_stage_info, set_tp_info, try_get_tp_info

from .test_glm5_next_stage import FIRST_DENSE, LAYERS, _hf_config, _layer_ids

FP8 = torch.float8_e4m3fn

# nvidia/GLM-5.3-Flash-NVFP4's quantization_config, trimmed to the test's layers
NVIDIA_NVFP4 = {
    "quant_method": "modelopt", "quant_algo": "NVFP4",
    "config_groups": {"group_0": {
        "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
        "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
        "targets": ["Linear"],
    }},
    "ignore": [
        "lm_head", "model.language_model.embed_tokens", "model.visual*",
        *[f"model.language_model.layers.{i}.self_attn*" for i in range(LAYERS)],
        *[f"model.language_model.layers.{i}.mlp.gate" for i in range(FIRST_DENSE, LAYERS)],
        *[f"model.language_model.layers.{i}.mlp.shared_experts*" for i in range(FIRST_DENSE, LAYERS)],
    ],
}


def _shim():
    from freetoken.utils.hf import RawConfigShim

    return RawConfigShim({**_hf_config()._data, "quantization_config": NVIDIA_NVFP4})


@pytest.fixture(autouse=True)
def _runtime(monkeypatch, tmp_path):
    import freetoken.engine.config as engine_config
    import freetoken.models.glm5_next.weight as weight

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    shim = _shim()
    monkeypatch.setattr(engine_config, "cached_load_hf_config", lambda path: shim)
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda path: shim)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    monkeypatch.setattr(weight, "download_hf_weight", lambda path: str(tmp_path))
    monkeypatch.setattr(weight, "_ShardReader", _Nvfp4Reader)
    clear_stage_info()
    yield
    clear_stage_info()


def _model(span: str = ""):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.layers.quantization import set_quant_config
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    clear_stage_info()
    config = EngineConfig(model_path="/fake", tp_info=DistributedInfo(rank=0, size=1),
                          dtype=torch.bfloat16, moe_strategy="offload", layer_range=span)
    object.__setattr__(config.model_config, "moe_strategy", "offload")
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    set_quant_config(config.model_config.quant)
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            return create_model(config.model_config), config.model_config
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


class _Nvfp4Reader:
    """Shards as NVIDIA wrote them: the dense-prefix MLP in NVFP4, everything else bf16."""

    def __init__(self, folder, weight_map, device):
        pass

    @staticmethod
    def _dense_mlp(name: str) -> bool:
        return any(f".layers.{i}.mlp." in name for i in range(FIRST_DENSE)) and ".experts." not in name

    def has(self, name):
        if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            return self._dense_mlp(name)
        return True

    def get(self, name):
        if self._dense_mlp(name):
            if name.endswith(".weight"):
                return torch.full((4, 8), 0x3C, dtype=torch.uint8)
            if name.endswith(".weight_scale"):
                return torch.ones(4, 1).to(FP8)
            if name.endswith(".weight_scale_2"):
                return torch.tensor([0.25])
            if name.endswith(".input_scale"):
                return torch.tensor([2.0])
        return torch.zeros(2, 2, dtype=torch.bfloat16)

    def close(self):
        pass


def test_the_dense_prefix_mlp_is_built_nvfp4():
    from freetoken.layers.quantization import QuantKind

    model, _ = _model()
    mlp = model.model.layers.op_list[0].mlp
    assert mlp.gate_proj.quant_method.kind is QuantKind.NVFP4


@pytest.mark.parametrize("span", ["", "0:3", "3:6"])
def test_the_dense_reader_feeds_exactly_what_the_model_built(span):
    import freetoken.models.glm5_next.weight as weight

    model, _ = _model(span)
    built = set(model.state_dict())
    read = {name for name, _ in weight.iter_weights(
        "/fake", torch.device("cpu"), include_moe_experts=False, include_non_moe=True, include_vision=False)}
    assert read == built
    first = int(span.split(":")[0]) if span else 0
    assert any(n.endswith("mlp.gate_proj.weight_global") for n in read) is (first < FIRST_DENSE)


def test_an_nvfp4_projection_keeps_codes_and_turns_the_global_into_rows():
    from freetoken.models.glm5_next.weight import _proj

    reader = _Nvfp4Reader(None, None, None)
    out = dict(_proj(reader, "model.language_model.layers.0.mlp.up_proj", "model.layers.0.mlp.up_proj"))
    assert set(out) == {f"model.layers.0.mlp.up_proj.{k}" for k in ("weight", "weight_scale", "weight_global", "input_scale")}
    assert out["model.layers.0.mlp.up_proj.weight"].dtype == torch.uint8
    assert torch.equal(out["model.layers.0.mlp.up_proj.weight_global"], torch.full((4,), 0.25, dtype=torch.float16))
    assert out["model.layers.0.mlp.up_proj.input_scale"].shape == ()


def test_bf16_and_fp8_projections_are_as_before():
    from freetoken.models.glm5_next.weight import _proj

    class Reader:
        def __init__(self, w):
            self.w = w

        def has(self, name):
            return False

        def get(self, name):
            return self.w if name.endswith(".weight") else torch.ones(1, 1)

    assert list(dict(_proj(Reader(torch.zeros(2, 2)), "a", "b"))) == ["b.weight"]
    assert list(dict(_proj(Reader(torch.zeros(2, 2).to(FP8)), "a", "b"))) == ["b.weight", "b.weight_scale_inv"]
