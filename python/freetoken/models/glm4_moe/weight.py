from __future__ import annotations

import json
import os
import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_stage_info, get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.loader import drop_page_cache
from freetoken.moe.expert_pieces import bank_layer_of
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config
from .df11_embedding import compress_df11_embedding
from .df11_linear import compress_df11_weight

# routed experts go to the offload cache, not the dense model.
_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_ROUTED_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_ROUTED_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    # dense prefix, the MTP layer (index num_layers) and, on a pipeline stage, the other
    # stages' layers have no bank here
    layer_to_bank=lambda layer, config: bank_layer_of(config, layer),
    desc="GLM NVFP4 experts",
)
# bf16 release (zai-org/GLM-4.5-Air): one plain weight per routed-expert projection.
_BF16_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>gate|up|down)_proj\.weight$"
)


# --------------------------------------------------------------------------------------
# Dense / resident weight streaming.
# --------------------------------------------------------------------------------------
class _ShardReader:
    """Opens safetensors shards on demand (mmap) and serves tensors on ``device``."""

    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = device
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=str(self._device)
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best effort
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _iter_resident_linear(
    reader: _ShardReader, src_prefix: str, dst_prefix: str
) -> Iterator[tuple[str, torch.Tensor]]:
    """An always-resident Linear as the checkpoint stores it: NVFP4 (GLM-4.7) or bf16 (GLM-4.5)."""
    if reader.has(f"{src_prefix}.weight_scale"):
        yield from _iter_nvfp4_resident(reader, src_prefix, dst_prefix)
    else:
        yield f"{dst_prefix}.weight", reader.get(f"{src_prefix}.weight").to(torch.bfloat16)


def _iter_nvfp4_resident(
    reader: _ShardReader, src_prefix: str, dst_prefix: str
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield native NVFP4 buffers for an always-resident Linear.

    ``weight`` (packed fp4) + ``weight_scale`` (fp8 block scale) verbatim; per-tensor
    ``weight_scale_2`` broadcast to per-row fp16 ``weight_global`` as the NVFP4 linear
    method / dequant_nvfp4 expect. lossless vs checkpoint; same dequant math as routed experts.
    """
    packed = reader.get(f"{src_prefix}.weight")  # [OUT, IN//2] uint8
    scale = reader.get(f"{src_prefix}.weight_scale")  # [OUT, IN//16] fp8-e4m3
    g = reader.get(f"{src_prefix}.weight_scale_2").reshape(()).to(torch.float16)  # scalar
    yield f"{dst_prefix}.weight", packed
    yield f"{dst_prefix}.weight_scale", scale
    yield f"{dst_prefix}.weight_global", g.expand(packed.shape[0]).contiguous()
    if reader.has(f"{src_prefix}.input_scale"):
        yield f"{dst_prefix}.input_scale", reader.get(f"{src_prefix}.input_scale").reshape(()).to(torch.float32)


def _iter_attn_df11(
    reader: _ShardReader, prefix: str, device: torch.device, dst: str | None = None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield DF11 buffers for an attention projection from its bf16 checkpoint weight.

    qkvo are bf16 in the checkpoint; DF11 compresses them losslessly (~10.7 bits/weight) to
    fit a 32 GB VRAM target, decoding bit-for-bit.
    """
    w = reader.get(f"{prefix}.weight").to(torch.bfloat16)
    bundle = compress_df11_weight(w)
    for name in ("low8", "bitstream", "chunk_start", "lut"):
        yield f"{dst or prefix}.{name}", bundle[name]


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the resident (non routed-expert) weights for GLM-4 MoE.

    - qkvo: lossless DF11 (+ bf16 .bias) so the ~25GB of bf16 attention fits a 32 GB VRAM target.
    - leading dense MLP layers + each MoE layer's shared expert: native NVFP4 (dequant in
      forward), faithful and smallest footprint.
    - embedding: row-contiguous DF11, decode only looked-up rows. router gate (+
      e_score_correction_bias), norms, lm_head pass through.
    - MTP layer (index num_layers) and routed experts are skipped.
    """
    assert not include_moe_experts, (
        "GLM-4 MoE only supports the offload backend; routed experts are loaded into the "
        "offload cache from their NVFP4 or bf16 pieces (iter_expert_pieces)."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    try:
        yield from _iter_resident_weights(reader, config, primary)
    finally:
        # drop shard page cache so the ~176GB expert banks (allocated next) start clean.
        reader.close()


def _iter_resident_weights(reader, config, primary) -> Iterator[tuple[str, torch.Tensor]]:
    device = reader._device
    dense = config.first_k_dense_replace
    # A pipeline stage reads its own layers and edges, numbering its layers from zero (OPList
    # loads by position). A whole model is the stage 0..num_layers.
    stage = get_stage_info(config.num_layers)

    for layer in tqdm(range(stage.first, stage.last), desc="Loading GLM dense weights", disable=not primary):
        src_layer = f"model.layers.{layer}"
        dst_layer = f"model.layers.{layer - stage.first}"
        a, da = f"{src_layer}.self_attn", f"{dst_layer}.self_attn"
        # DF11 projections (+ qkv bias, bias-free o_proj).
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield from _iter_attn_df11(reader, f"{a}.{proj}", device, f"{da}.{proj}")
            bias_name = f"{a}.{proj}.bias"
            if reader.has(bias_name):
                yield f"{da}.{proj}.bias", reader.get(bias_name).to(torch.bfloat16)
        for norm in ("q_norm", "k_norm"):
            name = f"{a}.{norm}.weight"
            if reader.has(name):
                yield f"{da}.{norm}.weight", reader.get(name)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            yield f"{dst_layer}.{norm}.weight", reader.get(f"{src_layer}.{norm}.weight")

        m, dm = f"{src_layer}.mlp", f"{dst_layer}.mlp"
        if layer < dense:
            # dense SwiGLU MLP: NVFP4 (GLM-4.7) or bf16 (GLM-4.5), separate gate/up/down.
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield from _iter_resident_linear(reader, f"{m}.{proj}", f"{dm}.{proj}")
        else:
            # router (bf16 gate + fp32 selection bias -> bf16) and shared expert.
            yield f"{dm}.gate.weight", reader.get(f"{m}.gate.weight")
            yield (
                f"{dm}.e_score_correction_bias",
                reader.get(f"{m}.gate.e_score_correction_bias").to(torch.bfloat16),
            )
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield from _iter_resident_linear(
                    reader, f"{m}.shared_experts.{proj}", f"{dm}.shared_experts.{proj}"
                )

    if stage.is_first:
        # bf16 embedding -> row-contiguous DF11 (~30% smaller), decode only looked-up rows.
        embed = reader.get("model.embed_tokens.weight").to(torch.bfloat16)
        for name, buf in compress_df11_embedding(embed).items():
            yield f"model.embed_tokens.{name}", buf
        del embed
    if stage.is_last:
        yield "model.norm.weight", reader.get("model.norm.weight")
        # lm_head stays bf16: full-vocab matmul needs a decode scratch as big as the weight, so
        # DF11 nets no savings (unlike the gathered embedding lookup).
        yield "lm_head.weight", reader.get("lm_head.weight")


# --------------------------------------------------------------------------------------
# Routed expert host banks (NVFP4) for the offload cache.
# --------------------------------------------------------------------------------------
def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


def iter_expert_pieces(model_path, config, kind: QuantKind, *, parallel: bool | None = False,
                       workers: int = 8, chunk: int = 8 << 20):
    """bf16 routed experts (GLM-4.5), one piece per expert: ``{gate, up, down}``; the
    unquantized bank concatenates gate|up itself. NVFP4 goes through ``nvfp4_expert_spec``."""
    if kind is not QuantKind.NONE:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("glm4_moe bf16 expert banks support TP=1 only")
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    def locate(raw_name: str):
        m = _BF16_EXPERT_RE.match(raw_name)
        if m is None:
            return None
        bank = bank_layer_of(config, int(m["layer"]))
        if bank is None:
            return None
        return bank, int(m["expert"]), m["proj"]

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        # O_DIRECT reads: nothing lands in the page cache next to the banks.
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=3)

    def _serial():
        folder = download_hf_weight(model_path)
        with open(os.path.join(folder, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        layers = [lid for lid in config.local_layer_ids if lid >= config.first_k_dense_replace]
        for layer in tqdm(layers, desc="Loading GLM bf16 experts (serial)", disable=not get_tp_info().is_primary()):
            reader = _ShardReader(folder, weight_map, torch.device("cpu"))
            try:
                for e in range(config.num_experts):
                    for proj in ("gate", "up", "down"):
                        name = f"model.layers.{layer}.mlp.experts.{e}.{proj}_proj.weight"
                        yield name, reader.get(name)
            finally:
                # the banks and the checkpoint's page cache do not fit in RAM together:
                # drop this layer's shards before the next one is read
                reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=3)


__all__ = ["iter_expert_pieces", "iter_weights", "nvfp4_expert_spec"]
