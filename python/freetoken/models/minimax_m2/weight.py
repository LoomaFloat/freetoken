from __future__ import annotations

import re
from typing import Iterator, Tuple

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.loader import (
    MergeRule,
    drop_page_cache,
    iter_merged_tensors,
    iter_weight_files,
    shard_tensor,
)
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_RE = re.compile(r"\.block_sparse_moe\.experts\.\d+\.")
_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"w1": "gate", "w3": "up", "w2": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="NVFP4 experts",
)


_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
}


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """yield the dense weights as stored (bf16, or block-fp8 codes with ``weight_scale_inv``);
    routed experts (NVFP4 or block-fp8) go to the offload cache from their pieces."""
    assert not include_moe_experts, (
        "MiniMax-M2 only supports the offload MoE backend; experts are loaded into the "
        "offload cache from their pieces, not into the dense model."
    )
    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def raw() -> Iterator[tuple[str, torch.Tensor]]:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading dense weights",
            disable=not tp_info.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for name in f.keys():
                    if _EXPERT_RE.search(name):
                        continue
                    # skip attention calibration scales (FP8 kv_cache_scheme): unused with
                    # BF16 KV/attention, and ``.k_proj``/``.v_proj`` in their names make the
                    # qkv merge see incomplete groups.
                    if name.endswith((".k_scale", ".v_scale")) or (
                        ".self_attn." in name and name.endswith("_scale")
                    ):
                        continue
                    if not include_non_moe:
                        continue
                    tensor = shard_tensor(
                        name,
                        f.get_tensor(name),
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    yield name, tensor

    yield from iter_merged_tensors(raw(), _MERGE_RULES, model_name="minimax_m2")


def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


# Block-fp8 release (MiniMaxAI/MiniMax-M2.7): per-expert w1/w2/w3 fp8 codes + 128x128
# ``weight_scale_inv``; the fp8_block bank concatenates gate|up itself.
_FP8_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|weight_scale_inv)$"
)
_FP8_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}
_FP8_SUFFIX = {"weight": "", "weight_scale_inv": "_scale"}


def iter_expert_pieces(model_path: str, config, kind: QuantKind, *, parallel: bool | None = False,
                       workers: int = 8, chunk: int = 8 << 20):
    """Block-fp8 routed experts, one piece per expert: ``{gate, up, down}`` codes and their
    ``_scale`` block scales. NVFP4 goes through ``nvfp4_expert_spec``."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("minimax_m2 fp8 expert banks support TP=1 only")
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    num_layers = config.num_layers

    def locate(raw_name: str):
        m = _FP8_EXPERT_KEY_RE.match(raw_name)
        if m is None:
            return None
        layer = int(m["layer"])
        if not 0 <= layer < num_layers:
            raise ValueError(f"unexpected routed-expert layer in {raw_name}")
        return layer, int(m["expert"]), _FP8_ROLE[m["proj"]] + _FP8_SUFFIX[m["kind"]]  # every layer is MoE

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: _FP8_EXPERT_KEY_RE.match(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        for file in tqdm(
            iter_weight_files(model_path), desc="Loading fp8 experts (serial)",
            disable=not get_tp_info().is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if _FP8_EXPERT_KEY_RE.match(name):
                        yield name, f.get_tensor(name)
            # the banks and the checkpoint's page cache do not fit in RAM together
            drop_page_cache(file)

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


__all__ = ["iter_expert_pieces", "iter_weights", "nvfp4_expert_spec"]
