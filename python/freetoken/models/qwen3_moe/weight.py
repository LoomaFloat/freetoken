from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    MergeRule,
    iter_merged_tensors,
    iter_stacked_experts,
    iter_weight_files,
    shard_tensor,
)
from freetoken.models.stage_weights import stage_keeps, stage_renumber
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
#: Края стадии конвейера: эмбеддинги нужны только первой, финальная норма и
#: lm_head — только последней (см. `stage_keeps`).
_FIRST_ONLY = ("model.embed_tokens",)
_LAST_ONLY = ("model.norm.", "lm_head")
_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()
    # Стадия конвейера: чужие слои из пограничных шардов не читаются вовсе.
    keep = stage_keeps(config.num_layers, first_only=_FIRST_ONLY, last_only=_LAST_ONLY)

    def sharded_tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading weights",
            disable=not tp_info.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = raw_name.removeprefix("language_model.")
                    if keep is not None and not keep(name):
                        continue
                    is_expert = _EXPERT_PATTERN.match(name) is not None
                    if is_expert and not include_moe_experts:
                        continue
                    if not is_expert and not include_non_moe:
                        continue

                    raw = f.get_tensor(raw_name)
                    tensor = shard_tensor(
                        name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    del raw
                    yield name, tensor

    merged = iter_merged_tensors(
        sharded_tensors(),
        _MERGE_RULES,
        model_name="qwen3_moe",
    )
    stacked = iter_stacked_experts(
        merged,
        num_experts=config.num_experts,
        model_name="qwen3_moe",
        expert_pattern=_EXPERT_PATTERN,
    )
    # Номера слоёв — в местные, но только для state dict модели: читалке
    # банков экспертов (один `include_moe_experts`) нужен глобальный номер.
    renumber = stage_renumber(config.num_layers) if include_non_moe else None
    if renumber is None:
        yield from stacked
    else:
        for name, tensor in stacked:
            yield renumber(name), tensor


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only iter_weights: raw experts read via the common chunked multi-threaded
    O_DIRECT reader, then same merge+stack pipeline."""
    assert include_moe_experts and not include_non_moe, (
        "qwen3_moe parallel reader is experts-only (used by the expert piece reader)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()
    keep = stage_keeps(config.num_layers)

    def _is_expert(raw_name: str) -> bool:
        name = raw_name.removeprefix("language_model.")
        return _EXPERT_PATTERN.match(name) is not None and (keep is None or keep(name))

    def raw_experts() -> Iterator[tuple[str, torch.Tensor]]:
        for raw_name, raw in iter_expert_tensors_parallel(
            model_path, _is_expert, workers=workers, chunk=chunk
        ):
            name = raw_name.removeprefix("language_model.")
            tensor = shard_tensor(
                name, raw, rank=tp_info.rank, world_size=tp_info.size,
                num_kv_heads=config.num_kv_heads,
            )
            yield name, tensor

    merged = iter_merged_tensors(raw_experts(), _MERGE_RULES, model_name="qwen3_moe")
    yield from iter_stacked_experts(
        merged, num_experts=config.num_experts, model_name="qwen3_moe",
        expert_pattern=_EXPERT_PATTERN,
    )


__all__ = ["iter_weights", "iter_weights_parallel"]
