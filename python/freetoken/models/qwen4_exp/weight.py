"""Qwen3.8-Flash-Next checkpoint reader (the NVFP4, the official block-fp8 and the bf16 releases).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_DenseFuser``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named, for the offload cache's expert reader.
* :func:`iter_expert_pieces` -- the routed experts of the other two releases: block-fp8 delegated to qwen3_5_moe, bf16 read here from the stacked per-layer tensors.

Dropped: ``mtp.*`` (speculative head, including its stacked ``mtp.layers.0.mlp.experts.*``); ``model.visual.*`` is kept only when the model built the tower.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.qwen3_vl.weight import rename_vl_prefix

from freetoken.models.config import VISION_KEY_PREFIXES
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.stage_weights import LAYER_KEY_RE, stage_keeps, stage_renumber
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.layers.quantization import QuantKind, get_quant_config
from freetoken.models.register import get_model_spec
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import cached_load_hf_config, download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.moe.expert_pieces import Piece

# Routed NVFP4 experts (nvidia modelopt layout): per-expert, un-fused. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP head's
# stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
# Routed experts in ANY of the layouts this reader knows, for the ``_rename`` skip: the
# quantized releases number the expert in the key, the bf16 release stacks a whole layer
# into one ``experts.gate_up_proj`` / ``experts.down_proj``. Matching only the numbered form
# let the stacked tensors fall through to the dense pass, which then tried to materialize
# 241 GiB of experts as dense state (Qwen/Qwen3.8-Flash-Next, 48 x 5.03 GiB).
_EXPERT_RE = re.compile(r"\.mlp\.experts\.(?:\d+\.|(?:gate_up|down)_proj$)")
# The bf16 release's stacked routed experts: one tensor per layer per projection,
# ``[num_experts, ...]``, gate and up already fused on the row axis. Anchored on
# ``model.language_model.`` for the same reason as _EXPERT_KEY_RE: it excludes the MTP
# head's ``mtp.layers.N.mlp.experts.*``, which this reader drops.
_STACKED_EXPERT_KEY = "model.language_model.layers.{layer}.mlp.experts.{leaf}"
def _nvfp4_bank(layer: int, config):
    # every layer is MoE, so for the whole model the bank is the layer; a pipeline stage
    # numbers its banks from its first layer and has none for the other stages' layers
    from freetoken.moe.expert_pieces import bank_layer_of

    return bank_layer_of(config, layer)


_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_nvfp4_bank,
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_FILE_BYTES = 4 << 30  # ple-table-*.safetensors written by ftw_side_files

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# The per-layer HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM pads the merged rows to a multiple of 16 for cuBLAS (hyperconnection.py pad_size).
# The top-level hyper_connection_mixer has no injection and never fuses.
_PAD_TO = {"input_mix_weight_down_block_inject": 16}
_HC_WITH_INJECT = (".attn_hyper_connection", ".mlp_hyper_connection")
_KIND_SUFFIXES = (".weight_scale_inv", ".weight")
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn}


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    return rename_vl_prefix(raw_name)


#: Номер слоя в УЖЕ переименованном ключе (`model.layers.7.…`).
_RENAMED_LAYER_RE = LAYER_KEY_RE


def _stage_keeps(total_layers: int):
    """Фильтр «это моё» по переименованным ключам; None — стадия одна.

    Края достаются краям: эмбеддинги первой стадии, смеситель и `lm_head`
    последней. Башня зрения идёт с эмбеддингами — она их и кормит.
    """
    return stage_keeps(
        total_layers,
        first_only=(*VISION_KEY_PREFIXES, "model.embed_tokens"),
        last_only=("lm_head", "model.hyper_connection_mixer"),
    )


def _stage_renumber(total_layers: int):
    """Глобальный номер слоя -> номер внутри стадии; None — стадия одна."""
    return stage_renumber(total_layers)


def _split_kind(name: str) -> tuple[str, str]:
    """``name`` -> ``(module, kind)``; kind is "" for tensors that are neither a weight nor a block scale."""
    for suffix in _KIND_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, ""


class _DenseFuser:
    """Concatenates checkpoint projection parts into the model's merged buffers, per kind (weight / block scale).

    The part table is the family's packed_modules_mapping. The QuantConfig picks the GDN in_proj layout and validates each part against the scheme the model built its buffer from.
    """

    def __init__(self, quant, packed: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        self.quant = quant
        self.groups = {fused: parts for fused, parts in packed if fused != "experts"}  # experts: bank reader
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        self.buf: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

    def scheme(self, module: str):
        return None if self.quant is None else self.quant.scheme_for(module)

    def _target(self, parent: str, leaf: str) -> tuple[str, int] | None:
        candidates = self.by_part.get(leaf)
        if not candidates:
            return None
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from the bf16 b|a; same test as gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
            if not candidates:
                raise ValueError(f"{parent}.{leaf}: no merged projection for the {'split' if split else 'fused'} GDN layout")
        fused, idx = candidates[0]
        if fused in _PAD_TO and not parent.endswith(_HC_WITH_INJECT):
            return None
        return f"{parent}.{fused}", idx

    def check(self, module: str, name: str, tensor: torch.Tensor) -> None:
        """``tensor`` (checkpoint key ``name``) must match the scheme the model built ``module`` from."""
        scheme = self.scheme(module)
        if name.endswith(".weight_scale_inv"):
            if scheme is None or not scheme.has("weight_scale_inv"):
                raise ValueError(f"{name}: {module} has no block scale in the checkpoint's quant config ({scheme})")
            return
        is_fp8 = tensor.dtype in _FP8_DTYPES
        if scheme is None:
            if is_fp8:
                raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
            return
        expected = _ELEM_DTYPES.get(scheme.weight.elem)
        if expected is not None and tensor.dtype is not expected:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {scheme}")
        rows, cols = (scheme.weight.group or (1, 1))
        if rows > 1 and tensor.shape[0] % rows or cols > 1 and tensor.shape[1] % cols:
            raise ValueError(f"{name}: {tuple(tensor.shape)} is not a multiple of the {rows}x{cols} scale block of {module}")

    def check_unfused(self, name: str, tensor: torch.Tensor) -> None:
        module, kind = _split_kind(name)
        if kind == ".weight_scale_inv" or (kind == ".weight" and tensor.dtype in _FP8_DTYPES):
            self.check(module, name, tensor)

    def fuse(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Buffer a part; return the merged ``[(name, tensor)]`` once its kind is complete, ``[]`` while incomplete, ``None`` if ``name`` is not a part."""
        module, kind = _split_kind(name)
        if not kind:
            return None
        parent, _, leaf = module.rpartition(".")
        hit = self._target(parent, leaf)
        if hit is None:
            return None
        fused, idx = hit
        self.check(fused, name, tensor)
        slots = self.buf.setdefault((fused, kind), {})
        slots[idx] = tensor
        parts = self.groups[fused.rpartition(".")[2]]
        if len(slots) < len(parts):
            return []
        del self.buf[(fused, kind)]
        rows = [slots[i] for i in range(len(parts))]
        pad_to = _PAD_TO.get(fused.rpartition(".")[2], 0) if kind == ".weight" else 0
        pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
        if pad:
            rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
        return [(fused + kind, torch.cat(rows, dim=0))]


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the model's state dict minus the routed experts.
    A dense projection is bf16 or 128x128 block-fp8 (``.weight`` e4m3 + ``.weight_scale_inv``) as the checkpoint's QuantConfig says: the official releases skip everything but the routed experts, the community NVFP4-FP8 requants quantize the attention / GDN projections.
    Fusions, per kind: attention q|k|v -> ``qkv_proj``; GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, or ``in_proj_qkvz`` + bf16 ``in_proj_ba`` when qkv|z is quantized; shared-expert gate|up -> ``gate_up_proj``; each per-layer HC's ``input_mix_weight_down`` | ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.
    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the routed experts always come from the offload cache's expert reader, in every layout (``nvfp4_expert_spec`` / ``iter_expert_pieces``), so the dense pass must not pick them up. It is ``_rename`` that keeps them out, and it has to know the stacked bf16 names too -- see ``_EXPERT_RE``.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return

    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    text = getattr(hf_config, "text_config", hf_config)
    mine = _stage_keeps(int(text.num_hidden_layers))
    local = _stage_renumber(int(text.num_hidden_layers))
    fuser = _DenseFuser(get_quant_config(), spec.packed_modules_mapping)
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                if not include_vision and name.startswith(VISION_KEY_PREFIXES):
                    continue
                if mine is not None and not mine(name):
                    continue
                tensor = f.get_tensor(raw_name)
                fused = fuser.fuse(name, tensor)
                if fused is None:
                    fuser.check_unfused(name, tensor)
                    fused = [(name, tensor)]
                for merged, value in fused:
                    yield (merged if local is None else local(merged)), value

    assert not fuser.buf, f"Incomplete projection fusions: {sorted(k[0] + k[1] for k in fuser.buf)}"


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is not None and name.startswith(VISION_KEY_PREFIXES):
                    yield name, f.get_tensor(raw_name)


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor scale.

    The scale is 1 for a bf16 table, which stores values outright and has no ``weight_scale``
    tensor to read. ``PinnedUVATable`` takes both dtypes and skips the dequant multiply for
    bf16, so the scale stays a plain factor here rather than an optional.
    """

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` view of the bank, in the checkpoint's dtype."""
        return self.bank.tensor


# The table's storage dtype per release: fp8 codes with a scalar scale in the quantized
# checkpoints, plain bf16 in ``Qwen/Qwen3.8-Flash-Next`` (128 shards, 102.4 GB, no scale).
# ``_PLE_ST_DTYPE`` keeps its name and meaning -- the fp8 one -- because ple_disk imports it.
_PLE_ST_DTYPE = "F8_E4M3"
_PLE_ST_DTYPES = {_PLE_ST_DTYPE: torch.float8_e4m3fn, "BF16": torch.bfloat16}


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def ftw_side_files(model_path: str, out_dir: str) -> list[str]:
    """Write the PLE n-gram table tensors, and only those, into ``ple-table-*.safetensors`` next to an FTW checkpoint.

    The table is served from safetensors files in the checkpoint dir (see load_ple_table), not from FTW entries."""
    from safetensors.torch import save_file

    folder = download_hf_weight(model_path)
    written: list[str] = []
    batch: dict[str, torch.Tensor] = {}
    size = 0

    def flush():
        nonlocal batch, size
        if batch:
            name = f"ple-table-{len(written):05d}.safetensors"
            save_file(batch, os.path.join(out_dir, name))
            written.append(name)
            batch, size = {}, 0

    for path in _ple_table_files(folder):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _PLE_TABLE_INFIX not in key:
                    continue
                t = f.get_tensor(key)
                batch[key] = t
                size += t.numel() * t.element_size()
                if size >= _PLE_FILE_BYTES:
                    flush()
    flush()
    return written


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB (fp8) or ~102.4 GB (bf16) and must not also sit in the page cache while the
    bank holds the same bytes.
    """
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    dtype: torch.dtype | None = None
    rows = cols = 0
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            stored = _PLE_ST_DTYPES.get(meta["dtype"])
            if stored is None:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")
            if dtype is not None and stored is not dtype:
                raise ValueError(f"PLE table mixes dtypes: {key} is {meta['dtype']}, expected {dtype}")
            dtype = stored
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        # An fp8 table is codes without its scale -- unusable. A bf16 table stores values
        # outright and ships no scale tensor at all, so 1 is the right factor, not a default
        # standing in for something missing.
        if dtype is not torch.bfloat16:
            raise ValueError("PLE table has no weight_scale")
        scale = torch.ones((), dtype=torch.bfloat16)

    bank = HostBank((expected * rows, cols), dtype)
    # BYTES, not elements: they coincide only for fp8. Counting bf16 rows as bytes would
    # size the bank at half the table and read every shard into the wrong offset.
    shard_bytes = rows * cols * torch.empty((), dtype=dtype).element_size()
    bar = byte_bar(expected * shard_bytes, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
            read_range_into(buf, path, file_offset=offset, nbytes=nbytes,
                            dest_offset=shard * shard_bytes, workers=workers, chunk=chunk)
            bar.update(nbytes)
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


# ======================================================================================
# Routed bf16 experts (the unquantized release)
# ======================================================================================


def _stacked_bf16_pieces(model_path: str, config) -> Iterator["Piece"]:
    """One piece per layer, straight from the checkpoint's stacked expert tensors.

    ``Qwen/Qwen3.8-Flash-Next`` stores a whole layer's routed experts as two tensors --
    ``experts.gate_up_proj`` ``[E, 2 * intermediate, hidden]`` and ``experts.down_proj``
    ``[E, hidden, intermediate]`` -- with gate and up already fused on the row axis. That is
    exactly the bf16 expert kernel's bank layout (UnquantizedMoEMethod.layout), so a piece is
    the checkpoint tensor itself: nothing is transposed, sliced or re-packed here.

    Serial by design, not as a fallback: ``experts_scattered`` calls this layout "pre-packed
    into a few large tensors", where a whole-shard parallel read only adds amplification --
    each ``gate_up_proj`` is 3.36 GiB and already fills one shard on its own.

    Page cache is dropped per shard once the consumer has packed the piece, for the reason
    ``drop_page_cache`` states: banks and the checkpoint's cache do not both fit in host RAM.
    The bf16 pool is 225 GiB, so on a host sized for it the duplicate is the difference
    between loading and being reaped. Peak extra host memory is one layer (~5.03 GiB).
    """
    from freetoken.models.loader import safetensors_weight_map

    from freetoken.moe.expert_pieces import bank_layer_of

    experts = int(config.num_experts)
    hidden = int(config.hidden_size)
    intermediate = int(config.moe_intermediate_size)
    dense = int(getattr(config, "first_k_dense_replace", 0) or 0)
    # Слои ЭТОЙ стадии, а не `range(num_moe_layers)`: у стадии конвейера банк
    # с индексом 0 — это глобальный слой 24, и брать чекпоинтные имена по
    # локальному счёту значит прочитать чужую половину модели.
    mine = [layer for layer in config.local_layer_ids if layer >= dense]
    shapes = {"gate_up": (experts, 2 * intermediate, hidden), "down": (experts, hidden, intermediate)}

    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)

    for layer in tqdm(
        mine, desc="Loading bf16 experts", disable=not get_tp_info().is_primary()
    ):
        bank_layer = bank_layer_of(config, layer)
        assert bank_layer is not None, layer
        piece: dict[str, torch.Tensor] = {}
        paths: list[str] = []
        for role, leaf in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
            name = _STACKED_EXPERT_KEY.format(layer=layer, leaf=leaf)
            shard = weight_map.get(name)
            if shard is None:
                raise ValueError(f"checkpoint has no stacked routed experts at {name}")
            path = os.path.join(folder, shard)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(name)
            if tuple(tensor.shape) != shapes[role]:
                raise ValueError(
                    f"{name} is {tuple(tensor.shape)}, expected {shapes[role]} "
                    f"(num_experts={experts}, hidden={hidden}, moe_intermediate={intermediate})"
                )
            if tensor.dtype is not torch.bfloat16:
                raise ValueError(f"{name} is {tensor.dtype}, expected bfloat16")
            piece[role] = tensor
            paths.append(path)
        yield bank_layer, 0, experts, piece
        # После yield: консьюмер уже упаковал кусок в банк, и те же байты
        # больше не нужны в кэше. Сам кусок не трогаем — он принадлежит ему.
        for path in dict.fromkeys(paths):
            drop_page_cache(path)


def iter_expert_pieces(
    model_path: str, config, kind: QuantKind, *, parallel: bool | None = False,
    workers: int = 8, chunk: int = 8 << 20,
) -> Iterator["Piece"] | None:
    """Routed experts of whichever release this checkpoint is, or None for the generic readers.

    Three layouts ship for one architecture, and only the last one is this family's own code:

    * ``FP8_BLOCK`` -- the official block-fp8 release, per-expert keys that qwen3_5_moe already
      reads (same ``model.language_model.layers.*`` dialect), so it is delegated unchanged;
    * ``NVFP4`` -- None, so the caller goes to :func:`nvfp4_expert_spec` as before;
    * ``NONE`` -- the bf16 release, whose stacked tensors no generic reader can reach: the
      shared ``stacked_expert_pieces`` wants ``model.layers.N...`` keys, and the only source it
      is fed from, ``iter_weights(include_non_moe=False)``, yields nothing here.

    The first two branches exist to keep the working paths working; the third is the new one.
    """
    if kind is QuantKind.FP8_BLOCK:
        from freetoken.models.qwen3_5_moe.weight import iter_expert_pieces as fp8_pieces

        return fp8_pieces(model_path, config, kind, parallel=parallel, workers=workers, chunk=chunk)
    if kind is not QuantKind.NONE:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp bf16 expert banks support TP=1 only")
    return _stacked_bf16_pieces(model_path, config)


__all__ = [
    "nvfp4_expert_spec",
    "PleTable",
    "iter_expert_pieces",
    "iter_weights",
    "load_ple_table",
]
