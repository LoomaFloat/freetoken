"""qwen4_exp routed experts from the bf16 release, whose layout no other reader can see.

``Qwen/Qwen3.8-Flash-Next`` stacks a whole layer's experts into two tensors instead of
naming each expert, so the two paths that serve the quantized releases both miss it: the
NVFP4 key pattern wants an expert number, and the shared ``stacked_expert_pieces`` wants
``model.layers.N...`` keys, not this multimodal checkpoint's ``model.language_model.``.

The synthetic checkpoint here is the released one's shape, not its size: same key names,
same dtype, same ``[E, 2 * I, H]`` / ``[E, H, I]`` geometry.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.qwen4_exp import iter_expert_pieces
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.weight import iter_weights, load_ple_table
from freetoken.utils import cached_load_hf_config

from .common import hf_config, install_quant_config

# The released geometry, shrunk: only the expert axes matter here, but the rest has to be
# consistent enough for parse_config to build the same two layer kinds the real config has.
H = 128                  # hidden_size
HC, LR = 4, 320          # hc_count, hc_lowrank
KH, VH, HD = 2, 4, 32    # GDN key / value heads, head dim
QH, KVH, AHD = 4, 2, 64  # QSA q / kv heads, head dim
IHD = 64                 # indexer head dim
E, I = 3, 6              # routed experts, moe_intermediate_size
LAYERS = 2
LM = "model.language_model"


@pytest.fixture(scope="module", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _stacked_checkpoint(folder, *, gate_up=None, down=None, drop=()) -> tuple[str, dict[str, torch.Tensor]]:
    """A bf16 release in miniature: stacked routed experts, no quantization_config.

    ``gate_up`` / ``down`` override the shapes and ``drop`` omits keys, for the checks that a
    wrong checkpoint is refused by name instead of quietly reshaped or half-loaded.
    """
    raw: dict[str, torch.Tensor] = {
        f"{LM}.embed_tokens.weight": _bf16(11, H),
        "lm_head.weight": _bf16(11, H),
    }
    for layer in range(LAYERS):
        mlp = f"{LM}.layers.{layer}.mlp"
        raw[f"{mlp}.experts.gate_up_proj"] = _bf16(*(gate_up or (E, 2 * I, H)))
        raw[f"{mlp}.experts.down_proj"] = _bf16(*(down or (E, H, I)))
        raw[f"{mlp}.gate.weight"] = _bf16(E, H)
        raw[f"{mlp}.shared_expert.gate_proj.weight"] = _bf16(I, H)
        raw[f"{mlp}.shared_expert.up_proj.weight"] = _bf16(I, H)
        raw[f"{mlp}.shared_expert.down_proj.weight"] = _bf16(H, I)
        raw[f"{mlp}.shared_expert_gate.weight"] = _bf16(1, H)
    # The MTP head stacks its experts the same way and must stay invisible either way.
    raw["mtp.layers.0.mlp.experts.gate_up_proj"] = _bf16(E, 2 * I, H)
    raw["mtp.layers.0.mlp.experts.down_proj"] = _bf16(E, H, I)
    for key in drop:
        raw.pop(key)

    names = sorted(raw)
    # Two shards, so a layer's gate_up and down can land in different files — they do in the
    # real checkpoint, where one gate_up_proj is 3.36 GiB and fills a shard on its own.
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    cfg = hf_config(
        num_layers=LAYERS, head_dim=AHD, num_q=QH, num_kv=KVH, index_head_dim=IHD,
        index_heads=2, budget=16, hidden=H, max_position=4096, rope_theta=10000.0,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=KH, linear_num_value_heads=VH,
        linear_key_head_dim=HD, linear_value_head_dim=HD,
        hc_lowrank=LR, ple_layer_ids=[1],
        num_experts=E, moe_intermediate_size=I, shared_expert_intermediate_size=I,
    )
    (folder / "config.json").write_text(
        json.dumps({**vars(cfg), "text_config": vars(cfg.text_config), "quantization_config": None})
    )
    return str(folder), raw


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return _stacked_checkpoint(tmp_path_factory.mktemp("qwen4_exp_bf16"))


def _config(path: str):
    install_quant_config(path)
    return parse_config(cached_load_hf_config(path))


def _pieces(path: str, kind: QuantKind = QuantKind.NONE):
    return list(iter_expert_pieces(path, _config(path), kind))


# ------------------------------------------------------------------ плотный проход


def test_stacked_experts_never_reach_the_dense_pass(checkpoint):
    """The regression that made the bf16 release unloadable.

    ``_EXPERT_RE`` matched only the numbered form, so the stacked tensors fell through to
    the dense iterator, which then tried to materialize every routed expert as dense state:
    48 layers x 5.03 GiB on the real checkpoint.
    """
    path, _raw = checkpoint
    install_quant_config(path)
    dense = [
        name for name, _ in iter_weights(
            path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True
        )
    ]

    assert dense, "the dense pass must still yield the non-expert weights"
    assert [n for n in dense if ".mlp.experts." in n] == []
    assert [n for n in dense if n.startswith("mtp.")] == []
    # The shared expert is dense and must survive: it is not a routed expert.
    assert any(n.endswith(".mlp.shared_expert.gate_up_proj.weight") for n in dense)


# ------------------------------------------------------------------ куски


def test_one_piece_per_layer_covering_every_expert(checkpoint):
    path, _raw = checkpoint
    pieces = _pieces(path)

    assert [(layer, e0, e1) for layer, e0, e1, _ in pieces] == [(0, 0, E), (1, 0, E)]


def test_piece_shapes_are_the_expert_kernel_bank_layout(checkpoint):
    """``UnquantizedMoEMethod.layout`` asks for ``(2 * I, H)`` and ``(H, I)`` per expert, and
    the checkpoint already stores exactly that stacked. Nothing here may reshape."""
    path, _raw = checkpoint

    for _layer, _e0, _e1, piece in _pieces(path):
        assert piece["gate_up"].shape == (E, 2 * I, H)
        assert piece["down"].shape == (E, H, I)
        assert piece["gate_up"].dtype is torch.bfloat16
        assert piece["down"].dtype is torch.bfloat16


def test_pieces_are_the_checkpoint_bytes_untouched(checkpoint):
    """No transpose, no re-order, no cast: a silent permutation here would not crash, it
    would return wrong tokens."""
    path, raw = checkpoint

    for layer, _e0, _e1, piece in _pieces(path):
        assert torch.equal(piece["gate_up"], raw[f"{LM}.layers.{layer}.mlp.experts.gate_up_proj"])
        assert torch.equal(piece["down"], raw[f"{LM}.layers.{layer}.mlp.experts.down_proj"])


def test_the_mtp_head_is_not_mistaken_for_a_layer(checkpoint):
    """Its experts are stacked identically; only the key prefix tells them apart."""
    path, raw = checkpoint
    mtp = raw["mtp.layers.0.mlp.experts.gate_up_proj"]

    for _layer, _e0, _e1, piece in _pieces(path):
        assert not torch.equal(piece["gate_up"], mtp)


def test_the_kernel_takes_the_fused_piece_as_is(checkpoint):
    """``pack`` calls ``fused_piece(pieces, "gate_up")``, which returns a fused piece
    unchanged and only concatenates when gate and up arrive apart. Our piece must hit the
    first branch, or the row order stops being the checkpoint's."""
    from freetoken.layers.quantization.moe.base import fused_piece

    path, _raw = checkpoint
    _layer, _e0, _e1, piece = _pieces(path)[0]

    assert fused_piece(piece, "gate_up") is piece["gate_up"]


# ------------------------------------------------------------------ развилка по формату


def test_nvfp4_still_goes_its_own_way(checkpoint):
    """None here is what sends the caller to ``nvfp4_expert_spec`` — the path the working
    NVFP4 deployment uses today."""
    path, _raw = checkpoint

    assert iter_expert_pieces(path, _config(path), QuantKind.NVFP4) is None


def test_block_fp8_is_delegated_not_reimplemented(checkpoint, monkeypatch):
    """The official fp8 release is qwen3_5_moe's job and must stay so."""
    path, _raw = checkpoint
    seen = {}

    def spy(model_path, config, kind, **kwargs):
        seen.update(model_path=model_path, kind=kind, **kwargs)
        return iter(())

    monkeypatch.setattr("freetoken.models.qwen3_5_moe.weight.iter_expert_pieces", spy)
    out = iter_expert_pieces(path, _config(path), QuantKind.FP8_BLOCK, parallel=True, workers=3)

    assert list(out) == []
    assert seen["kind"] is QuantKind.FP8_BLOCK
    assert seen["parallel"] is True and seen["workers"] == 3


# ------------------------------------------------------------------ битый чекпоинт


def test_a_wrong_expert_count_is_refused_by_name(tmp_path):
    path, _raw = _stacked_checkpoint(tmp_path, gate_up=(E + 1, 2 * I, H))

    with pytest.raises(ValueError, match=r"experts\.gate_up_proj is \(4, 12, 128\)"):
        _pieces(path)


def test_a_transposed_projection_is_refused(tmp_path):
    """Right number of bytes, wrong axes: the check is on the shape, not the size."""
    path, _raw = _stacked_checkpoint(tmp_path, down=(E, I, H))

    with pytest.raises(ValueError, match=r"experts\.down_proj is \(3, 6, 128\)"):
        _pieces(path)


def test_a_missing_layer_names_the_tensor(tmp_path):
    path, _raw = _stacked_checkpoint(tmp_path, drop=(f"{LM}.layers.1.mlp.experts.down_proj",))

    with pytest.raises(ValueError, match=r"no stacked routed experts at .*layers\.1\.mlp\.experts\.down_proj"):
        _pieces(path)


# ------------------------------------------------------------------ таблица n-грамм

NGRAM_DIM, NGRAM_ROWS, NGRAM_SHARDS = 4, 7, 4
PLE = f"{LM}.layers.1.ple.ple_embedding.ngram_embedding"
PLE_ARGS = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)


def _table_checkpoint(folder, *, fp8: bool) -> tuple[str, dict[str, torch.Tensor]]:
    """Only the n-gram table; the released bf16 one ships no ``weight_scale`` at all."""
    shards = {
        f"{PLE}.shard_{i}.weight": (
            torch.arange(i * NGRAM_ROWS * NGRAM_DIM, (i + 1) * NGRAM_ROWS * NGRAM_DIM)
            .remainder(200).to(torch.uint8).view(NGRAM_ROWS, NGRAM_DIM)
        )
        for i in range(NGRAM_SHARDS)
    }
    if fp8:
        shards = {k: v.view(torch.float8_e4m3fn) for k, v in shards.items()}
        shards[f"{PLE}.weight_scale"] = torch.tensor([0.125], dtype=torch.bfloat16)
    else:
        shards = {k: v.to(torch.bfloat16) for k, v in shards.items()}
    save_file(shards, str(folder / "model-ple-00000.safetensors"))
    return str(folder), shards


def test_a_bf16_table_loads_without_a_scale(tmp_path):
    """The released bf16 table stores values, not codes: there is no ``weight_scale`` tensor
    to find, and refusing over its absence is what stopped the load."""
    path, raw = _table_checkpoint(tmp_path, fp8=False)

    table = load_ple_table(path, PLE_ARGS, pin=False)

    assert table.tensor.shape == (NGRAM_SHARDS * NGRAM_ROWS, NGRAM_DIM)
    assert table.tensor.dtype is torch.bfloat16
    assert float(table.weight_scale) == 1.0


def test_the_bf16_table_is_assembled_in_shard_order(tmp_path):
    """Two bytes per value, so the per-shard byte offsets are twice the fp8 ones. Getting
    that wrong reads every shard into the wrong half of the bank."""
    path, raw = _table_checkpoint(tmp_path, fp8=False)

    table = load_ple_table(path, PLE_ARGS, pin=False)

    for shard in range(NGRAM_SHARDS):
        rows = table.tensor[shard * NGRAM_ROWS: (shard + 1) * NGRAM_ROWS]
        assert torch.equal(rows, raw[f"{PLE}.shard_{shard}.weight"])


def test_an_fp8_table_still_needs_its_scale(tmp_path):
    """The bf16 branch must not become a silent default for a broken fp8 checkpoint: codes
    without their scale are unusable, and 1.0 would be a wrong answer, not a missing one."""
    path, raw = _table_checkpoint(tmp_path, fp8=True)
    del raw[f"{PLE}.weight_scale"]
    save_file(raw, str(tmp_path / "model-ple-00000.safetensors"))

    with pytest.raises(ValueError, match="no weight_scale"):
        load_ple_table(path, PLE_ARGS, pin=False)


def test_the_fp8_table_is_unchanged_by_the_bf16_branch(tmp_path):
    path, raw = _table_checkpoint(tmp_path, fp8=True)

    table = load_ple_table(path, PLE_ARGS, pin=False)

    assert table.tensor.dtype is torch.float8_e4m3fn
    assert float(table.weight_scale) == 0.125
    for shard in range(NGRAM_SHARDS):
        rows = table.tensor[shard * NGRAM_ROWS: (shard + 1) * NGRAM_ROWS]
        assert torch.equal(rows.view(torch.uint8),
                           raw[f"{PLE}.shard_{shard}.weight"].view(torch.uint8))


def test_a_mixed_dtype_table_is_refused(tmp_path):
    """Half fp8, half bf16 is not a layout we know; picking either would corrupt the rest."""
    path, raw = _table_checkpoint(tmp_path, fp8=False)
    raw[f"{PLE}.shard_2.weight"] = raw[f"{PLE}.shard_2.weight"].to(torch.uint8).view(torch.float8_e4m3fn)
    save_file(raw, str(tmp_path / "model-ple-00000.safetensors"))

    with pytest.raises(ValueError, match="mixes dtypes"):
        load_ple_table(path, PLE_ARGS, pin=False)
