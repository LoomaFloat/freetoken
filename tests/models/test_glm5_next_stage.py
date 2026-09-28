"""Стадия конвейера у GLM-5.3-Flash (glm5_next): свой отрезок слоёв, свои края, свои веса.

Остаток между слоями у этого семейства отложенный и широкий: слой отдаёт
четвёрку ``(x, residual, post, comb)``, а следующий складывает post с pre в
одном ядре. Через шов едет вся четвёрка, и хвост продолжает ровно с того
состояния, на котором остановилась голова. Здесь — что стадия строит, что
читает, как нумерует банки при плотном префиксе и что шов ничего не меняет.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import clear_stage_info, set_tp_info, try_get_stage_info, try_get_tp_info
from freetoken.utils.hf import RawConfigShim

LAYERS, HIDDEN, VOCAB, EXPERTS = 6, 64, 128, 8
DSA = (3, 5)                       # остальные — линейное внимание KDA
FIRST_DENSE = 1                    # слой 0 плотный, 1..5 — MoE


def _hf_config() -> RawConfigShim:
    text = {
        "hidden_size": HIDDEN, "intermediate_size": 96, "num_hidden_layers": LAYERS,
        "num_attention_heads": 2, "vocab_size": VOCAB, "hidden_act": "silu",
        "rms_norm_eps": 1e-5, "max_position_embeddings": 4096, "tie_word_embeddings": False,
        "q_lora_rank": 48, "kv_lora_rank": 32, "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 0, "v_head_dim": 32, "mla_use_nope": True,
        "index_n_heads": 16, "index_head_dim": 64, "index_topk": 32,
        "indexer_types": ["full"] * LAYERS, "indexer_rope_interleave": True,
        "index_kpool": 4, "index_kpool_compress": True, "index_kpool_always_select_tail": True,
        "linear_attn_config": {"num_heads": 2, "head_dim": 128,
                               "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        "layer_types": ["deepseek_sparse_attention" if i in DSA else "linear_attention"
                        for i in range(LAYERS)],
        "mlp_layer_types": ["dense"] * FIRST_DENSE + ["sparse"] * (LAYERS - FIRST_DENSE),
        "first_k_dense_replace": FIRST_DENSE,
        "mhc": True, "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
        "n_routed_experts": EXPERTS, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "moe_intermediate_size": 32, "norm_topk_prob": True,
        "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
        "n_group": 1, "topk_group": 1, "swiglu_limit": 10.0,
        "attention_bias": False, "model_type": "glm5_next_text",
    }
    return RawConfigShim({"architectures": ["Glm5NextForConditionalGeneration"],
                          "model_type": "glm5_next", "text_config": text})


@pytest.fixture(autouse=True)
def _runtime(monkeypatch, tmp_path):
    """Конфиг — из шима (как у движка, когда transformers не знает glm5_next), веса — ниоткуда."""
    import freetoken.engine.config as engine_config
    import freetoken.models.glm5_next.weight as weight

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    shim = _hf_config()
    monkeypatch.setattr(engine_config, "cached_load_hf_config", lambda path: shim)
    monkeypatch.setattr(engine_config, "checkpoint_quant_config", lambda *args: None)
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda path: shim)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    monkeypatch.setattr(weight, "download_hf_weight", lambda path: str(tmp_path))
    clear_stage_info()
    yield
    clear_stage_info()


def _model(span: str = ""):
    """Модель, собранная движком для этого отрезка, на meta-устройстве."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    clear_stage_info()
    config = EngineConfig(model_path="/fake", tp_info=DistributedInfo(rank=0, size=1),
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


# ------------------------------------------------------------------ модель


def test_the_whole_model_builds_as_before():
    model, config = _model()
    keys = set(model.state_dict())
    assert _layer_ids(keys) == set(range(LAYERS))
    assert {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} <= keys
    moe = [layer for layer in model.model.layers.op_list if hasattr(layer.mlp, "experts")]
    assert [layer.mlp.experts.layer_id for layer in moe] == list(range(LAYERS - FIRST_DENSE))
    assert model.produces_logits and model.stage_input_width == 0


@pytest.mark.parametrize("span, embed, head, banks", [
    ("0:3", True, False, [0, 1]),        # слой 0 плотный: банки у слоёв 1 и 2
    ("3:6", False, True, [0, 1, 2]),
    ("2:4", False, False, [0, 1]),
])
def test_a_stage_builds_its_layers_and_its_edges(span, embed, head, banks):
    model, config = _model(span)
    keys = set(model.state_dict())
    first, last = map(int, span.split(":"))
    assert _layer_ids(keys) == set(range(last - first)), "слои стадии лежат подряд с нуля"
    assert ("model.embed_tokens.weight" in keys) is embed
    assert ("model.norm.weight" in keys) is head and ("lm_head.weight" in keys) is head
    assert model.produces_logits is head
    assert model.stage_input_width == (0 if embed else model.model.stream_width)
    moe = [layer for layer in model.model.layers.op_list if hasattr(layer.mlp, "experts")]
    assert [layer.mlp.experts.layer_id for layer in moe] == banks, "кэш экспертов адресуется банком"
    assert config.num_moe_layers == len(banks)


def test_two_stages_are_the_model_cut_not_rebuilt():
    from freetoken.models.stage_weights import LAYER_KEY_RE

    whole = set(_model()[0].state_dict())
    first = set(_model("0:3")[0].state_dict())
    last = set(_model("3:6")[0].state_dict())
    shift = lambda keys, by: {LAYER_KEY_RE.sub(lambda m: f"{m['head']}{int(m['id']) + by}.", k, count=1)
                              for k in keys}
    first, last = shift(first, 0), shift(last, 3)
    assert first | last == whole and not first & last


def test_kv_groups_cover_only_the_stage_and_keep_the_indexer_slots():
    """Слоты индексатора раздаются по всем DSA-слоям модели, поэтому их число
    не сужается; сужаются только слои групп."""
    _, whole = _model()
    _, tail = _model("3:6")
    groups = {g.name: g for g in tail.attention_groups}
    assert groups["full"].layer_ids == DSA and groups["linear"].layer_ids == (4,)
    whole_full = next(g for g in whole.attention_groups if g.name == "full")
    assert groups["full"].num_index_layers == whole_full.num_index_layers


# ------------------------------------------------------------------ веса


class _FakeReader:
    """Читатель шардов без шардов: отдаёт нули и помнит, что у него просили."""

    asked: list = []

    def __init__(self, folder, weight_map, device):
        pass

    def get(self, name):
        _FakeReader.asked.append(name)
        if name.endswith(".weight_scale_inv"):
            return torch.ones(1, 1)
        if ".mlp.experts." in name:
            return torch.zeros(2, 2).to(torch.float8_e4m3fn)
        return torch.zeros(2, 2, dtype=torch.bfloat16)

    def close(self):
        pass


@pytest.mark.parametrize("span", ["", "0:3", "3:6", "2:4"])
def test_the_dense_reader_feeds_exactly_what_the_stage_built(monkeypatch, span):
    """Хвост Qwen3-MoE на стенде упал ровно на этом: модель ждала то, чего
    читалка не дала. Лишний ключ `load_state_dict` тоже не простит."""
    import freetoken.models.glm5_next.weight as weight

    monkeypatch.setattr(weight, "_ShardReader", _FakeReader)
    model, _ = _model(span)
    built = set(model.state_dict())
    read = {name for name, _ in weight.iter_weights(
        "/fake", torch.device("cpu"), include_moe_experts=False, include_non_moe=True,
        include_vision=False)}
    assert read == built


@pytest.mark.parametrize("span, layers", [("0:3", {1, 2}), ("3:6", {3, 4, 5})])
def test_the_fp8_expert_reader_takes_only_this_stage(monkeypatch, span, layers):
    """Эксперты — только своих слоёв; слой MTP (номер = num_layers) не читается никогда."""
    import freetoken.models.glm5_next.weight as weight
    from freetoken.layers.quantization import QuantKind

    monkeypatch.setattr(weight, "_ShardReader", _FakeReader)
    _FakeReader.asked = []
    _, config = _model(span)
    list(weight.iter_expert_pieces("/fake", config, QuantKind.FP8_BLOCK, parallel=False))
    asked = {int(n.split(".")[3]) for n in _FakeReader.asked}
    assert asked == layers
    assert weight._layer_to_bank(LAYERS, config) is None, "слой MTP"
    first = min(layers)
    assert [weight._layer_to_bank(lid, config) for lid in sorted(layers)] == list(range(len(layers)))
    assert weight._layer_to_bank(first - 1, config) is None


# ------------------------------------------------------------------ шов


class _Layer:
    """Слой-заменитель с настоящей арифметикой mHC и тем же контрактом.

    Настоящие KDA и DSA требуют ядер; шов от того, что считает слой, не
    зависит. hc_expand / mhc_post / hc_contract — те же функции, что у модели.
    """

    def __init__(self, global_id: int, is_last: bool, n: int = 4):
        self.s, self.is_last, self.n = 1.0 + 0.1 * global_id, is_last, n

    def forward(self, x, residual, post, comb):
        from freetoken.layers.mhc import hc_contract, hc_expand, mhc_post

        if post is None:
            if residual is None:
                residual = hc_expand(x, self.n)
        else:
            residual = mhc_post(x, residual, post, comb)
        mean = residual.float().mean(-1)                                 # [T, n]
        post = torch.tanh(mean * self.s).unsqueeze(-1)                   # [T, n, 1] fp32
        comb = torch.softmax(mean.unsqueeze(-1) * mean.unsqueeze(-2) * self.s, dim=-1)
        x = torch.tanh(residual.float().sum(1) * 0.1 * self.s).to(residual.dtype)
        if self.is_last:
            return hc_contract(mhc_post(x, residual, post, comb)), None, None, None
        return x, residual, post, comb


class _Norm:
    def forward(self, x):
        return x * 2.0


class _Embed:
    def __init__(self, table):
        self.table = table

    def forward(self, input_ids):
        return self.table[input_ids]


def _stand_in(model, first: int, table):
    inner = model.model
    count = len(inner.layers.op_list)
    inner.layers.op_list[:] = [_Layer(first + i, first + i == LAYERS - 1) for i in range(count)]
    if inner.embed_tokens is not None:
        inner.embed_tokens = _Embed(table)
    if inner.norm is not None:
        inner.norm = _Norm()
    return inner


def test_the_seam_carries_the_mhc_state_and_the_cut_changes_nothing(monkeypatch):
    import freetoken.models.glm5_next.model as family

    monkeypatch.setattr(family, "embed_input_ids", lambda embed, ids, batch: embed.forward(ids))
    monkeypatch.setattr(family, "get_global_ctx", lambda: SimpleNamespace(batch=None))
    table = torch.randn(VOCAB, HIDDEN).to(torch.bfloat16)
    ids = torch.tensor([3, 17, 42, 99])

    whole = _stand_in(_model()[0], 0, table).forward(ids)
    head = _stand_in(_model("0:3")[0], 0, table)
    tail = _stand_in(_model("3:6")[0], 3, table)

    seam = head.forward(ids)
    assert seam.dtype == torch.float32 and seam.shape == (4, head.stream_width)
    assert torch.equal(tail.forward(ids, hidden=seam), whole)


def test_packing_the_state_loses_nothing():
    model, _ = _model("0:3")
    inner = model.model
    t, n, h = 5, 4, HIDDEN
    x = torch.randn(t, h).to(torch.bfloat16)
    residual = torch.randn(t, n, h).to(torch.bfloat16)
    post, comb = torch.randn(t, n, 1), torch.randn(t, n, n)
    back = inner._unpack(inner._pack(x, residual, post, comb))
    for got, want in zip(back, (x, residual, post, comb)):
        assert got.dtype == want.dtype and got.shape == want.shape and torch.equal(got, want)


def test_the_seam_refuses_what_it_cannot_continue(monkeypatch):
    import freetoken.models.glm5_next.model as family

    monkeypatch.setattr(family, "get_global_ctx", lambda: SimpleNamespace(batch=None))
    tail = _stand_in(_model("3:6")[0], 3, torch.zeros(VOCAB, HIDDEN))
    ids = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="не получила остаток"):
        tail.forward(ids)
    with pytest.raises(ValueError, match="шириной"):
        tail.forward(ids, hidden=torch.zeros(2, HIDDEN))


def test_a_stage_is_no_longer_refused_for_this_family():
    _model("0:3")
    assert try_get_stage_info() is not None
