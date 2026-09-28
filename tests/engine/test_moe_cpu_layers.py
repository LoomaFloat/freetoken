"""Resolver for the hybrid CPU/GPU MoE decode split (--moe-cpu-layers).

CPU-only: exercises _parse_cpu_layers_spec / _resolve_cpu_layers without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine.engine import _parse_cpu_layers_spec as parse
from freetoken.engine.engine import _resolve_cpu_layers as resolve

L = 40


def test_explicit_list():
    assert parse("3,7,11", L) == frozenset({3, 7, 11})
    assert parse("3, 7 ,11,", L) == frozenset({3, 7, 11})  # whitespace + trailing comma
    assert parse("5,5,5", L) == frozenset({5})  # dups collapse


def test_count_evenly_strided():
    assert parse("8", L) == frozenset({0, 5, 10, 15, 20, 25, 30, 35})
    assert parse("1", L) == frozenset({0})
    assert len(parse(str(L), L)) == L  # all layers
    assert parse("0", L) == frozenset()


def test_fraction():
    assert len(parse("0.5", L)) == L // 2
    assert len(parse("1.0", L)) == L
    assert parse("0.0", L) == frozenset()


def test_empty():
    assert parse("", L) == frozenset()
    assert parse("   ", L) == frozenset()


@pytest.mark.parametrize("spec", ["99", "40,1", "-1", "1.5"])
def test_out_of_range_raises(spec):
    with pytest.raises(ValueError):
        parse(spec, L)


def _cfg(backend, spec=None):
    return SimpleNamespace(moe_strategy=backend, moe_cpu_layers=spec)


def test_resolve_backend_dispatch():
    # cpu backend -> every layer, ignoring any spec
    assert resolve(_cfg("cpu"), L) == frozenset(range(L))
    assert resolve(_cfg("cpu", "8"), L) == frozenset(range(L))
    # offload + spec -> parsed subset
    assert len(resolve(_cfg("offload", "8"), L)) == 8
    # offload, no spec -> none (plain offload)
    assert resolve(_cfg("offload", None), L) == frozenset()
    # non-offload backend ignores the spec (validation lives in _adjust_config)
    assert resolve(_cfg("fused", "8"), L) == frozenset()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


# ---- auto strategy under a pin cap (WSL): lock the over-budget layers instead of refusing ----

GiB = 2**30


def _auto_config(monkeypatch, *, budget, banks, viable=True, **overrides):
    """A bare-invocation bf16 MoE config; the pin cap, bank size and CPU executor are stubbed."""
    import torch

    import freetoken.engine.engine as engine
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.moe import bench_profile

    monkeypatch.setattr(engine, "_pin_budget_bytes", lambda reserved=0: budget)
    monkeypatch.setattr(engine, "_bank_bytes", lambda config, method=None: banks)
    monkeypatch.setattr(engine, "_cpu_moe_executor_viable", lambda model_config: viable)
    monkeypatch.setattr(bench_profile, "load_backend_recommendation", lambda *a, **k: "offload")
    config = EngineConfig(model_path="/tmp/freetoken-test-model", tp_info=DistributedInfo(rank=0, size=1),
                          dtype=torch.bfloat16, attention_backend="triton", **overrides)
    object.__setattr__(config, "model_config", SimpleNamespace(
        has_swa_attention=False, has_linear_attention=False, is_moe=True, num_layers=24,
        num_moe_layers=24, num_experts=128, expert_quant="none", hidden_act="silu",
        moe_strategy="auto"))
    return config


def test_auto_locks_cpu_layers_when_banks_exceed_the_pin_cap(monkeypatch):
    """Stand 2026-09-28: 27.0 GiB of banks against a 24.9 GiB WSL cap refused to boot."""
    from freetoken.engine.engine import _adjust_config

    config = _auto_config(monkeypatch, budget=int(24.9 * GiB), banks=27 * GiB)
    _adjust_config(config)
    assert config.moe_strategy == "offload" and config.moe_cpu_layers == "auto"
    assert config.model_config.decode_target == "cpu"


@pytest.mark.parametrize("budget, banks, viable", [
    (int(24.9 * GiB), 20 * GiB, True),    # banks fit: nothing to lock
    (None, 27 * GiB, True),               # plain Linux: no cap at all
    (int(24.9 * GiB), 27 * GiB, False),   # no CPU executor: keep the actionable refusal
])
def test_auto_leaves_cpu_layers_alone_otherwise(monkeypatch, budget, banks, viable):
    from freetoken.engine.engine import _adjust_config

    config = _auto_config(monkeypatch, budget=budget, banks=banks, viable=viable)
    _adjust_config(config)
    assert config.moe_cpu_layers is None


def test_an_explicit_strategy_is_not_second_guessed(monkeypatch):
    """Only an auto pick is adjusted; --moe-strategy offload keeps the loud pin-budget error."""
    from freetoken.engine.engine import _adjust_config

    config = _auto_config(monkeypatch, budget=int(24.9 * GiB), banks=27 * GiB, moe_strategy="offload")
    _adjust_config(config)
    assert config.moe_cpu_layers is None
