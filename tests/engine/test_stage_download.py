"""Что стадия скачивает: свои слои, а не весь чекпоинт.

На Qwen3.8-Flash-Next целый чекпоинт это 335.3 ГиБ, и без отбора каждая
стадия тянула бы его целиком — 670 ГиБ на два узла и столько же на дисках.
Стадия знает свой отрезок, индекс знает, какой тензор в каком шарде, значит
набор шардов считается точно.

Правило нарочно осторожное: отбрасывается только то, что ТОЧНО принадлежит
слоям другой стадии. Ошибка в другую сторону — это модель, которая молча
стартует на половине весов.
"""

from __future__ import annotations

import pytest

from freetoken.distributed import clear_stage_info, set_stage_info
from freetoken.utils.hf import stage_shards

LM = "model.language_model"


@pytest.fixture(autouse=True)
def _clean():
    clear_stage_info()
    yield
    clear_stage_info()


def _map(layers: int = 4) -> dict:
    """Индекс в миниатюре: слой на шард, плюс края и голова MTP."""
    weight_map = {
        f"{LM}.embed_tokens.weight": "edges.safetensors",
        "lm_head.weight": "edges.safetensors",
        f"{LM}.hyper_connection_mixer.hc_norm.weight": "edges.safetensors",
        "model.visual.blocks.0.attn.qkv.weight": "vision.safetensors",
    }
    for layer in range(layers):
        weight_map[f"{LM}.layers.{layer}.mlp.experts.gate_up_proj"] = f"L{layer}-a.safetensors"
        weight_map[f"{LM}.layers.{layer}.mlp.experts.down_proj"] = f"L{layer}-b.safetensors"
        weight_map[f"{LM}.layers.{layer}.mlp.gate.weight"] = f"L{layer}-b.safetensors"
    return weight_map


def test_the_whole_model_takes_every_shard():
    weight_map = _map()

    assert stage_shards(weight_map) == sorted(set(weight_map.values()))


def test_a_stage_takes_only_the_shards_of_its_layers():
    set_stage_info(2, 4, 4)

    got = stage_shards(_map())

    assert "L2-a.safetensors" in got and "L3-b.safetensors" in got
    assert not any(name.startswith(("L0", "L1")) for name in got)


def test_the_edges_come_to_every_stage():
    """Эмбеддинги, голова и смеситель малы рядом со слоями, а угадывать, кому
    они достались, значит рисковать моделью на половине весов."""
    set_stage_info(2, 4, 4)

    got = stage_shards(_map())

    assert "edges.safetensors" in got
    assert "vision.safetensors" in got


def test_a_shard_shared_by_two_layers_goes_to_both():
    """Резать файлы мы не умеем: шард берётся целиком, если в нём есть хоть
    один нужный тензор."""
    weight_map = {
        f"{LM}.layers.0.mlp.gate.weight": "both.safetensors",
        f"{LM}.layers.1.mlp.gate.weight": "both.safetensors",
    }
    set_stage_info(0, 1, 2)
    первая = stage_shards(weight_map)
    clear_stage_info()
    set_stage_info(1, 2, 2)
    вторая = stage_shards(weight_map)

    assert первая == вторая == ["both.safetensors"]


def test_the_mtp_head_does_not_pull_a_stage_it_does_not_belong_to():
    """`mtp.layers.0.…` содержит номер слоя, и по нему он достаётся стадии,
    у которой слой 0. Читалка его всё равно выбрасывает, но качать его на
    каждую стадию незачем."""
    weight_map = {"mtp.layers.0.mlp.experts.gate_up_proj": "mtp.safetensors",
                  f"{LM}.layers.3.mlp.gate.weight": "L3.safetensors"}
    set_stage_info(3, 4, 4)

    assert stage_shards(weight_map) == ["L3.safetensors"]


def test_a_key_without_a_layer_number_is_kept():
    from freetoken.distributed import StageInfo

    stage = StageInfo(2, 4, 4)

    assert stage.owns_key("lm_head.weight")
    assert stage.owns_key(f"{LM}.embed_tokens.weight")
    assert stage.owns_key(f"{LM}.layers.3.mlp.gate.weight")
    assert not stage.owns_key(f"{LM}.layers.1.mlp.gate.weight")
