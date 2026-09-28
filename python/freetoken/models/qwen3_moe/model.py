from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.utils import nvtx_annotate

from freetoken.models.blocks import BaseLLMModel
from freetoken.moe.expert_pieces import bank_layer_of

from .attention import Qwen3MoeAttention as Qwen3Attn
from .moe import Qwen3MoeMLP as Qwen3MLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True, prefix=f"{prefix}.self_attn")
        # Кэш экспертов адресуется индексом банка, а банки нумерованы от стадии;
        # `layer_id` здесь глобальный. Для целой модели это одно и то же.
        bank_layer = bank_layer_of(config, layer_id)
        assert bank_layer is not None, f"слой {layer_id} собран как MoE, а банка экспертов у него нет"
        self.mlp = Qwen3MLP(config, bank_layer, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        # Стадия конвейера строит только свои слои, а края — только тем, кому
        # они достались: эмбеддинги первой, финальную норму последней. Без
        # отрезка всё это есть, и модель собирается ровно как раньше.
        self.hidden_size = config.hidden_size
        self.embed_tokens = (
            VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            )
            if config.owns_first_layer
            else None
        )
        self.layers = OPList(
            [
                Qwen3DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in config.local_layer_ids
            ]
        )
        self.norm = (
            RMSNormFused(
                size=config.hidden_size,
                eps=config.rms_norm_eps,
            )
            if config.owns_last_layer
            else None
        )

    @property
    def stream_width(self) -> int:
        """Ширина остатка на шве: пара ``(x, residual)`` по hidden каждая."""
        return 2 * self.hidden_size

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Свои слои. ``hidden`` — остаток от предыдущей стадии.

        Через шов едет пара ``(x, residual)``, а не её сумма. Остаток у этого
        декодера отложенный: следующий слой складывает их внутри своей нормы, в
        fp32. Сумма, отправленная в bf16, была бы уже округлена, и хвост
        нормировал бы не то же самое, что целая модель. Пара даёт ровно то
        состояние, на котором голова остановилась.
        """
        residual: torch.Tensor | None = None
        if hidden is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "стадия без эмбеддингов не получила остаток предыдущей: "
                    "его кладут в batch.stage_hidden, и считать без него нечего"
                )
            x = self.embed_tokens.forward(input_ids)
        elif hidden.shape[-1] != self.stream_width:
            raise ValueError(
                f"остаток шириной {hidden.shape[-1]}, ожидалось {self.stream_width} "
                f"(x и residual по {self.hidden_size})"
            )
        else:
            x, residual = (part.contiguous() for part in hidden.split(self.hidden_size, dim=-1))
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        if self.norm is None:
            return torch.cat((x, residual), dim=-1)
        return self.norm.forward(x, residual)[0]


class Qwen3MoeForCausalLM(BaseLLMModel):
    model_cls = Qwen3Model

    def __init__(self, config: ModelConfig):
        if config.tie_word_embeddings and config.owns_last_layer and not config.owns_first_layer:
            # lm_head берёт веса у эмбеддингов, а они на первой стадии.
            raise ValueError(
                "связанные эмбеддинги (tie_word_embeddings) на конвейере не поддержаны: "
                "lm_head последней стадии берёт веса у эмбеддингов первой"
            )
        self.model = self.model_cls(config)
        self.lm_head = (
            ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
                quant_config=config.quant,
                prefix="lm_head",
            )
            if config.owns_last_layer
            else None
        )
        super().__init__()

    @property
    def stage_input_width(self) -> int:
        """Ширина остатка на входе. Ноль у первой стадии и у целой модели."""
        return 0 if self.model.embed_tokens is not None else self.model.stream_width

    @property
    def produces_logits(self) -> bool:
        """Логиты есть только у последней стадии; остальные отдают остаток."""
        return self.lm_head is not None

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        hidden = getattr(batch, "stage_hidden", None)
        # Без остатка — прежний вызов: Qwen3-VL подменяет текстовую модель на
        # свою, и её forward аргумента для остатка не знает.
        if hidden is None:
            output = self.model.forward(batch.input_ids)
        else:
            output = self.model.forward(batch.input_ids, hidden=hidden)
        if self.lm_head is None:
            return output
        return self.lm_head.forward(output)


__all__ = ["Qwen3MoeForCausalLM"]
