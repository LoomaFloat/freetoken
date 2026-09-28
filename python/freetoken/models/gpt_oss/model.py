from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import GptOssAttention
from .moe import GptOssMLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class GptOssDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.self_attn = GptOssAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = GptOssMLP(config, layer_id, prefix=f"{prefix}.mlp")
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


class GptOssModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        # A pipeline stage builds only its layers; the edges go to the stages that
        # own them (embeddings first, final norm last). The whole model has both.
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
                GptOssDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
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
        """Seam width: the ``(x, residual)`` pair, hidden each."""
        return 2 * self.hidden_size

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """This stage's layers. ``hidden`` is the previous stage's stream.

        The pair crosses the seam, not its sum: the next norm adds them in fp32,
        and a bf16 sum would already be rounded.
        """
        residual: torch.Tensor | None = None
        if hidden is None:
            if self.embed_tokens is None:
                raise ValueError("a stage without embeddings got no stream from the previous stage")
            x = self.embed_tokens.forward(input_ids)
        elif hidden.shape[-1] != self.stream_width:
            raise ValueError(
                f"stream is {hidden.shape[-1]} wide, expected {self.stream_width} "
                f"(x and residual, {self.hidden_size} each)"
            )
        else:
            x, residual = (part.contiguous() for part in hidden.split(self.hidden_size, dim=-1))
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        if self.norm is None:
            return torch.cat((x, residual), dim=-1)
        return self.norm.forward(x, residual)[0]

    def prepare_for_runtime(self) -> None:
        for layer in self.layers.op_list:
            layer.mlp.prepare_for_runtime()


class GptOssForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        if config.tie_word_embeddings and config.owns_last_layer and not config.owns_first_layer:
            raise ValueError(
                "tied embeddings are not supported on a pipeline: the last stage's "
                "lm_head would need the first stage's embeddings"
            )
        self.model = GptOssModel(config)
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
        self.config = config
        super().__init__()

    @property
    def stage_input_width(self) -> int:
        """Stream width this stage takes in; 0 for the first stage and the whole model."""
        return 0 if self.model.embed_tokens is not None else self.model.stream_width

    @property
    def produces_logits(self) -> bool:
        """Only the last stage has logits; the others hand the stream on."""
        return self.lm_head is not None

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        output = self.model.forward(batch.input_ids, hidden=getattr(batch, "stage_hidden", None))
        if self.lm_head is None:
            return output
        return self.lm_head.forward(output)

    def prepare_for_runtime(self) -> None:
        self.model.prepare_for_runtime()


__all__ = ["GptOssDecoderLayer", "GptOssForCausalLM", "GptOssModel"]
