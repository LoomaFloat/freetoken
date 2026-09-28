"""GLM-5.3-Flash (glm5_next) model: hybrid KDA/DSA decoder with mHC residual streams.

Layer layout comes from the checkpoint's ``layer_types`` (34 KDA linear-attention
layers, 11 NoPE-MLA/DSA layers at 3:1) and ``mlp_layer_types`` (3 dense + 42 MoE).
The residual stream is mHC-widened to ``hc_mult`` (4) parallel streams:

    layer 0:  residual = hc_expand(x);  (post, comb, x) = mhc_pre(residual, hc_attn_*)
    each sublayer boundary fuses the previous hc_post with the next hc_pre
    (mhc_fused_post_pre), and the sublayer input is RMS-normed AFTER the mix
    (the reference fuses the norm into its hc kernels; decomposed here, same math).
    last layer: x = mhc_post(...); x = hc_contract(x)  -> final norm -> lm_head.

The deferred (post, comb) pair threads through the layer loop exactly like
glm_moe_dsa's (x, residual) pair. lm_head quant mirrors glm_moe_dsa (optional
W8A16 fp8 at load; the ~1.2 GiB bf16 head is read every decode step).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from freetoken.layers.mhc import hc_contract, hc_expand, mhc_fused_post_pre, mhc_post, mhc_pre
from freetoken.models.blocks import BaseLLMModel, embed_input_ids
from freetoken.utils import nvtx_annotate

from .attention import Glm5NextAttention
from .kda import Glm5NextKDA
from .mlp import Glm5NextGatedMLP
from .moe import Glm5NextSparseBlock
from .vision import Glm5NextVisionModel

if TYPE_CHECKING:
    from freetoken.message import MMItem
    from freetoken.models.config import ModelConfig


class Glm5NextDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.glm5_args
        self._layer_id = layer_id
        self._is_last = layer_id == config.num_layers - 1
        self.mhc = args.mhc
        self._n = args.mhc_num_residual_streams
        self._hc_eps = args.hc_eps
        self._rms_eps = args.norm_eps
        self._post_mult = args.mhc_post_mult_value
        self._sinkhorn = args.mhc_sinkhorn_iterations

        if args.is_kda_layer(layer_id):
            self.self_attn: BaseOP = Glm5NextKDA(config, layer_id, prefix=f"{prefix}.self_attn")
        else:
            self.self_attn = Glm5NextAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = Glm5NextSparseBlock(config, layer_id, prefix=f"{prefix}.mlp")
        else:
            self.mlp = Glm5NextGatedMLP(
                config.hidden_size, config.intermediate_size, swiglu_limit=config.swiglu_limit,
                quant_config=config.quant, prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(size=config.hidden_size, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(size=config.hidden_size, eps=args.norm_eps)

        if self.mhc:
            n, hidden = self._n, config.hidden_size
            mix = 2 * n + n * n
            # fp32 mHC weights (models/weight.py exempts hc_* from the dtype downcast).
            self.hc_attn_fn = torch.empty(mix, n * hidden, dtype=torch.float32)
            self.hc_attn_base = torch.empty(mix, dtype=torch.float32)
            self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
            self.hc_ffn_fn = torch.empty(mix, n * hidden, dtype=torch.float32)
            self.hc_ffn_base = torch.empty(mix, dtype=torch.float32)
            self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)

    def _pre(self, residual, fn, scale, base):
        # Layer 0's standalone pre rides the fused kernel too (HAS_POST=False
        # path; x/post/comb are the no-post sentinels) -- same dispatch, same
        # numerics, and the kernel wins at every batch size (see layers/mhc.py).
        if residual.is_cuda:
            _, post, comb, x = mhc_fused_post_pre(
                residual.new_empty(residual.shape[0], residual.shape[-1]),
                residual, None, None, fn, scale, base,
                self._rms_eps, self._hc_eps, self._post_mult, self._sinkhorn,
            )
            return post, comb, x
        return mhc_pre(
            residual, fn, scale, base,
            self._rms_eps, self._hc_eps, self._post_mult, self._sinkhorn,
        )

    def _fused(self, x, residual, post, comb, fn, scale, base):
        return mhc_fused_post_pre(
            x, residual, post, comb, fn, scale, base,
            self._rms_eps, self._hc_eps, self._post_mult, self._sinkhorn,
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if post is None:
            if residual is None:
                residual = hc_expand(x, self._n)
            post, comb, x = self._pre(
                residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        else:
            residual, post, comb, x = self._fused(
                x, residual, post, comb,
                self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            )
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x)

        residual, post, comb, x = self._fused(
            x, residual, post, comb,
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
        )
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)

        if self._is_last:
            x = mhc_post(x, residual, post, comb)
            return hc_contract(x), None, None, None
        return x, residual, post, comb


class Glm5NextModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        # Стадия конвейера строит только свои слои, а края — тем, кому они
        # достались: эмбеддинги первой, финальную норму последней. Без отрезка
        # всё это есть, и модель собирается ровно как раньше.
        whole = config.owns_first_layer and config.owns_last_layer
        if not whole and not config.glm5_args.mhc:
            raise ValueError("стадия конвейера glm5_next рассчитана на остаток mHC (hc_mult > 1)")
        self._n = config.glm5_args.mhc_num_residual_streams
        self._hidden = config.hidden_size
        # x и потоки остатка едут через шов во fp32 и возвращаются в тип модели;
        # модель строят под torch_dtype движка, он же и тип по умолчанию здесь.
        self._dtype = torch.get_default_dtype()
        self.embed_tokens = (
            VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            )
            if config.owns_first_layer
            else None
        )
        self.layers = OPList(
            [Glm5NextDecoderLayer(config, i, prefix=f"{prefix}.layers.{i}") for i in config.local_layer_ids]
        )
        self.norm = (
            RMSNorm(size=config.hidden_size, eps=config.rms_norm_eps)
            if config.owns_last_layer
            else None
        )

    @property
    def stream_width(self) -> int:
        """Ширина остатка на шве: x, n потоков, post и comb (см. `_pack`)."""
        n, h = self._n, self._hidden
        return h + n * h + n + n * n

    def _pack(self, x, residual, post, comb) -> torch.Tensor:
        """Отложенное состояние mHC на границе слоёв -> один тензор fp32.

        Едет вся четвёрка, а не применённый к остатку post. Следующий слой
        складывает post с pre в одном ядре, и с тем же состоянием на входе
        хвост посчитает ровно то, что посчитала бы целая модель, как бы ядро ни
        округляло внутри. bf16 -> fp32 -> bf16 без потерь; post и comb и так fp32.
        """
        t = x.shape[0]
        return torch.cat(
            (x.float(), residual.reshape(t, -1).float(),
             post.reshape(t, -1).float(), comb.reshape(t, -1).float()),
            dim=-1,
        )

    def _unpack(self, hidden: torch.Tensor):
        n, h, t = self._n, self._hidden, hidden.shape[0]
        x, residual, post, comb = hidden.split((h, n * h, n, n * n), dim=-1)
        return (
            x.to(self._dtype).contiguous(),
            residual.reshape(t, n, h).to(self._dtype).contiguous(),
            post.reshape(t, n, 1).to(torch.float32).contiguous(),
            comb.reshape(t, n, n).to(torch.float32).contiguous(),
        )

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Свои слои. ``hidden`` — состояние предыдущей стадии (см. `_pack`)."""
        residual = post = comb = None
        if hidden is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "стадия без эмбеддингов не получила остаток предыдущей: "
                    "его кладут в batch.stage_hidden, и считать без него нечего"
                )
            x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        elif hidden.shape[-1] != self.stream_width:
            raise ValueError(
                f"остаток шириной {hidden.shape[-1]}, ожидалось {self.stream_width} "
                f"(x, {self._n} потоков mHC, post и comb)"
            )
        else:
            x, residual, post, comb = self._unpack(hidden)
        for layer in self.layers.op_list:
            x, residual, post, comb = layer.forward(x, residual, post, comb)
        if self.norm is None:
            return self._pack(x, residual, post, comb)
        return self.norm.forward(x)


class Glm5NextForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        if config.tie_word_embeddings and config.owns_last_layer and not config.owns_first_layer:
            # lm_head берёт веса у эмбеддингов, а они на первой стадии.
            raise ValueError(
                "связанные эмбеддинги (tie_word_embeddings) на конвейере не поддержаны: "
                "lm_head последней стадии берёт веса у эмбеддингов первой"
            )
        self._config = config
        self.model = Glm5NextModel(config)
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

    @property
    def stage_input_width(self) -> int:
        """Ширина остатка на входе. Ноль у первой стадии и у целой модели."""
        return 0 if self.model.embed_tokens is not None else self.model.stream_width

    @property
    def produces_logits(self) -> bool:
        """Логиты есть только у последней стадии; остальные отдают остаток."""
        return self.lm_head is not None

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing hook: materialize the DSA layers' bmm-ready
        kv_b splits and free the checkpoint-layout originals (glm_moe_dsa
        precedent)."""
        for layer in self.model.layers.op_list:
            if isinstance(layer.self_attn, Glm5NextAttention):
                layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        hidden = getattr(batch, "stage_hidden", None)
        output = self.model.forward(batch.input_ids, hidden=hidden)
        if self.lm_head is None:
            return output
        return self.lm_head.forward(output)


class Glm5NextForConditionalGeneration(Glm5NextForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            self.visual = Glm5NextVisionModel(config.vision_config)

    def place_encoder_weights(self, mode: str) -> None:
        self.visual.place_weights(mode)

    def encode(self, item: MMItem) -> torch.Tensor:
        return self.visual.forward(item.feature, [item.grid_thw])


__all__ = ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]
