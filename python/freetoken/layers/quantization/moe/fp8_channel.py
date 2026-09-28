"""fp8 e4m3 experts with one fp32 scale per output row (llm-compressor ``strategy: channel``,
e.g. zai-org/GLM-4.5-Air-FP8); a per-tensor scale is broadcast to its rows.

Served by the block-fp8 grouped GEMMs in their per-row mode: the row scale does not depend
on k, so it is applied once after the K-loop -- the checkpoint is used as is, no requant.
"""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import BankSpec, ExpertView, MoEConfig, MoEKernel, MoEMethod, fused_piece, gated_epilogue_reason, limit_or_inf

FP8 = torch.float8_e4m3fn
BLOCK = 128  # the grouped GEMMs tile K (and prefill N) by 128


def _rows(piece: torch.Tensor, rows: int) -> torch.Tensor:
    """``[e, rows, 1]`` / ``[e, rows]`` channel scales or ``[e]`` tensor scales -> ``[e, rows]`` fp32."""
    flat = piece.reshape(piece.shape[0], -1).to(torch.float32)
    if flat.shape[1] == rows:
        return flat
    assert flat.shape[1] == 1, (tuple(piece.shape), rows)
    return flat.expand(-1, rows)


class TritonFp8ChannelMoEKernel(MoEKernel):
    name = "triton"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(cfg, resident_ok=True, tp_ok=False, cpu_ok=False, plain_silu_only=False)
        if reason:
            return reason
        reason = gated_epilogue_reason(cfg)
        if reason:
            return f"fp8 channel MoE kernel: {reason}"
        if cfg.apply_router_weight_on_input:
            return "fp8 channel MoE kernel cannot apply the router weight on the input"
        if cfg.hidden % BLOCK or cfg.intermediate % BLOCK:
            return f"fp8 channel MoE kernel needs hidden and intermediate divisible by {BLOCK}"
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((2 * i, h), FP8),
            "gate_up_scale": BankSpec((2 * i,), torch.float32),
            "down": BankSpec((h, i), FP8),
            "down_scale": BankSpec((h,), torch.float32),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        i, h = cfg.intermediate, cfg.hidden
        out["gate_up"].copy_(fused_piece(pieces, "gate_up"))
        out["down"].copy_(pieces["down"])
        if "gate_up_scale" in pieces:
            out["gate_up_scale"].copy_(_rows(pieces["gate_up_scale"], 2 * i))
        else:
            out["gate_up_scale"].copy_(torch.cat([_rows(pieces["gate_scale"], i), _rows(pieces["up_scale"], i)], dim=1))
        out["down_scale"].copy_(_rows(pieces["down_scale"], h))
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.kernel.triton.fp8_blockscale_moe import fused_experts_decode_fp8_blockscale, fused_experts_fp8_blockscale

        t = view.tensors
        alpha, limit = float(layer.alpha), limit_or_inf(layer)
        if is_prefill:
            n = view.n if view.n is not None else layer.num_experts
            return fused_experts_fp8_blockscale(
                x, t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"], topk_weights, topk_ids, n,
                layer.activation, alpha, limit, per_row_scale=True,
            )
        return fused_experts_decode_fp8_blockscale(
            x, t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"], topk_weights, topk_ids,
            layer.activation, alpha, limit, per_row_scale=True,
        )


@register_method(QuantKind.FP8_TENSOR, LayerKind.MOE)
class Fp8ChannelMoEMethod(MoEMethod):
    candidates = (TritonFp8ChannelMoEKernel,)

    def create_weights(self, layer) -> None:
        g = self.cfg
        e, i, h = g.num_experts, g.intermediate, g.hidden
        layer.gate_up_proj = torch.empty(e, 2 * i, h, dtype=FP8)
        layer.gate_up_scale = torch.empty(e, 2 * i, dtype=torch.float32)
        layer.down_proj = torch.empty(e, h, i, dtype=FP8)
        layer.down_scale = torch.empty(e, h, dtype=torch.float32)

    def resident_view(self, layer) -> ExpertView:
        return ExpertView({
            "gate_up": layer.gate_up_proj, "gate_up_scale": layer.gate_up_scale,
            "down": layer.down_proj, "down_scale": layer.down_scale,
        })
