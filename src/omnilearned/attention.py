"""MHA-compatible attention: FlexAttention for CUDA bias, varlen, and SDPA."""

import math

import torch

# Compile once, not inside each module or forward call. PyTorch 2.11 API.
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention

from omnilearned.packing import padded

select_algorithm.PRINT_AUTOTUNE = False
inductor_config.max_autotune_report_choices_stats = False

_flex_attention = torch.compile(
    flex_attention,
    fullgraph=True,
    dynamic=True,
    options={
        "max_autotune": False,
        "max_autotune_gemm": False,
        "triton.cudagraphs": False,
    },
    # mode="max-autotune-no-cudagraphs",
)


def _flex_with_bias(q, k, v, bias, key_padding_mask, scale):
    """Read the existing bias in tiles; retain its autograd connection."""
    if bias.ndim == 2:
        bias = bias[None, None]
    elif bias.ndim == 3:
        bias = bias.reshape(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    elif bias.ndim != 4:
        raise ValueError("Attention mask must have 2, 3, or 4 dimensions")
    # A view: no full-sized bias copy, including for broadcast masks.
    bias = bias.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    boolean_bias = bias.dtype == torch.bool

    def score_mod(score, b, h, i, j):
        entry = bias[b, h, i, j]
        score = torch.where(entry, -torch.inf, score) if boolean_bias else score + entry
        if key_padding_mask is not None:
            score = torch.where(key_padding_mask[b, j], -torch.inf, score)
        return score

    return _flex_attention(
        q,
        k,
        v,
        score_mod=score_mod,
        scale=scale,
        kernel_options={"BACKEND": "TRITON"},
    )


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=False,
        batch_first=True,
        qk_norm=False,
        backend="auto",
    ):
        super().__init__()
        if embed_dim % num_heads or not batch_first:
            raise ValueError(
                "Attention needs batch_first and divisible head dimensions"
            )
        self.embed_dim, self.num_heads, self.head_dim = (
            embed_dim,
            num_heads,
            embed_dim // num_heads,
        )
        self.backend, self.last_backend = backend, "uninitialized"
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        nn.init.xavier_uniform_(self.in_proj_weight)
        self.register_parameter("qk_scale", None)
        if qk_norm:
            self.enable_qk_norm()

    def enable_qk_norm(self):
        self.qk_scale = nn.Parameter(
            torch.full((self.num_heads,), math.sqrt(self.head_dim))
        )

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=False,
        offsets=None,
    ):
        if self.backend not in ("auto", "sdpa", "varlen"):
            raise ValueError("attention backend must be auto, sdpa, or varlen")
        if need_weights:
            raise ValueError("Attention weights are not materialized")
        packed = query.ndim == 2
        if query is key and key is value:
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(
                3, -1
            )
        else:
            biases = (
                (None,) * 3 if self.in_proj_bias is None else self.in_proj_bias.chunk(3)
            )
            q, k, v = [
                F.linear(x, w, b)
                for x, w, b in zip(
                    (query, key, value),
                    self.in_proj_weight.chunk(3),
                    biases,
                    strict=False,
                )
            ]
        if packed:
            if offsets is None:
                raise ValueError("Packed attention requires event offsets")
            q, k, v = [x.reshape(-1, self.num_heads, self.head_dim) for x in (q, k, v)]
        else:
            q, k, v = [
                x.reshape(
                    x.shape[0], x.shape[1], self.num_heads, self.head_dim
                ).transpose(1, 2)
                for x in (q, k, v)
            ]
        if self.qk_scale is not None:
            shape = (1, -1, 1) if packed else (1, -1, 1, 1)
            q = F.normalize(q, dim=-1) * self.qk_scale.reshape(shape).to(q.dtype)
            k = F.normalize(k, dim=-1)
        scale = 1.0 if self.qk_scale is not None else None
        eligible = (
            packed
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and attn_mask is None
            and self.head_dim % 8 == 0
            and self.head_dim <= 128
            and torch.cuda.get_device_capability(q.device)[0] >= 8
        )
        if self.backend == "varlen" and attn_mask is not None:
            raise ValueError(
                "varlen does not support an attention bias; use auto or sdpa"
            )
        if self.backend == "varlen" and packed and not eligible:
            raise ValueError(
                "varlen requires supported CUDA FP16/BF16 packed inputs without interaction bias; choose auto/sdpa for FP32"
            )
        if eligible and self.backend in ("auto", "varlen"):
            from torch.nn.attention.varlen import varlen_attn

            cu = offsets.to(torch.int32)
            maximum = int((offsets[1:] - offsets[:-1]).max())
            out = varlen_attn(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                cu,
                cu,
                maximum,
                maximum,
                scale=scale,
            )
            self.last_backend = "varlen"
            return self.out_proj(out.flatten(-2)), None
        if packed:
            q, mask = padded(q, offsets)
            k, _ = padded(k, offsets)
            v, _ = padded(v, offsets)
            q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
            key_padding_mask = ~mask.squeeze(-1)
        use_flex = self.backend == "auto" and q.is_cuda and attn_mask is not None
        if use_flex:
            out = _flex_with_bias(q, k, v, attn_mask, key_padding_mask, scale)
            self.last_backend = "flex"
        else:
            bias = attn_mask
            if bias is not None:
                if bias.ndim == 3:
                    bias = bias.reshape(
                        q.shape[0], self.num_heads, q.shape[2], k.shape[2]
                    )
                if bias.dtype == torch.bool:
                    bias = torch.zeros_like(bias, dtype=q.dtype).masked_fill(
                        bias, -torch.inf
                    )
                bias = bias.to(q.dtype)
            if key_padding_mask is not None:
                valid = ~key_padding_mask[:, None, None, :]
                bias = valid if bias is None else bias.masked_fill(~valid, -torch.inf)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=0.0, scale=scale
            )
            self.last_backend = "sdpa"
        out = out.transpose(1, 2).flatten(-2)
        if packed:
            out = out[mask.squeeze(-1)]
        return self.out_proj(out), None
