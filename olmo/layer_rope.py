"""
LayerRoPE: per-layer complex rotation of (shared) RMSNorm gamma, with per-pair
RoPE-style frequency modulation. Port of the "complex-rotation global depth
gates + shared gamma + RoPE-global" path in the upstream ``large-activations``
codebase (modeling_llama.py:186-443, 4038-4267), adapted to OLMo's precision
conventions so that comparisons against OLMo's stock PRE / LNS pipelines are
apples-to-apples at every model scale.

Two variants are supported (selected via :attr:`LayerRoPEConfig.norm_after`):

  * ``norm_after = False`` ("pre" variant)
      attn_norm and ff_norm are replaced with :class:`ComplexRotRMSNorm` using
      the *input* rotation params; an explicit residual gate
      ``y_mod = complex_rot(gamma_residual, mag_res, theta_res) * y_sublayer``
      is applied to each sublayer output before the residual add, using the
      *residual* rotation params.
  * ``norm_after = True`` ("norm_after" variant)
      attn_norm and ff_norm are still replaced with :class:`ComplexRotRMSNorm`
      using the input rotation params. In addition, a *post-block* norm is
      inserted after each sublayer output (also a :class:`ComplexRotRMSNorm`,
      but using the *residual* rotation params, with a dedicated
      gamma_post_block_attn / gamma_post_block_mlp). The explicit residual
      gate is then SKIPPED — the rotation is "baked into" the post-block norm.

Math:

    u(l)        = log(l + 1)                                    # 0-indexed l
    mag(l)      = alpha     + beta     * u(l)                   # scalar
    theta(l)    = exp(alpha_rot + beta_rot * u(l))              # scalar
    theta_j(l)  = theta(l) * base_freq^(-2j/H), j = 0 ... H/2-1 # vector

    For a real H-vector ``g`` with consecutive pairs (g[2j], g[2j+1])
    treated as (Re, Im) of a complex number, the rotation is
        g_rot[2j]   = exp(mag) * (g[2j]   * cos(theta_j)
                               -  g[2j+1] * sin(theta_j))
        g_rot[2j+1] = exp(mag) * (g[2j]   * sin(theta_j)
                               +  g[2j+1] * cos(theta_j))

Precision policy (matches :class:`olmo.model.RMSLayerNorm` exactly):

  * Storage: the four shared gamma vectors and the 16 schedule scalars are
    declared as FP32 :class:`nn.Parameter`. Under DDP they remain FP32 in the
    forward; under FSDP ``MixedPrecision`` they are cast to ``param_dtype``
    (typically bf16) by FSDP itself.
  * Norm math (variance, ``rsqrt``, normalize): forced to FP32 inside a
    ``torch.autocast(enabled=False)`` scope, exactly like
    :class:`RMSLayerNorm`. The normalized result is cast back to the input
    dtype before exiting the scope.
  * Gamma multiply / complex rotation / residual gate: NO defensive
    ``.to(torch.float32)`` casts. Dtype follows the same type-promotion rules
    that govern OLMo's ``self.weight * x`` (DDP: FP32 weight × bf16 normed →
    FP32; FSDP-pure: bf16 × bf16 → bf16). This is the deliberate point of
    parity with PRE / LNS — see ``ComplexRotRMSNorm.forward`` below.

The ``base_freq`` for LayerRoPE is intentionally distinct from the attention
RoPE ``rope_theta`` field on :class:`olmo.config.ModelConfig` and never feeds
into :class:`olmo.model.RotaryEmbedding`.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

__all__ = [
    "complex_rotate_pairs",
    "compute_layer_rope_factors",
    "LayerRoPESharedParams",
    "ComplexRotRMSNorm",
]


def complex_rotate_pairs(
    x: torch.Tensor,
    mag: torch.Tensor,
    theta: torch.Tensor,
    force_fp32: bool = False,
) -> torch.Tensor:
    """Apply ``exp(mag + i * theta)`` to consecutive pairs of ``x``.

    The last dimension of ``x`` must be even; pairs ``(x[..., 2j], x[..., 2j+1])``
    are treated as ``(Re, Im)`` parts of complex numbers.

    Args:
        x: Real tensor of shape ``(..., H)``.
        mag: scalar (or scalar-shaped) tensor — log-magnitude.
        theta: scalar tensor or ``(H/2,)`` tensor — rotation angle(s).
        force_fp32: when ``True``, cast ``x`` / ``mag`` / ``theta`` to FP32
            inside an autocast-disabled scope, run the full complex rotation in
            FP32, and cast the result back to the input dtype. Defeats any
            FSDP MixedPrecision bf16 cast. When ``False`` (default), the math
            runs in whatever dtype the inputs carry — same regime as OLMo's
            stock ``self.weight * x``.
    """
    H = x.shape[-1]
    if H % 2 != 0:
        raise ValueError(f"complex_rotate_pairs requires last dim to be even, got {H}")

    if force_fp32:
        og_dtype = x.dtype
        with torch.autocast(enabled=False, device_type=x.device.type):
            x_f = x.to(torch.float32)
            mag_f = mag.to(torch.float32)
            theta_f = theta.to(torch.float32)
            u = x_f[..., 0::2]
            v = x_f[..., 1::2]
            scale = torch.exp(mag_f)
            cos_t = torch.cos(theta_f)
            sin_t = torch.sin(theta_f)
            u_out = scale * (u * cos_t - v * sin_t)
            v_out = scale * (u * sin_t + v * cos_t)
            out = torch.stack([u_out, v_out], dim=-1).flatten(-2)
        return out.to(og_dtype)

    u = x[..., 0::2]  # (..., H/2)
    v = x[..., 1::2]  # (..., H/2)

    scale = torch.exp(mag)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    u_out = scale * (u * cos_t - v * sin_t)
    v_out = scale * (u * sin_t + v * cos_t)

    return torch.stack([u_out, v_out], dim=-1).flatten(-2)


def _compute_theta_vec(
    theta_scalar: torch.Tensor,
    base_freq: float,
    H: int,
) -> torch.Tensor:
    """Per-pair RoPE frequency vector ``theta * base_freq^(-2j/H)``.

    The frequency lookup ``base_freq^(-2j/H)`` is computed in the same dtype
    and on the same device as ``theta_scalar``. No defensive FP32 cast.
    """
    half_H = H // 2
    j = torch.arange(half_H, dtype=theta_scalar.dtype, device=theta_scalar.device)
    base_t = torch.tensor(
        float(base_freq), dtype=theta_scalar.dtype, device=theta_scalar.device
    )
    freqs = torch.pow(base_t, -2.0 * j / float(H))
    return theta_scalar * freqs


def compute_layer_rope_factors(
    layer_log_depth: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    alpha_rot: torch.Tensor,
    beta_rot: torch.Tensor,
    base_freq: float,
    H: int,
    force_fp32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(mag, theta_vec)`` for one (layer, branch, gate-kind) slot.

    Args:
        force_fp32: when ``True``, cast every input to FP32 inside an
            autocast-disabled scope so the per-layer scalars (alpha, beta,
            alpha_rot, beta_rot) and the layer-log-depth buffer are FP32 at
            use time, even under FSDP MixedPrecision. The returned ``mag`` and
            ``theta_vec`` are FP32. When ``False`` (default), the math runs
            in the storage dtype of the inputs.
    """
    if force_fp32:
        with torch.autocast(enabled=False, device_type=layer_log_depth.device.type):
            log_l = layer_log_depth.to(torch.float32)
            alpha_f = alpha.to(torch.float32)
            beta_f = beta.to(torch.float32)
            alpha_rot_f = alpha_rot.to(torch.float32)
            beta_rot_f = beta_rot.to(torch.float32)
            mag = alpha_f + beta_f * log_l
            theta_scalar = torch.exp(alpha_rot_f + beta_rot_f * log_l)
            theta_vec = _compute_theta_vec(theta_scalar, base_freq, H)
        return mag, theta_vec

    log_l = layer_log_depth
    mag = alpha + beta * log_l
    theta_scalar = torch.exp(alpha_rot + beta_rot * log_l)
    theta_vec = _compute_theta_vec(theta_scalar, base_freq, H)
    return mag, theta_vec


class LayerRoPESharedParams(nn.Module):
    """Top-level container for parameters shared across ALL layers under LayerRoPE.

    Lives on ``OLMo.transformer.layer_rope_params`` so it is registered exactly
    once for FSDP / DDP. Individual blocks hold a non-owning reference (set via
    ``object.__setattr__`` to bypass ``nn.Module`` child registration) and pull
    the relevant parameters at forward time.

    Parameters created (always FP32, init on ``init_device``):
      * gamma vectors of shape ``(H,)`` initialised to ones:
          - ``gamma_input``                 (used by attn_norm pre-norm)
          - ``gamma_post``                  (used by ff_norm pre-norm)
          - if ``norm_after``:
              ``gamma_post_block_attn``,
              ``gamma_post_block_mlp``
          - else:
              ``gamma_residual_attn``,
              ``gamma_residual_mlp``
      * 16 length-1 vector params for (alpha, beta, alpha_rot, beta_rot) x
        (input, residual) x (attn, mlp). Stored as ``shape=(1,)`` rather than
        scalars because FSDP rejects 0-D parameters; broadcasting in
        :func:`compute_layer_rope_factors` handles the (1,) shape transparently.
    """

    # Marker the optimizer's param-group classifier looks for so the params
    # are routed to the no-decay group along with norm weights and biases.
    _is_layer_rope_shared = True

    BRANCHES = ("attn", "mlp")
    GATE_KINDS = ("input", "residual")

    def __init__(
        self,
        d_model: int,
        norm_after: bool,
        alpha_init: float,
        beta_init: float,
        alpha_rot_init: float,
        beta_rot_init: float,
        base_freq: float,
        init_device: str = "cpu",
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"LayerRoPE requires d_model to be even, got {d_model}")
        self.d_model = int(d_model)
        self.norm_after = bool(norm_after)
        self.base_freq = float(base_freq)

        kw = dict(dtype=torch.float32, device=init_device)

        self.gamma_input = nn.Parameter(torch.ones(self.d_model, **kw))
        self.gamma_post = nn.Parameter(torch.ones(self.d_model, **kw))

        if self.norm_after:
            self.gamma_post_block_attn = nn.Parameter(torch.ones(self.d_model, **kw))
            self.gamma_post_block_mlp = nn.Parameter(torch.ones(self.d_model, **kw))
        else:
            self.gamma_residual_attn = nn.Parameter(torch.ones(self.d_model, **kw))
            self.gamma_residual_mlp = nn.Parameter(torch.ones(self.d_model, **kw))

        # FSDP rejects 0-D parameters, so the schedule scalars are stored as
        # length-1 vectors. Broadcasting in compute_layer_rope_factors collapses
        # the leading (1,) to scalar-equivalent semantics.
        for gate_kind in self.GATE_KINDS:
            for branch in self.BRANCHES:
                self.register_parameter(
                    f"alpha_{gate_kind}_{branch}",
                    nn.Parameter(torch.full((1,), float(alpha_init), **kw)),
                )
                self.register_parameter(
                    f"beta_{gate_kind}_{branch}",
                    nn.Parameter(torch.full((1,), float(beta_init), **kw)),
                )
                self.register_parameter(
                    f"alpha_rot_{gate_kind}_{branch}",
                    nn.Parameter(torch.full((1,), float(alpha_rot_init), **kw)),
                )
                self.register_parameter(
                    f"beta_rot_{gate_kind}_{branch}",
                    nn.Parameter(torch.full((1,), float(beta_rot_init), **kw)),
                )

        self._alpha_init = float(alpha_init)
        self._beta_init = float(beta_init)
        self._alpha_rot_init = float(alpha_rot_init)
        self._beta_rot_init = float(beta_rot_init)

    def reset_parameters(self) -> None:
        """Re-initialize all LayerRoPE shared parameters in place."""
        with torch.no_grad():
            self.gamma_input.fill_(1.0)
            self.gamma_post.fill_(1.0)
            if self.norm_after:
                self.gamma_post_block_attn.fill_(1.0)
                self.gamma_post_block_mlp.fill_(1.0)
            else:
                self.gamma_residual_attn.fill_(1.0)
                self.gamma_residual_mlp.fill_(1.0)
            for gate_kind in self.GATE_KINDS:
                for branch in self.BRANCHES:
                    getattr(self, f"alpha_{gate_kind}_{branch}").fill_(self._alpha_init)
                    getattr(self, f"beta_{gate_kind}_{branch}").fill_(self._beta_init)
                    getattr(self, f"alpha_rot_{gate_kind}_{branch}").fill_(self._alpha_rot_init)
                    getattr(self, f"beta_rot_{gate_kind}_{branch}").fill_(self._beta_rot_init)

    def gamma_for_pre_norm(self, branch: str) -> nn.Parameter:
        if branch == "attn":
            return self.gamma_input
        if branch == "mlp":
            return self.gamma_post
        raise ValueError(branch)

    def gamma_for_post_block_norm(self, branch: str) -> nn.Parameter:
        if not self.norm_after:
            raise RuntimeError("post_block gammas only exist when norm_after=True")
        if branch == "attn":
            return self.gamma_post_block_attn
        if branch == "mlp":
            return self.gamma_post_block_mlp
        raise ValueError(branch)

    def gamma_for_residual_gate(self, branch: str) -> nn.Parameter:
        if self.norm_after:
            raise RuntimeError("explicit residual gammas only exist when norm_after=False")
        if branch == "attn":
            return self.gamma_residual_attn
        if branch == "mlp":
            return self.gamma_residual_mlp
        raise ValueError(branch)

    def schedule_params(
        self, gate_kind: str, branch: str
    ) -> Tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
        """Return (alpha, beta, alpha_rot, beta_rot) for the requested slot."""
        if gate_kind not in self.GATE_KINDS:
            raise ValueError(gate_kind)
        if branch not in self.BRANCHES:
            raise ValueError(branch)
        return (
            getattr(self, f"alpha_{gate_kind}_{branch}"),
            getattr(self, f"beta_{gate_kind}_{branch}"),
            getattr(self, f"alpha_rot_{gate_kind}_{branch}"),
            getattr(self, f"beta_rot_{gate_kind}_{branch}"),
        )


class ComplexRotRMSNorm(nn.Module):
    """RMSNorm whose gamma is a complex-rotated, per-call (mag, theta_vec).

    Unlike :class:`olmo.model.RMSLayerNorm`, this norm owns NO affine
    parameters of its own. The caller passes ``gamma_shared`` (an ``(H,)``
    tensor) and the per-layer ``mag`` / ``theta_vec`` to ``forward()`` so that
    a single shared gamma per role can be used across all layers.

    Forward — mirrors :class:`olmo.model.RMSLayerNorm.forward` precision-wise:

        with torch.autocast(enabled=False):
            og_dtype = x.dtype
            x = x.to(torch.float32)                      # FP32
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)     # FP32
            x = x.to(og_dtype)                           # back to bf16 / og_dtype
        # exit autocast-disabled scope; outer autocast (if any) re-engages.
        gamma_eff = complex_rotate_pairs(gamma_shared, mag, theta_vec)
        return gamma_eff * x

    The ``gamma_eff * x`` multiply happens OUTSIDE the autocast-disabled
    scope, with no defensive ``.to(torch.float32)`` casts on the gamma side.
    Its output dtype therefore follows the same type-promotion rules as
    OLMo's stock RMSLayerNorm:

        DDP (params kept FP32)             FP32 gamma_eff × bf16 x → FP32
        FSDP MixedPrecision.pure (bf16)    bf16 gamma_eff × bf16 x → bf16

    This guarantees the LayerRoPE norm pipeline is byte-equivalent to
    PRE / LNS at every model scale.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        x: torch.Tensor,
        gamma_shared: torch.Tensor,
        mag: torch.Tensor,
        theta_vec: torch.Tensor,
        force_fp32: bool = False,
    ) -> torch.Tensor:
        """Forward.

        Args:
            force_fp32: when ``True``, run the entire norm + complex-rotation
                + gamma multiply in FP32 inside an autocast-disabled scope,
                explicitly casting ``gamma_shared`` / ``mag`` / ``theta_vec``
                to FP32. This defeats any FSDP MixedPrecision bf16 cast on
                the shared gamma and matches the upstream
                ``--rmsnorm_fp32`` flag. Output is cast back to the input
                dtype before returning. When ``False`` (default), only the
                variance / rsqrt math is in FP32 (matching :class:`RMSLayerNorm`)
                and the gamma multiply follows storage dtype.
        """
        if force_fp32:
            og_dtype = x.dtype
            with torch.autocast(enabled=False, device_type=x.device.type):
                x_f = x.to(torch.float32)
                variance = x_f.pow(2).mean(-1, keepdim=True)
                normed = x_f * torch.rsqrt(variance + self.eps)
                gamma_eff = complex_rotate_pairs(
                    gamma_shared.to(torch.float32),
                    mag.to(torch.float32),
                    theta_vec.to(torch.float32),
                    force_fp32=False,  # already in FP32 within this scope
                )
                out = gamma_eff * normed
            return out.to(og_dtype)

        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        gamma_eff = complex_rotate_pairs(gamma_shared, mag, theta_vec)
        return gamma_eff * x

    def reset_parameters(self) -> None:
        # No own parameters; nothing to reset.
        return None


def layer_log_depth_buffer(layer_id: int) -> torch.Tensor:
    """Helper: ``log(layer_id + 1)`` as a FP32 scalar tensor."""
    return torch.tensor(math.log(int(layer_id) + 1.0), dtype=torch.float32)
