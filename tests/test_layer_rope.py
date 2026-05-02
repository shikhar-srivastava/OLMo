"""
Smoke tests for the :mod:`olmo.layer_rope` port.

These cover the math primitives (`complex_rotate_pairs`,
`compute_layer_rope_factors`) plus an end-to-end forward/backward through a
tiny OLMo model with LayerRoPE enabled in both variants (norm_after False/True).
"""

import math

import pytest
import torch

from olmo import OLMo
from olmo.config import LayerRoPEConfig, ModelConfig
from olmo.layer_rope import (
    ComplexRotRMSNorm,
    LayerRoPESharedParams,
    complex_rotate_pairs,
    compute_layer_rope_factors,
)


def _tiny_model_config(layer_rope: LayerRoPEConfig) -> ModelConfig:
    return ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        max_sequence_length=8,
        vocab_size=64,
        embedding_size=128,
        rope=True,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        include_bias=False,
        bias_for_layer_norm=False,
        layer_norm_with_affine=True,
        weight_tying=False,
        init_device="cpu",
        layer_rope=layer_rope,
    )


def test_complex_rotate_pairs_zero_mag_zero_theta_is_identity():
    H = 8
    x = torch.randn(2, 3, H)
    mag = torch.tensor(0.0)
    theta = torch.tensor(0.0)
    out = complex_rotate_pairs(x, mag, theta)
    assert torch.allclose(out, x, atol=1e-6)


def test_complex_rotate_pairs_known_rotation():
    H = 4
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    # 90° rotation: (1, 0) -> (0, 1).
    mag = torch.tensor(0.0)
    theta = torch.tensor(math.pi / 2)
    out = complex_rotate_pairs(x, mag, theta)
    expected = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_complex_rotate_pairs_magnitude_scaling():
    H = 4
    x = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    mag = torch.tensor(math.log(2.0))  # exp(log 2) = 2
    theta = torch.tensor(0.0)
    out = complex_rotate_pairs(x, mag, theta)
    expected = 2.0 * x
    assert torch.allclose(out, expected, atol=1e-6)


def test_complex_rotate_pairs_per_pair_theta_vector():
    H = 4
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mag = torch.tensor(0.0)
    # First pair rotates 90°, second pair stays.
    theta = torch.tensor([math.pi / 2, 0.0])
    out = complex_rotate_pairs(x, mag, theta)
    expected = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_compute_layer_rope_factors_zero_init_at_layer0():
    """At α=β=αr=βr=0 and layer 0: mag=0, θ=1 (since exp(0)=1), θ_j = base^(-2j/H)."""
    H = 8
    base = 100.0
    log_l = torch.tensor(0.0)  # layer 0
    z = torch.tensor(0.0)
    mag, theta_vec = compute_layer_rope_factors(
        log_l, z, z, z, z, base_freq=base, H=H
    )
    assert torch.allclose(mag, torch.tensor(0.0), atol=1e-7)
    j = torch.arange(H // 2, dtype=torch.float32)
    expected = torch.pow(torch.tensor(base), -2.0 * j / float(H))
    assert torch.allclose(theta_vec, expected, atol=1e-6)


def test_compute_layer_rope_factors_nonzero_layer():
    """At layer 2 with α_rot=0, β_rot=0.5: θ = exp(0 + 0.5 * log(3)) = sqrt(3)."""
    H = 4
    base = 10.0
    log_l = torch.tensor(math.log(3.0))
    alpha = torch.tensor(0.0)
    beta = torch.tensor(0.5)
    alpha_rot = torch.tensor(0.0)
    beta_rot = torch.tensor(0.5)
    mag, theta_vec = compute_layer_rope_factors(
        log_l, alpha, beta, alpha_rot, beta_rot, base_freq=base, H=H
    )
    # mag = 0 + 0.5 * log(3) = log(sqrt(3))
    assert torch.allclose(mag, torch.tensor(0.5 * math.log(3.0)), atol=1e-6)
    # theta_scalar = exp(0.5 * log 3) = sqrt(3); theta_j = sqrt(3) * base^(-2j/H)
    sqrt3 = math.sqrt(3.0)
    j = torch.arange(H // 2, dtype=torch.float32)
    expected = sqrt3 * torch.pow(torch.tensor(base), -2.0 * j / float(H))
    assert torch.allclose(theta_vec, expected, atol=1e-5)


def test_complex_rot_rms_norm_at_zero_init_matches_rmsnorm_with_gamma():
    """With mag=0 and theta=0, ComplexRotRMSNorm reduces to standard gamma * RMSNorm(x).

    This is the byte-equivalence check against :class:`olmo.model.RMSLayerNorm`'s
    precision pipeline: variance / rsqrt in FP32, normalized result cast back
    to og_dtype, then ``gamma * x`` outside the autocast-disabled scope.
    """
    H = 8
    x = torch.randn(2, 3, H)
    gamma = torch.randn(H, dtype=torch.float32)
    mag = torch.tensor(0.0)
    theta = torch.zeros(H // 2)

    norm = ComplexRotRMSNorm(eps=1e-6)
    out = norm(x, gamma, mag, theta)

    # Reference mirrors RMSLayerNorm.forward exactly:
    #   FP32 variance + rsqrt, cast back to og_dtype, then gamma * x.
    og_dtype = x.dtype
    x_f = x.to(torch.float32)
    variance = x_f.pow(2).mean(-1, keepdim=True)
    normed = (x_f * torch.rsqrt(variance + 1e-6)).to(og_dtype)
    expected = gamma * normed
    assert out.dtype == expected.dtype
    assert torch.allclose(out, expected, atol=1e-6)


def test_complex_rot_rms_norm_matches_rmslayernorm_at_zero_init_under_bf16_autocast():
    """Under bf16 autocast with FP32 ``gamma`` (DDP regime), ComplexRotRMSNorm
    at zero rotation must match a freshly-built :class:`RMSLayerNorm` tensor-for-tensor.

    Verifies that the LayerRoPE norm pipeline is precision-byte-equivalent to
    OLMo's stock RMSNorm at the DDP scale (60M / 150M / 300M)."""
    from olmo.config import LayerNormType, ModelConfig
    from olmo.model import RMSLayerNorm

    if not torch.cuda.is_available():
        pytest.skip("autocast bf16 path requires CUDA")
    H = 16
    cfg = ModelConfig(
        d_model=H, n_heads=2, n_layers=1, mlp_ratio=2,
        layer_norm_type=LayerNormType.rms,
        layer_norm_with_affine=True,
        layer_norm_eps=1e-6,
        bias_for_layer_norm=False,
        include_bias=False,
    )
    rms = RMSLayerNorm(cfg, size=H).cuda()
    crn = ComplexRotRMSNorm(eps=cfg.layer_norm_eps).cuda()
    # Use the same gamma vector so the two norms are computing the same function.
    gamma = rms.weight.data.normal_().clone()
    rms.weight.data.copy_(gamma)
    mag = torch.tensor(0.0, device="cuda")
    theta = torch.zeros(H // 2, device="cuda")

    x = torch.randn(2, 3, H, device="cuda")
    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        out_rms = rms(x)
        out_crn = crn(x, gamma, mag, theta)
    assert out_rms.dtype == out_crn.dtype, (out_rms.dtype, out_crn.dtype)
    assert torch.allclose(out_rms, out_crn, atol=1e-5)


def test_complex_rot_rms_norm_dtype_follows_gamma_storage_dtype():
    """Output dtype must follow the gamma storage dtype (no defensive casts).

    DDP regime: FP32 gamma  → FP32 * bf16 normed → FP32 output (type promotion).
    FSDP regime: bf16 gamma → bf16 * bf16 normed → bf16 output.

    Mirrors OLMo's ``self.weight * x`` exactly.
    """
    H = 8
    norm = ComplexRotRMSNorm(eps=1e-6)
    x_bf16 = torch.randn(2, 3, H, dtype=torch.bfloat16)

    # DDP-style: gamma is FP32. Output must be FP32 (FP32 × bf16 promotes to FP32).
    gamma_fp32 = torch.ones(H, dtype=torch.float32)
    mag_fp32 = torch.tensor(0.0, dtype=torch.float32)
    theta_fp32 = torch.zeros(H // 2, dtype=torch.float32)
    out = norm(x_bf16, gamma_fp32, mag_fp32, theta_fp32)
    assert out.dtype == torch.float32, f"DDP path: expected FP32, got {out.dtype}"

    # FSDP-pure-style: gamma cast to bf16 by FSDP. Output must be bf16.
    gamma_bf16 = torch.ones(H, dtype=torch.bfloat16)
    mag_bf16 = torch.tensor(0.0, dtype=torch.bfloat16)
    theta_bf16 = torch.zeros(H // 2, dtype=torch.bfloat16)
    out = norm(x_bf16, gamma_bf16, mag_bf16, theta_bf16)
    assert out.dtype == torch.bfloat16, f"FSDP path: expected bf16, got {out.dtype}"


def test_layer_rope_shared_params_creates_correct_set_no_norm_after():
    sp = LayerRoPESharedParams(
        d_model=8,
        norm_after=False,
        alpha_init=0.0,
        beta_init=0.0,
        alpha_rot_init=0.0,
        beta_rot_init=0.0,
        base_freq=10000.0,
    )
    names = {n for n, _ in sp.named_parameters()}
    expected = {
        "gamma_input",
        "gamma_post",
        "gamma_residual_attn",
        "gamma_residual_mlp",
        "alpha_input_attn",
        "alpha_input_mlp",
        "alpha_residual_attn",
        "alpha_residual_mlp",
        "beta_input_attn",
        "beta_input_mlp",
        "beta_residual_attn",
        "beta_residual_mlp",
        "alpha_rot_input_attn",
        "alpha_rot_input_mlp",
        "alpha_rot_residual_attn",
        "alpha_rot_residual_mlp",
        "beta_rot_input_attn",
        "beta_rot_input_mlp",
        "beta_rot_residual_attn",
        "beta_rot_residual_mlp",
    }
    assert names == expected


def test_layer_rope_shared_params_creates_correct_set_norm_after():
    sp = LayerRoPESharedParams(
        d_model=8,
        norm_after=True,
        alpha_init=0.0,
        beta_init=0.0,
        alpha_rot_init=0.0,
        beta_rot_init=0.0,
        base_freq=10000.0,
    )
    gamma_names = {n for n, _ in sp.named_parameters() if n.startswith("gamma")}
    assert gamma_names == {
        "gamma_input",
        "gamma_post",
        "gamma_post_block_attn",
        "gamma_post_block_mlp",
    }


@pytest.mark.parametrize("norm_after", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_olmo_forward_with_layer_rope_runs_and_grads(norm_after: bool, dtype: torch.dtype):
    torch.manual_seed(0)
    layer_rope = LayerRoPEConfig(
        enabled=True,
        norm_after=norm_after,
        alpha_init=0.0,
        beta_init=0.0,
        alpha_rot_init=0.0,
        beta_rot_init=0.0,
        rope_base_freq=10000.0,
    )
    cfg = _tiny_model_config(layer_rope)
    model = OLMo(cfg)
    if dtype != torch.float32:
        model = model.to(dtype)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 4))
    out = model(input_ids)
    logits = out.logits
    assert logits.shape == (2, 4, cfg.embedding_size)
    assert torch.isfinite(logits).all()

    # Ensure the LayerRoPE shared params receive gradients.
    loss = logits.float().pow(2).mean()
    loss.backward()
    sp = model.transformer.layer_rope_params  # type: ignore[attr-defined]
    for name, p in sp.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        # gamma vectors definitely receive grad; some scalars may have zero grad
        # under some inputs but they should at least be allocated.


def test_olmo_layer_rope_param_group_assignment():
    """All LayerRoPE shared params must end up in the optimizer's no_decay group
    by default (decay_norm_and_bias=False is OLMo's training default in our scripts)."""
    from olmo.config import OptimizerConfig, TrainConfig
    from olmo.optim import get_param_groups

    cfg_layer_rope = LayerRoPEConfig(enabled=True, norm_after=False)
    model_cfg = _tiny_model_config(cfg_layer_rope)
    train_cfg = TrainConfig(
        save_folder="/tmp/_test_layer_rope_runs",
        model=model_cfg,
        optimizer=OptimizerConfig(decay_norm_and_bias=False, decay_embeddings=False),
    )
    model = OLMo(model_cfg)
    groups = get_param_groups(train_cfg, model)
    # Find the no-decay group.
    no_decay_group = next(g for g in groups if g.get("weight_decay", None) == 0.0)
    no_decay_names = set(no_decay_group["param_names"])
    layer_rope_names = {
        n for n, _ in model.named_parameters() if n.startswith("transformer.layer_rope_params.")
    }
    missing = layer_rope_names - no_decay_names
    assert not missing, f"LayerRoPE params not routed to no_decay: {missing}"


def test_olmo_layer_rope_disabled_default_is_unchanged():
    """A ModelConfig without layer_rope.enabled should produce a vanilla OLMo block."""
    cfg = _tiny_model_config(LayerRoPEConfig(enabled=False))
    model = OLMo(cfg)
    assert not hasattr(model.transformer, "layer_rope_params")
    block = model.transformer.blocks[0]
    assert not block.layer_rope_enabled  # type: ignore[attr-defined]


def test_olmo_layer_rope_rejects_combo_with_swin_norm_after():
    cfg = _tiny_model_config(LayerRoPEConfig(enabled=True))
    cfg.norm_after = True  # OLMo's Swin-style flag
    from olmo.exceptions import OLMoConfigurationError

    with pytest.raises(OLMoConfigurationError):
        OLMo(cfg)
