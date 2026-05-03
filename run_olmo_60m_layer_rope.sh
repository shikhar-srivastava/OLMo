#!/bin/bash
#
# Run OLMo 60M training on 4 GPUs with the LayerRoPE feature enabled.
# Faithful port of the upstream "complex-rotation global depth gates + shared
# gamma + RoPE-global" path from /scratch/ssrivas9/large-activations.
#
# Two variants are supported:
#   pre         – pre-norms become ComplexRotRMSNorm (input rot params),
#                 plus an explicit residual gate using the residual rot
#                 params and gamma_residual_attn / gamma_residual_mlp.
#                 Reproduces:
#                   run_60m_4gpus_llama_global_depth_gates_complex_rot_shared_gamma.sh \
#                       --rope_global --input_gamma_rot --residual_gamma_rot
#   norm_after  – pre-norms become ComplexRotRMSNorm (input rot params),
#                 PLUS post-block ComplexRotRMSNorms (residual rot params)
#                 with their own gamma_post_block_attn / gamma_post_block_mlp.
#                 Explicit residual gate is skipped (baked into post-block norm).
#                 Reproduces:
#                   run_60m_4gpus_llama_global_depth_gates_complex_rot_rope_global_shared_gamma_norm_after.sh \
#                       --rope_global --input_gamma_rot --residual_gamma_rot
#
# Config selection:
#   1. configs/tiny/OLMo-60M-local.yaml   – locally-downloaded shards (preferred)
#   2. configs/tiny/OLMo-60M-public.yaml  – streams from olmo-data.org
#
# Token budget capped at 20B via --max_duration='2e10T'.
#
# Usage:
#   ./run_olmo_60m_layer_rope.sh <variant> [master_port] \
#       [alpha_init] [beta_init] [alpha_rot_init] [beta_rot_init] [rope_base_freq]
#
# Defaults reproduce the user's two reference commands:
#   alpha_init=0.0  beta_init=0.0  alpha_rot_init=0.0  beta_rot_init=0.0
#   rope_base_freq=10000
#
# Examples:
#   ./run_olmo_60m_layer_rope.sh pre
#   ./run_olmo_60m_layer_rope.sh norm_after 29515
#   ./run_olmo_60m_layer_rope.sh pre 29515 0.0 -0.5 0.0 -0.5 10000

set -e

variant=$1
if [ -z "$variant" ]; then
    echo "ERROR: variant argument is required (pre | norm_after)."
    echo "Usage: ./run_olmo_60m_layer_rope.sh <variant> [master_port] [alpha_init] [beta_init] [alpha_rot_init] [beta_rot_init] [rope_base_freq]"
    exit 1
fi

case "$variant" in
    pre)
        layer_rope_norm_after=false
        ;;
    norm_after)
        layer_rope_norm_after=true
        ;;
    *)
        echo "ERROR: variant must be 'pre' or 'norm_after', got '$variant'"
        exit 1
        ;;
esac

if [ -n "$2" ]; then
    export MASTER_PORT=$2
else
    export MASTER_PORT=29500
fi

alpha_init=${3:-0.0}
beta_init=${4:-0.0}
alpha_rot_init=${5:-0.0}
beta_rot_init=${6:-0.0}
rope_base_freq=${7:-10000.0}

export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

if [ -n "${WANDB_ENTITY:-}" ]; then
    WANDB_ENTITY_ARG=(--wandb.entity="${WANDB_ENTITY}")
else
    WANDB_ENTITY_ARG=(--wandb.entity=null)
fi

RUN_NAME="OLMo-60M-layer-rope-${variant}-a${alpha_init}-b${beta_init}-ar${alpha_rot_init}-br${beta_rot_init}-bf${rope_base_freq}_nfp32_wd0"

LOCAL_CONFIG="configs/tiny/OLMo-60M-local.yaml"
PUBLIC_CONFIG="configs/tiny/OLMo-60M-public.yaml"
if [ -f "$LOCAL_CONFIG" ]; then
    TRAIN_CONFIG="$LOCAL_CONFIG"
    CONFIG_MODE="local (no streaming)"
else
    TRAIN_CONFIG="$PUBLIC_CONFIG"
    CONFIG_MODE="public (streaming from olmo-data.org)"
fi

echo "=========================================="
echo "OLMo 60M LayerRoPE training on 4 GPUs"
echo "=========================================="
echo "Variant         : ${variant}  (layer_rope.norm_after=${layer_rope_norm_after})"
echo "Run name        : ${RUN_NAME}"
echo "Master port     : ${MASTER_PORT}"
echo "Config          : ${TRAIN_CONFIG}  [${CONFIG_MODE}]"
echo "alpha_init      : ${alpha_init}"
echo "beta_init       : ${beta_init}"
echo "alpha_rot_init  : ${alpha_rot_init}"
echo "beta_rot_init   : ${beta_rot_init}"
echo "rope_base_freq  : ${rope_base_freq}  (LayerRoPE; distinct from attention rope_theta)"
echo "Token budget    : 20B (capped via --max_duration='2e10T')"
echo "Started at      : $(date)"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node 4 \
    --master_port "$MASTER_PORT" \
    scripts/train.py \
        "$TRAIN_CONFIG" \
        --run_name="${RUN_NAME}" \
        --save_folder="checkpoints/${RUN_NAME}" \
        --remote_save_folder=null \
        --save_overwrite \
        --wandb.name="${RUN_NAME}" \
        --wandb.project="olmo-runs" \
        "${WANDB_ENTITY_ARG[@]}" \
        --max_duration='2e10T' \
        --device_train_microbatch_size=8 \
        --model.layer_rope.enabled=true \
        --model.layer_rope.norm_after="${layer_rope_norm_after}" \
        --model.layer_rope.alpha_init="${alpha_init}" \
        --model.layer_rope.beta_init="${beta_init}" \
        --model.layer_rope.alpha_rot_init="${alpha_rot_init}" \
        --model.layer_rope.beta_rot_init="${beta_rot_init}" \
        --model.layer_rope.rope_base_freq="${rope_base_freq}" \
        --model.rmsnorm_in_fp32=true \
        --optimizer.decay_norm_and_bias=false

echo ""
echo "=========================================="
echo "Training complete: ${RUN_NAME}"
echo "Finished at: $(date)"
echo "=========================================="
