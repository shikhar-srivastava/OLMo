#!/bin/bash
#
# Run OLMo 7B training on 8× A100 80GB with the LayerRoPE feature enabled.
# See ./run_olmo_60m_layer_rope.sh for the variant semantics and full doc.

set -e

variant=$1
if [ -z "$variant" ]; then
    echo "ERROR: variant argument is required (pre | norm_after)."
    exit 1
fi

case "$variant" in
    pre)        layer_rope_norm_after=false ;;
    norm_after) layer_rope_norm_after=true  ;;
    *) echo "ERROR: variant must be 'pre' or 'norm_after'"; exit 1 ;;
esac

if [ -n "$2" ]; then export MASTER_PORT=$2; else export MASTER_PORT=29500; fi
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

RUN_NAME="OLMo-7B-layer-rope-${variant}-a${alpha_init}-b${beta_init}-ar${alpha_rot_init}-br${beta_rot_init}-bf${rope_base_freq}_no_scaleinit"

LOCAL_CONFIG="configs/official-0724/OLMo-7B-local.yaml"
PUBLIC_CONFIG="configs/official-0724/OLMo-7B.yaml"
if [ -f "$LOCAL_CONFIG" ]; then
    TRAIN_CONFIG="$LOCAL_CONFIG"
    CONFIG_MODE="local (no streaming)"
else
    TRAIN_CONFIG="$PUBLIC_CONFIG"
    CONFIG_MODE="public (streaming from olmo-data.org)"
fi

echo "=========================================="
echo "OLMo 7B LayerRoPE training on 8x A100 80GB"
echo "=========================================="
echo "Variant         : ${variant}  (layer_rope.norm_after=${layer_rope_norm_after})"
echo "Init            : normal, cutoff=3  (override; removing layer init to normal — default 7B config uses mitchell)"
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

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node 8 \
    --master_port "$MASTER_PORT" \
    scripts/train.py \
        "$TRAIN_CONFIG" \
        --run_name="${RUN_NAME}" \
        --save_folder="checkpoints/${RUN_NAME}" \
        --save_overwrite \
        --wandb.name="${RUN_NAME}" \
        --wandb.project="olmo-runs" \
        "${WANDB_ENTITY_ARG[@]}" \
        --max_duration='2e10T' \
        --save_interval_unsharded=10000 \
        --device_train_microbatch_size=4 \
        --model.init_fn=normal \
        --model.init_cutoff_factor=3 \
        --model.layer_rope.enabled=true \
        --model.layer_rope.norm_after="${layer_rope_norm_after}" \
        --model.layer_rope.alpha_init="${alpha_init}" \
        --model.layer_rope.beta_init="${beta_init}" \
        --model.layer_rope.alpha_rot_init="${alpha_rot_init}" \
        --model.layer_rope.beta_rot_init="${beta_rot_init}" \
        --model.layer_rope.rope_base_freq="${rope_base_freq}"

echo ""
echo "=========================================="
echo "Training complete: ${RUN_NAME}"
echo "Finished at: $(date)"
echo "=========================================="
