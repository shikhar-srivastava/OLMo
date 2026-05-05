#!/bin/bash
#
# Run OLMo 1B training on 4 GPUs with LayerNorm Scaling (LNS) and *non-scaled*
# weight init (init_fn=normal).
#
# Why a separate script: the OLMo 1B base configs ship with init_fn=mitchell,
# which applies depth-dependent attn_out / ff_out std scaling
# (1/sqrt(2 * d_model * (layer_id+1))). LNS itself already imposes a depth
# 1/sqrt(layer_id+1) factor at the norm; stacking it on top of Mitchell's
# depth-scaled init double-counts the depth correction. This script removes
# the layer-dependent init by overriding init_fn back to "normal" — i.e.,
# removing layer init to normal. init_cutoff_factor=3 is set explicitly to
# preserve the ±3σ truncation that the default mitchell path applies via its
# `init_cutoff_factor or 3.0` fallback (the "normal" branch has no such
# fallback, so leaving it unset would silently switch to untruncated tails).
# init_std is left at its OLMo default of 0.02.
#
# Config selection (in priority order):
#   1. configs/official-0724/OLMo-1B-local.yaml  – locally-downloaded shards
#   2. configs/official-0724/OLMo-1B.yaml        – streams from olmo-data.org
#
# Usage:
#   ./run_olmo_1b_lns.sh [master_port]
#
# Arguments:
#   master_port   (optional) torchrun master port; defaults to 29500
#

set -e

if [ -n "$1" ]; then
    export MASTER_PORT=$1
else
    export MASTER_PORT=29500
fi

export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

if [ -n "${WANDB_ENTITY:-}" ]; then
    WANDB_ENTITY_ARG=(--wandb.entity="${WANDB_ENTITY}")
else
    WANDB_ENTITY_ARG=(--wandb.entity=null)
fi

RUN_NAME="OLMo-1B-lns_no_scaleinit"

LOCAL_CONFIG="configs/official-0724/OLMo-1B-local.yaml"
PUBLIC_CONFIG="configs/official-0724/OLMo-1B.yaml"
if [ -f "$LOCAL_CONFIG" ]; then
    TRAIN_CONFIG="$LOCAL_CONFIG"
    CONFIG_MODE="local (no streaming)"
else
    TRAIN_CONFIG="$PUBLIC_CONFIG"
    CONFIG_MODE="public (streaming from olmo-data.org)"
fi

echo "=========================================="
echo "OLMo 1B LNS training on 4 GPUs (init_fn=normal override)"
echo "=========================================="
echo "Norm type   : lns  (model.layer_norm_type=lns)"
echo "Init        : normal, cutoff=3  (override; removing layer init to normal)"
echo "Run name    : ${RUN_NAME}"
echo "Master port : ${MASTER_PORT}"
echo "GPUs        : ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Config      : ${TRAIN_CONFIG}  [${CONFIG_MODE}]"
echo "Token budget: 20B (capped via --max_duration='2e10T')"
echo "Started at  : $(date)"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" torchrun \
    --nproc_per_node 4 \
    --master_port "$MASTER_PORT" \
    scripts/train.py \
        "$TRAIN_CONFIG" \
        --run_name="${RUN_NAME}" \
        --model.layer_norm_type=lns \
        --model.init_fn=normal \
        --model.init_cutoff_factor=3 \
        --save_folder="checkpoints/${RUN_NAME}" \
        --save_overwrite \
        --wandb.name="${RUN_NAME}" \
        --wandb.project="olmo-runs" \
        "${WANDB_ENTITY_ARG[@]}" \
        --max_duration='2e10T' \
        --device_train_microbatch_size=8

echo ""
echo "=========================================="
echo "Training complete: ${RUN_NAME}"
echo "Finished at: $(date)"
echo "=========================================="
