#!/bin/bash
#
# Run OLMo 300M training on 4 GPUs for a given norm type.
#
# Config selection (in priority order):
#   1. configs/tiny/OLMo-300M-local.yaml   – locally-downloaded shards (no streaming)
#   2. configs/tiny/OLMo-300M-public.yaml  – streams from https://olmo-data.org/
#
# Generate the local config with:
#   python scripts/download_olmo_data.py --data-dir /scratch/ssrivas9/datasets/olmo-data
#   (only needs to run once; shared by 60M / 150M / 300M)
#
#   block_type: sequential | layer_norm_type: rms | distributed: ddp
#   global_train_batch_size: 1024 | max_sequence_length: 4096
#   stop_at: 406,934 steps  =>  ~1.71T tokens  (capped at 20B via --max_duration=2e10T)
#
# Usage:
#   ./run_olmo_300m.sh <norm_type> [master_port]
#
# Arguments:
#   norm_type     Normalisation placement / type:
#                   pre   – standard RMSNorm pre-normalisation (baseline)
#                   lns   – LayerNorm Scaling: RMSNorm + 1/sqrt(layer_id+1) depth scaling
#                   rms   – passed through directly to model.layer_norm_type
#   master_port   (optional) torchrun master port; defaults to 29500
#
# Examples:
#   ./run_olmo_300m.sh pre
#   ./run_olmo_300m.sh lns 29515
#

set -e

norm_type=$1
if [ -z "$norm_type" ]; then
    echo "ERROR: norm_type argument is required."
    echo "Usage: ./run_olmo_300m.sh <norm_type> [master_port]"
    exit 1
fi

# Map human-readable norm_type to OLMo model.layer_norm_type values.
# "pre"  → standard RMSNorm pre-norm (the baseline without depth scaling)
# "lns"  → LayerNorm Scaling (RMSNorm + 1/sqrt(layer_id+1) depth scaling)
# anything else is forwarded verbatim
case "$norm_type" in
    pre)  olmo_norm_type="rms"  ;;
    lns)  olmo_norm_type="lns"  ;;
    *)    olmo_norm_type="$norm_type" ;;
esac

if [ -n "$2" ]; then
    export MASTER_PORT=$2
else
    export MASTER_PORT=29500
fi

export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

# W&B: TrainConfig defaults wandb.entity to "ai2-llm" (Ai2). Override; see run_olmo_60m.sh header.
if [ -n "${WANDB_ENTITY:-}" ]; then
    WANDB_ENTITY_ARG=(--wandb.entity="${WANDB_ENTITY}")
else
    WANDB_ENTITY_ARG=(--wandb.entity=null)
fi

RUN_NAME="OLMo-300M-${norm_type}"

LOCAL_CONFIG="configs/tiny/OLMo-300M-local.yaml"
PUBLIC_CONFIG="configs/tiny/OLMo-300M-public.yaml"
if [ -f "$LOCAL_CONFIG" ]; then
    TRAIN_CONFIG="$LOCAL_CONFIG"
    CONFIG_MODE="local (no streaming)"
else
    TRAIN_CONFIG="$PUBLIC_CONFIG"
    CONFIG_MODE="public (streaming from olmo-data.org)"
fi

echo "=========================================="
echo "OLMo 300M training on 4 GPUs"
echo "=========================================="
echo "Norm type  : ${norm_type}  (model.layer_norm_type=${olmo_norm_type})"
echo "Run name   : ${RUN_NAME}"
echo "Master port: ${MASTER_PORT}"
echo "Config     : ${TRAIN_CONFIG}  [${CONFIG_MODE}]"
echo "Token budget: 20B (capped via --max_duration=2e10T)"
echo "Started at : $(date)"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node 4 \
    --master_port "$MASTER_PORT" \
    scripts/train.py \
        "$TRAIN_CONFIG" \
        --run_name="${RUN_NAME}" \
        --model.layer_norm_type="${olmo_norm_type}" \
        --save_folder="checkpoints/${RUN_NAME}" \
        --remote_save_folder=null \
        --save_overwrite \
        --wandb.name="${RUN_NAME}" \
        --wandb.project="olmo-runs" \
        "${WANDB_ENTITY_ARG[@]}" \
        --max_duration=2e10T \
        --device_train_microbatch_size=8

echo ""
echo "=========================================="
echo "Training complete: ${RUN_NAME}"
echo "Finished at: $(date)"
echo "=========================================="
