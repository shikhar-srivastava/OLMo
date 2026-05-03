#!/bin/bash
#
# Dispatch OLMo 60M norm-placement ablation on 4 GPUs.
#
# Runs two variants back-to-back:
#   1. pre  – standard RMSNorm pre-normalisation (baseline)
#   2. lns  – LayerNorm Scaling: RMSNorm + 1/sqrt(layer_id+1) depth scaling
#
# Usage:
#   ./dispatch_olmo_60m.sh [--first4gpus | --last4gpus]
#
# GPU selection (optional):
#   --first4gpus   Use GPUs 0,1,2,3 (default if neither flag is given)
#   --last4gpus    Use GPUs 4,5,6,7
#

if [[ "$1" == "--first4gpus" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    shift
elif [[ "$1" == "--last4gpus" ]]; then
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    shift
fi

# Pre-normalisation (standard RMSNorm baseline)
./run_olmo_60m.sh pre 29515

# LayerNorm Scaling
./run_olmo_60m.sh lns 29515

echo ""
echo "All 60M training jobs completed!"
