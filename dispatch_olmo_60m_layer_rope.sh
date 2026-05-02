#!/bin/bash
#
# Dispatch OLMo 60M LayerRoPE ablation on 4 GPUs.
#
# Runs both LayerRoPE variants back-to-back at the user's reference settings:
#   alpha_init=0.0  beta_init=0.0  alpha_rot_init=0.0  beta_rot_init=0.0
#   rope_base_freq=10000
#
# Variant 1: pre        – ComplexRotRMSNorm pre-norms + explicit residual gate
# Variant 2: norm_after – ComplexRotRMSNorm pre-norms + post-block ComplexRotRMSNorm
#                         (residual gate baked into the post-block norm)
#
# Override defaults by passing positional args:
#   ./dispatch_olmo_60m_layer_rope.sh [alpha_init] [beta_init] [alpha_rot_init] [beta_rot_init] [rope_base_freq]

set -e

alpha_init=${1:-0.0}
beta_init=${2:-0.0}
alpha_rot_init=${3:-0.0}
beta_rot_init=${4:-0.0}
rope_base_freq=${5:-10000.0}

./run_olmo_60m_layer_rope.sh pre        29515 "$alpha_init" "$beta_init" "$alpha_rot_init" "$beta_rot_init" "$rope_base_freq"
./run_olmo_60m_layer_rope.sh norm_after 29515 "$alpha_init" "$beta_init" "$alpha_rot_init" "$beta_rot_init" "$rope_base_freq"

echo ""
echo "All 60M LayerRoPE training jobs completed!"
