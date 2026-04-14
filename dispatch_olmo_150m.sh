#!/bin/bash
#
# Dispatch OLMo 150M norm-placement ablation on 4 GPUs.
#
# Runs two variants back-to-back:
#   1. pre  – standard RMSNorm pre-normalisation (baseline)
#   2. lns  – LayerNorm Scaling: RMSNorm + 1/sqrt(layer_id+1) depth scaling
#
# Usage:
#   ./dispatch_olmo_150m.sh
#

# Pre-normalisation (standard RMSNorm baseline)
./run_olmo_150m.sh pre 29515

# LayerNorm Scaling
./run_olmo_150m.sh lns 29515

echo ""
echo "All 150M training jobs completed!"
