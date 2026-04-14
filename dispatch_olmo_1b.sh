#!/bin/bash
#
# Dispatch OLMo 1B (~1B params) norm-placement ablation on 4 GPUs.
#
# Runs two variants back-to-back:
#   1. pre  – standard RMSNorm pre-normalisation (baseline)
#   2. lns  – LayerNorm Scaling: RMSNorm + 1/sqrt(layer_id+1) depth scaling
#
# Usage:
#   ./dispatch_olmo.sh
#

# Pre-normalisation (standard RMSNorm baseline)
./run_olmo_1b.sh pre 29515

# LayerNorm Scaling
./run_olmo_1b.sh lns 29515

echo ""
echo "All training jobs completed!"
