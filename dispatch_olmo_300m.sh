#!/bin/bash
#
# Dispatch OLMo 300M norm-placement ablation on 4 GPUs.
#
# Runs two variants back-to-back:
#   1. pre  – standard RMSNorm pre-normalisation (baseline)
#   2. lns  – LayerNorm Scaling: RMSNorm + 1/sqrt(layer_id+1) depth scaling
#
# Usage:
#   ./dispatch_olmo_300m.sh
#

# Pre-normalisation (standard RMSNorm baseline)
./run_olmo_300m.sh pre 29515

# LayerNorm Scaling
./run_olmo_300m.sh lns 29515

echo ""
echo "All 300M training jobs completed!"
