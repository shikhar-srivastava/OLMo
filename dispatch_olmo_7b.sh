#!/bin/bash
#
# Dispatch OLMo 7B norm-placement ablation on 4 GPUs.
#
# Runs two variants back-to-back:
#   1. pre  – standard RMSNorm pre-normalisation (baseline)
#   2. lns  – LayerNorm Scaling: RMSNorm + 1/sqrt(layer_id+1) depth scaling
#
# Usage:
#   ./dispatch_olmo_7b.sh
#

# Pre-normalisation (standard RMSNorm baseline; keeps OLMo default init_fn=mitchell)
./run_olmo_7b.sh pre 29515

# LayerNorm Scaling (uses dedicated LNS script that overrides init_fn=normal so
# Mitchell's depth-scaled init does not stack with LNS's depth norm scaling)
./run_olmo_7b_lns.sh 29515

echo ""
echo "All 7B training jobs completed!"
