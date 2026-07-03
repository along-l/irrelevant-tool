#!/bin/bash
# Mechanistic analysis (section 5): Contrastive Attention Attribution (CAA).
# Computes semantic and structural attribution on the D_1 training split for every
# model, writing results/<model>/attr_analysis/*.pt that the downstream scripts
# (evaluate.py, scale_search*.py, evaluate_compare.py) and notebooks consume.
#
#   CUDA_VISIBLE_DEVICES=0 bash mech_analysis/run_attribution.sh
#
# Requires the behavior data (run_behavior.sh) to exist first.
set -e
cd "$(dirname "$0")/.."

MODELS=("qwen3-4b" "qwen3-8b" "qwen3-14b" "toolace-2.5-8b" "watt-tool-8b")
CIRCUITS=("semantic" "structural")

for model in "${MODELS[@]}"; do
    for circuit in "${CIRCUITS[@]}"; do
        echo "=== CAA | model=$model | circuit=$circuit ==="
        python -m mech_analysis.attribute \
            --model_name "$model" \
            --add 1 \
            --split train \
            --circuit_type "$circuit" \
            --trunc 500
    done
done

echo "Attribution done. Next:"
echo "  python -m mech_analysis.scale_search        # pick rho on val (Appendix K)"
echo "  python -m mech_analysis.evaluate            # Table 3 rows on test"
echo "  python -m mech_analysis.scale_search_single # Fig 7 heatmaps"
echo "  python -m mech_analysis.evaluate_compare    # Fig 6 pathway comparison"
