#!/bin/bash
# Behavior analysis (§4): generate per-model SABEval next-token logits, then the
# refusal-token statistics (Table 8). Choose a GPU via CUDA_VISIBLE_DEVICES.
#
#   CUDA_VISIBLE_DEVICES=0 bash behavior_analysis/run_behavior.sh
#
# Outputs land in data/<model>/ and data/<model>/refuse_tokens/.
set -e
cd "$(dirname "$0")/.."

MODELS=("qwen3-4b" "qwen3-8b" "qwen3-14b" "toolace-2.5-8b" "watt-tool-8b")

# 1. Next-token logits over SABEval for every (mode, add) the paper needs.
python -m behavior_analysis.generate_data --model_name "${MODELS[@]}"

# 2. Refusal-token analysis (Table 8).
for model in "${MODELS[@]}"; do
    python -m behavior_analysis.get_refuse_tokens --model_name "$model"
done

echo "Behavior data generated. Open notebooks/ to build Table 1, Fig 3, Fig 4, Table 4."
