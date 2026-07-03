"""Identify the most common first-token predictions when the model refuses a tool call.

Reads all pair_add_*.json files under data/<model_name>/ and counts the top predicted
tokens for samples where the tool-call token is NOT rank-0 (i.e., the model refuses).
Results are written to data/<model_name>/refuse_tokens/refuse_token_analysis.json.

This produces the refusal-token statistics reported in Table 8 (Appendix F).

Run from the repo root::

    python -m behavior_analysis.get_refuse_tokens --model_name qwen3-8b
"""
import os
import json
import argparse
from collections import Counter

from model_utils.base_model import BaseModel


def analyze_refuse_tokens(data_dir: str, output_file: str, model_name: str):
    base_model = BaseModel.create(model_name)
    tokenizer = base_model.model.tokenizer

    counter = Counter()
    error_items = 0

    files = [f for f in os.listdir(data_dir) if f.startswith("pair_add_") and f.endswith(".json")]
    print(f"Scanning {len(files)} file(s) in: {data_dir}")

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            logits_info = item.get("logits_info")
            if not logits_info:
                continue
            if logits_info.get("tool_call_token_rank") != 0:
                top_ids = logits_info.get("top_token_ids", [])
                if top_ids:
                    counter[top_ids[0]] += 1
                    error_items += 1

    results = []
    for token_id, count in counter.most_common():
        results.append({
            "token_id": token_id,
            "token_str": tokenizer.decode([token_id]),
            "count": count,
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Refusal events: {error_items}")
    print(f"Results saved to: {output_file}")
    print("\n--- Top 10 refusal tokens ---")
    for i, res in enumerate(results[:10]):
        print(f"#{i+1}: {repr(res['token_str'])} (ID: {res['token_id']}) — Count: {res['count']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze refusal tokens from SABEval logits files.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g. qwen3-8b).")
    args = parser.parse_args()

    data_dir = os.path.join("data", args.model_name)
    output_file = os.path.join("data", args.model_name, "refuse_tokens", "refuse_token_analysis.json")

    analyze_refuse_tokens(data_dir, output_file, args.model_name)
