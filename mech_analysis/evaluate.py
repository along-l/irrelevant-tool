import os
import random
import argparse
import json

import torch
import numpy as np
from tqdm import tqdm

from model_utils.base_model import BaseModel
from mech_analysis.patch_edge import get_attention_hooks, get_top_contrast_edges_by_subtraction
from utils.data_util import create_data, select_split


def get_attribution_results(model_name, add, trunc, split, circuit_type):
    file_path = os.path.join(
        "results",
        model_name,
        "attr_analysis",
        f"add_{''.join(map(str, add))}_trunc_{trunc}_{split}_{circuit_type}.pt",
    )
    print(f"Loading attribution results from {file_path}...")
    return torch.load(file_path)


def _is_tool_call(base_model, logits, item):
    """Whether the top-1 next-token prediction constitutes a tool invocation for this model."""
    token_id = item["tool_call_token_id"]
    if base_model.model_name not in ["toolace-2.5-8b", "watt-tool-8b"]:
        # For Qwen3, tool-call token is fixed; rank-0 == invocation.
        target_logit = logits[token_id].item()
        greater = (logits > target_logit).sum().item()
        ties = torch.nonzero(logits == target_logit).squeeze(-1)
        rank = greater + (ties < token_id).sum().item()
        return int(rank == 0)
    # For Llama-based tool models, "[" is shared across tools — check the actual decoded prefix.
    top_id = torch.topk(logits, k=5).indices[0]
    out_str = base_model.tokenizer.decode(top_id, skip_special_tokens=True)
    if "]" not in out_str and out_str.startswith("["):
        if len(out_str) == 1 or out_str[1] == item["tool_name"][0]:
            return 1
    return 0


def _logit_diff_fix(logits, item, refuse_token_ids):
    """m_str = log P(w_tool) - max_{w in R} log P(w) at the last position (Eq. 5, App. F)."""
    target_logit = logits[item["tool_call_token_id"]].item()
    max_refuse_logit = logits[refuse_token_ids].max().item()
    return target_logit - max_refuse_logit


def _build_hooks(base_model, item, patch_dict):
    """Convert a {scale: [SpanEdge, ...]} dict into TransformerLens forward hooks."""
    if patch_dict is None:
        return []
    spans = base_model.split_span(item["prompt"], item["tool_schema"], item["meta_data"]["true_tool_derived_class"])
    hook_list = []
    for patch_value, span_edges in patch_dict.items():
        edges = [span_edge.get_raw_edge(spans) for span_edge in span_edges]
        hook_list += get_attention_hooks(edges, "scale", patch_value, reallocate=False)
    return hook_list


def _forward_last_logits(base_model, prompt, hook_list):
    with torch.inference_mode():
        input_ids = base_model.to_input_ids(prompt)
        if hook_list:
            with base_model.model.hooks(fwd_hooks=hook_list):
                logits = base_model.model(input_ids, return_type="logits").detach()
        else:
            logits = base_model.model(input_ids, return_type="logits").detach()
    return logits[0, -1, :]


def patch_evaluate(data_list: list[dict], base_model: BaseModel, patch_dict=None):
    """Run forward passes (optionally with attention-scaling hooks) and return TIR."""
    tool_call_list = []
    for item in tqdm(data_list, leave=False):
        hooks = _build_hooks(base_model, item, patch_dict)
        logits = _forward_last_logits(base_model, item["prompt"], hooks)
        tool_call_list.append(_is_tool_call(base_model, logits, item))

    return {
        "tool_calling_rate": float(np.mean(tool_call_list)),
        "number_of_examples": len(tool_call_list),
    }


def patch_evaluate_diff(data_list: list[dict], base_model: BaseModel, patch_dict=None):
    """Like patch_evaluate but returns the *change* in TIR and in m_str (Eq. 5) relative to
    the unpatched baseline. m_str deltas are consumed by Fig 6 / Fig 18."""
    refuse_token_ids = torch.tensor(list(base_model.refuse_tokens.values()), dtype=torch.long, device=base_model.model.cfg.device)

    tool_call_diffs, logit_diff_fix_diffs = [], []
    for item in tqdm(data_list, leave=False):
        hooks = _build_hooks(base_model, item, patch_dict)
        patched_logits = _forward_last_logits(base_model, item["prompt"], hooks)
        raw_logits = _forward_last_logits(base_model, item["prompt"], [])

        tool_call_diffs.append(_is_tool_call(base_model, patched_logits, item) - _is_tool_call(base_model, raw_logits, item))
        logit_diff_fix_diffs.append(_logit_diff_fix(patched_logits, item, refuse_token_ids) - _logit_diff_fix(raw_logits, item, refuse_token_ids))

    return {
        "tool_calling_rate": float(np.mean(tool_call_diffs)),
        "avg_logit_diff_fix": float(np.mean(logit_diff_fix_diffs)),
        "number_of_examples": len(tool_call_diffs),
    }


def _dedup_by_prompt(data_list):
    seen, out = set(), []
    for item in data_list:
        if item["prompt"] not in seen:
            seen.add(item["prompt"])
            out.append(item)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the CAA intervention on the test split (Table 3).")
    parser.add_argument("--model_name", type=str, nargs="+",
                        default=["qwen3-4b", "qwen3-8b", "qwen3-14b", "toolace-2.5-8b", "watt-tool-8b"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--trunc", type=int, default=500)
    parser.add_argument("--add", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--result_dir", type=str, default="results")
    args = parser.parse_args()

    for model_name in args.model_name:
        base_model = BaseModel.create(model_name)

        factor_path = f"results/{model_name}/scale_search/best_factor.json"
        with open(factor_path, "r", encoding="utf-8") as f:
            factors = json.load(f)
        top_n, sem_factor, str_factor = factors["top_n"], factors["sem_factor"], factors["str_factor"]

        # Attribution is computed once on D_1 train (paper section 5.2).
        sem_res = get_attribution_results(model_name, [1], args.trunc, "train", "semantic")
        str_res = get_attribution_results(model_name, [1], args.trunc, "train", "structural")
        span_types = sem_res["span_types"]
        top_sem_edges = get_top_contrast_edges_by_subtraction(sem_res["attr"]["total"], sem_res["cf_attr"]["total"], span_types, top_n=top_n)
        top_str_edges = get_top_contrast_edges_by_subtraction(str_res["attr"]["total"], str_res["cf_attr"]["total"], span_types, top_n=top_n)
        patch_dict = {sem_factor: top_sem_edges, str_factor: top_str_edges}

        for add in args.add:
            output_dir = os.path.join(args.result_dir, model_name, "patch_evaluate")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"add_{add}_trunc_{args.trunc}_{args.split}_semfactor_{sem_factor}_strfactor_{str_factor}_topn_{top_n}.json",
            )
            if os.path.exists(output_path):
                print(f"Exists, skipping: {output_path}")
                continue

            test_pair = select_split(create_data(base_model, "data/SABEval", "data", "pair", add_param_num=add), args.split)
            test_gt   = _dedup_by_prompt(select_split(create_data(base_model, "data/SABEval", "data", "ground_truth", add_param_num=add), args.split))

            random.seed(42); random.shuffle(test_pair)
            random.seed(42); random.shuffle(test_gt)

            print(f"pair: {len(test_pair)}, gt: {len(test_gt)}")

            result = {
                "pair_result": patch_evaluate(test_pair, base_model, patch_dict=patch_dict),
                "gt_result":   patch_evaluate(test_gt,   base_model, patch_dict=patch_dict),
                "raw_pair_result": patch_evaluate(test_pair, base_model, patch_dict=None),
                "raw_gt_result":   patch_evaluate(test_gt,   base_model, patch_dict=None),
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
