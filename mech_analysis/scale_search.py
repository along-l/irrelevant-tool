"""Grid-search the semantic/structural scaling coefficients (rho_sem, rho_str) on the
validation split (paper section 5.2, Appendix K).

Sweep coefficients over a grid, evaluate on pair (D_call) and ground-truth
(D_nocall) data, then pick the combination that maximises the intervention effect.

Output: results/<model>/scale_search/add_<k>_trunc_<n>_<split>_topn_<n>.json
"""
import os
import json
import random
import argparse

from tqdm import tqdm

from mech_analysis.patch_edge import get_top_contrast_edges_by_subtraction
from mech_analysis.evaluate import get_attribution_results, patch_evaluate
from model_utils.base_model import BaseModel
from utils.data_util import create_data, select_split


def _dedup_by_prompt(data_list):
    seen, out = set(), []
    for item in data_list:
        if item["prompt"] not in seen:
            seen.add(item["prompt"])
            out.append(item)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, nargs="+",
                        default=["qwen3-4b", "qwen3-8b", "qwen3-14b", "toolace-2.5-8b", "watt-tool-8b"])
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--trunc", type=int, default=500)
    parser.add_argument("--add", type=int, nargs="+", default=[1])
    parser.add_argument("--top_n_factor", type=float, nargs="+", default=[0.01, 0.02, 0.03],
                        help="Top-k attention-head fraction to patch (paper uses 0.02).")
    parser.add_argument("--scale_factor_sem", type=float, nargs="+", default=[1.1, 1.2, 1.3, 1.4, 1.5],
                        help="Semantic-pathway scaling coefficients (rho_sem > 1).")
    parser.add_argument("--scale_factor_str", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Structural-pathway scaling coefficients (rho_str < 1).")
    parser.add_argument("--result_dir", type=str, default="results")
    args = parser.parse_args()

    for model_name in args.model_name:
        base_model = BaseModel.create(model_name)
        model = base_model.model

        val_pair_data, val_gt_data = [], []
        for k in args.add:
            val_pair_data += create_data(base_model, "data/SABEval", "data", "pair", add_param_num=k)
            val_gt_data   += create_data(base_model, "data/SABEval", "data", "ground_truth", add_param_num=k)

        val_pair_data = select_split(val_pair_data, args.split)
        val_gt_data = _dedup_by_prompt(select_split(val_gt_data, args.split))

        random.seed(42); random.shuffle(val_pair_data)
        random.seed(42); random.shuffle(val_gt_data)

        n = min(len(val_pair_data), len(val_gt_data))
        val_pair_data = val_pair_data[:n]
        val_gt_data = val_gt_data[:n]

        val_pair_call_rate = sum(1 for item in val_pair_data if item["logits_info"]["tool_call_token_rank"] == 0) / len(val_pair_data)
        val_gt_call_rate = sum(1 for item in val_gt_data if item["logits_info"]["tool_call_token_rank"] == 0) / len(val_gt_data)
        print(f"Using {len(val_pair_data)} pair / {len(val_gt_data)} gt for {model_name} val evaluation.")

        sem_res = get_attribution_results(model_name, args.add, args.trunc, "train", "semantic")
        str_res = get_attribution_results(model_name, args.add, args.trunc, "train", "structural")
        span_types = sem_res["span_types"]

        for top_n_factor in args.top_n_factor:
            top_n = int(model.cfg.n_layers * model.cfg.n_heads * top_n_factor)
            top_sem_edges = get_top_contrast_edges_by_subtraction(sem_res["attr"]["total"], sem_res["cf_attr"]["total"], span_types, top_n=top_n)
            top_str_edges = get_top_contrast_edges_by_subtraction(str_res["attr"]["total"], str_res["cf_attr"]["total"], span_types, top_n=top_n)

            output_dir = os.path.join(args.result_dir, model_name, "scale_search")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"add_{''.join(map(str, args.add))}_trunc_{args.trunc}_{args.split}_topn_{top_n}.json",
            )

            result = {}
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    result = json.load(f)

            print(f"Scale search: model={model_name}, top_n={top_n}")

            for scale_sem in tqdm(args.scale_factor_sem):
                for scale_str in tqdm(args.scale_factor_str, leave=False):
                    key = f"scale_sem_{scale_sem}_str_{scale_str}"
                    if key in result:
                        continue
                    patch_dict = {scale_sem: top_sem_edges, scale_str: top_str_edges}
                    result[key] = {
                        "patch_score_pair_data": patch_evaluate(val_pair_data, base_model, patch_dict=patch_dict),
                        "patch_score_gt_data":   patch_evaluate(val_gt_data,   base_model, patch_dict=patch_dict),
                    }

            result["raw_pair_call_rate"] = val_pair_call_rate
            result["raw_gt_call_rate"] = val_gt_call_rate

            if "0" not in result:
                result["0"] = {
                    "patch_score_pair_data": patch_evaluate(val_pair_data, base_model, patch_dict=None),
                    "patch_score_gt_data":   patch_evaluate(val_gt_data,   base_model, patch_dict=None),
                }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved: {output_path}")
