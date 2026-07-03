"""Sweep a single scaling coefficient (semantic OR structural) on the test split
to produce the TIR-vs-scaling-coefficient heatmaps (Fig. 7, Appendix J)."""
import os
import json
import random
import argparse

from tqdm import tqdm

from mech_analysis.patch_edge import get_top_contrast_edges_by_subtraction
from mech_analysis.evaluate import get_attribution_results, patch_evaluate
from model_utils.base_model import BaseModel
from utils.data_util import create_data, select_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, nargs="+",
                        default=["qwen3-4b", "qwen3-8b", "qwen3-14b", "toolace-2.5-8b", "watt-tool-8b"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--trunc", type=int, default=500)
    parser.add_argument("--add", type=int, nargs="+", default=[1])
    parser.add_argument("--data_num", type=int, default=500, help="Number of samples per condition (paper: 500).")
    parser.add_argument("--top_n_factor", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    parser.add_argument("--scale_factor_sem", type=float, nargs="+", default=[round(0.1 * i, 1) for i in range(21)])
    parser.add_argument("--scale_factor_str", type=float, nargs="+", default=[round(0.1 * i, 1) for i in range(21)])
    parser.add_argument("--result_dir", type=str, default="results")
    args = parser.parse_args()

    for model_name in args.model_name:
        base_model = BaseModel.create(model_name)
        model = base_model.model

        pair_data, gt_data, cf_data = [], [], []
        for k in args.add:
            pair_data += create_data(base_model, "data/SABEval", "data", "pair", add_param_num=k)
            gt_data   += create_data(base_model, "data/SABEval", "data", "ground_truth", add_param_num=k)
            cf_data   += create_data(base_model, "data/SABEval", "data", "counterfactual", add_param_num=k)

        pair_data = select_split(pair_data, args.split)
        cf_data = select_split(cf_data, args.split)

        seen, gt_data_dedup = set(), []
        for item in select_split(gt_data, args.split):
            if item["prompt"] not in seen:
                seen.add(item["prompt"])
                gt_data_dedup.append(item)

        for d in (pair_data, gt_data_dedup, cf_data):
            random.seed(42); random.shuffle(d)

        pair_data = pair_data[:args.data_num]
        gt_data = gt_data_dedup[:args.data_num]
        cf_data = cf_data[:args.data_num]
        print(f"pair: {len(pair_data)}, gt: {len(gt_data)}, cf: {len(cf_data)}")

        sem_res = get_attribution_results(model_name, args.add, args.trunc, "train", "semantic")
        str_res = get_attribution_results(model_name, args.add, args.trunc, "train", "structural")
        span_types = sem_res["span_types"]

        for top_n_factor in args.top_n_factor:
            top_n = int(model.cfg.n_layers * model.cfg.n_heads * top_n_factor)
            print(f"top_n_factor={top_n_factor}, top_n={top_n}")

            top_sem_edges = get_top_contrast_edges_by_subtraction(sem_res["attr"]["total"], sem_res["cf_attr"]["total"], span_types, top_n=top_n)
            top_str_edges = get_top_contrast_edges_by_subtraction(str_res["attr"]["total"], str_res["cf_attr"]["total"], span_types, top_n=top_n)

            output_dir = os.path.join(args.result_dir, model_name, "scale_search_single_test")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"num_{args.data_num}_add_{''.join(map(str, args.add))}_trunc_{args.trunc}_{args.split}_topn_{top_n}.json",
            )

            result = {}
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    result = json.load(f)

            # Semantic-coefficient sweep (pair vs ground-truth data).
            for scale_sem in tqdm(args.scale_factor_sem, desc="sem"):
                key = f"sem_scale_{scale_sem}"
                if key in result:
                    continue
                result[key] = {
                    "pair_data": patch_evaluate(pair_data, base_model, patch_dict={scale_sem: top_sem_edges}),
                    "gt_data":   patch_evaluate(gt_data,   base_model, patch_dict={scale_sem: top_sem_edges}),
                }

            # Structural-coefficient sweep (pair vs counterfactual data).
            for scale_str in tqdm(args.scale_factor_str, desc="str"):
                key = f"str_scale_{scale_str}"
                if key in result:
                    continue
                result[key] = {
                    "pair_data": patch_evaluate(pair_data, base_model, patch_dict={scale_str: top_str_edges}),
                    "cf_data":   patch_evaluate(cf_data,   base_model, patch_dict={scale_str: top_str_edges}),
                }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved: {output_path}")
