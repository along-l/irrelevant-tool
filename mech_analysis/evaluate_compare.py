"""Measure the causal impact of semantic vs. structural pathways on model behavior.

Two modes:
  * ``pathway`` (Fig 6): on D_1, split test pair data into invocation (rank==0) and
    no-invocation cases, then progressively patch the top-K CIS edges of each
    pathway in chunks of ``chunk_size`` heads. Reports the delta of ``m_str`` and
    the TIR on both slices.
  * ``degree`` (Fig 18): on D_0..D_4, patch the top-``chunk_size`` CIS edges of
    each pathway and record the delta on the full pair-test-set.

Outputs land in ``results/<model>/evaluate_compare/`` and are consumed by
``fig6_pathway_strength.ipynb``.
"""
import os
import json
import random
import argparse

from mech_analysis.patch_edge import get_top_contrast_edges_by_subtraction
from mech_analysis.evaluate import get_attribution_results, patch_evaluate_diff
from model_utils.base_model import BaseModel
from utils.data_util import create_data, select_split, get_data_list


def _sample_all_pair(base_model, add, trunc):
    pair_data = select_split(create_data(base_model, "data/SABEval", "data", "pair", add_param_num=add), "test")
    random.seed(42); random.shuffle(pair_data)
    return pair_data[:trunc]


def run_pathway(base_model, add, trunc, chunk_size, chunk_n, top_sem_edges, top_str_edges, out_dir):
    """Fig 6 data: chunked patching over invocation / non-invocation slices."""
    sem_data, _ = get_data_list(base_model, [add], "semantic", "test", trunc=trunc)   # no-invocation (rank>0)
    str_data, _ = get_data_list(base_model, [add], "structural", "test", trunc=trunc) # invocation (rank==0)

    output_path = os.path.join(
        out_dir,
        f"pathway_add_{add}_trunc_{trunc}_test_chunk_{chunk_size}_n_{chunk_n}.json",
    )
    result = json.load(open(output_path)) if os.path.exists(output_path) else {}

    for i in range(chunk_n):
        start, end = chunk_size * i, chunk_size * (i + 1)
        key = f"start_{start}_end_{end}"
        result.setdefault(key, {})

        if "sem_patch_nocall_data" not in result[key]:
            result[key]["sem_patch_nocall_data"] = patch_evaluate_diff(
                sem_data, base_model, patch_dict={0.0: top_sem_edges[start:end]}
            )
        if "sem_patch_call_data" not in result[key]:
            result[key]["sem_patch_call_data"] = patch_evaluate_diff(
                str_data, base_model, patch_dict={0.0: top_sem_edges[start:end]}
            )
        if "str_patch_nocall_data" not in result[key]:
            result[key]["str_patch_nocall_data"] = patch_evaluate_diff(
                sem_data, base_model, patch_dict={0.0: top_str_edges[start:end]}
            )
        if "str_patch_call_data" not in result[key]:
            result[key]["str_patch_call_data"] = patch_evaluate_diff(
                str_data, base_model, patch_dict={0.0: top_str_edges[start:end]}
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


def run_degree(base_model, data_id, trunc, chunk_size, top_sem_edges, top_str_edges, out_dir):
    """Fig 18 data: one chunk of top-K edges patched over the full pair-test-set of D_k."""
    output_path = os.path.join(
        out_dir,
        f"degree_data_{data_id}_trunc_{trunc}_test_chunk_{chunk_size}.json",
    )
    result = json.load(open(output_path)) if os.path.exists(output_path) else {}

    all_pair_data = _sample_all_pair(base_model, data_id, trunc)
    key = f"start_0_end_{chunk_size}"
    result.setdefault(key, {})

    if "sem_patch_all_pair_data" not in result[key]:
        result[key]["sem_patch_all_pair_data"] = patch_evaluate_diff(
            all_pair_data, base_model, patch_dict={0.0: top_sem_edges[:chunk_size]}
        )
    if "str_patch_all_pair_data" not in result[key]:
        result[key]["str_patch_all_pair_data"] = patch_evaluate_diff(
            all_pair_data, base_model, patch_dict={0.0: top_str_edges[:chunk_size]}
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, nargs="+",
                        default=["qwen3-4b", "qwen3-8b", "qwen3-14b", "toolace-2.5-8b", "watt-tool-8b"])
    parser.add_argument("--mode", type=str, choices=["pathway", "degree", "both"], default="both",
                        help="pathway = Fig 6 (chunked, nocall/call); degree = Fig 18 (per-D_k, all_pair)")
    parser.add_argument("--trunc", type=int, default=500)
    parser.add_argument("--add", type=int, default=1,
                        help="Which D_k to source the attribution and (for pathway mode) the eval samples from.")
    parser.add_argument("--pathway_chunk_size", type=int, default=10,
                        help="Head count per chunk for pathway mode (Fig 6 uses 10).")
    parser.add_argument("--pathway_chunk_n", type=int, default=10,
                        help="Number of chunks for pathway mode (Fig 6 uses 10 -> top-100 edges).")
    args = parser.parse_args()

    for model_name in args.model_name:
        base_model = BaseModel.create(model_name)
        model = base_model.model

        sem_res = get_attribution_results(model_name, [args.add], args.trunc, "train", "semantic")
        str_res = get_attribution_results(model_name, [args.add], args.trunc, "train", "structural")
        span_types = sem_res["span_types"]

        top_sem_edges = get_top_contrast_edges_by_subtraction(sem_res["attr"]["total"], sem_res["cf_attr"]["total"], span_types, top_n=10000)
        top_str_edges = get_top_contrast_edges_by_subtraction(str_res["attr"]["total"], str_res["cf_attr"]["total"], span_types, top_n=10000)

        out_dir = os.path.join("results", model_name, "evaluate_compare")
        os.makedirs(out_dir, exist_ok=True)

        if args.mode in ("pathway", "both"):
            run_pathway(base_model, args.add, args.trunc, args.pathway_chunk_size, args.pathway_chunk_n,
                        top_sem_edges, top_str_edges, out_dir)

        if args.mode in ("degree", "both"):
            # Fig 18 uses the top-2% attention-head count as its single chunk.
            degree_chunk = int(model.cfg.n_layers * model.cfg.n_heads * 0.02)
            for data_id in [0, 1, 2, 3, 4]:
                run_degree(base_model, data_id, args.trunc, degree_chunk,
                           top_sem_edges, top_str_edges, out_dir)
