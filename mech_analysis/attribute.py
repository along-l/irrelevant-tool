"""Contrastive Attention Attribution (CAA) over SABEval spans (paper section 5.1).

For a (semantic or structural) circuit, computes per-span-edge attribution using
attribution patching (Eq. 6), aggregates token-level scores into span-level scores
(Eq. 7), and stores the result for both the original and counterfactual datasets.

Output: results/<model>/attr_analysis/add_<k>_trunc_<n>_<split>_<circuit>.pt
"""
import os
import argparse
from typing import Callable, Dict, List, Literal, Tuple

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import ActivationCache

from model_utils.base_model import BaseModel
from utils.data_util import get_data_list


def get_metric_fn(circuit_type: Literal["semantic", "structural"], base_model: BaseModel) -> Callable:
    """Task metric m(x) = log P(w_tool) - max_{w in R} log P(w) (Eq. 4/5, Appendix F).

    Returned as raw logit difference; a Softmax is not needed since it is monotonic
    and only relative ranking is used downstream. For the semantic circuit we negate
    so higher attribution corresponds to `no-invocation`.
    """
    if not getattr(base_model, "refuse_tokens", None):
        raise ValueError("base_model must expose a non-empty 'refuse_tokens' dict")
    refuse_ids_cpu = torch.tensor(list(base_model.refuse_tokens.values()), dtype=torch.long)
    cached_refuse_ids = None

    def metric_fn(logits: Float[Tensor, "batch seq vocab"], token_id: Int[Tensor, " batch"]) -> Float[Tensor, " batch"]:
        nonlocal cached_refuse_ids
        final_logits = logits[:, -1, :]
        target_logits = final_logits.gather(1, token_id.unsqueeze(1)).squeeze(1)
        if cached_refuse_ids is None or cached_refuse_ids.device != final_logits.device:
            cached_refuse_ids = refuse_ids_cpu.to(final_logits.device)
        max_refuse_logits = final_logits[:, cached_refuse_ids].max(dim=-1).values
        metric = target_logits - max_refuse_logits
        return -metric if circuit_type == "semantic" else metric

    return metric_fn


def attention_attribution(base_model, tokens, target_token_id, metric_fn):
    """Attribution patching (Nanda 2023): first-order Taylor approximation of the
    indirect effect of every attention weight on the task metric (Eq. 6)."""
    model = base_model.model
    attn_filter = lambda name: "pattern" in name  # [batch, head, dest, src]

    cache, grad_cache = {}, {}

    def fwd_hook(act, hook): cache[hook.name] = act.detach()
    def bwd_hook(act, hook): grad_cache[hook.name] = act.detach()

    with model.hooks(fwd_hooks=[(attn_filter, fwd_hook)], bwd_hooks=[(attn_filter, bwd_hook)]):
        logits = model(tokens)
        metric = metric_fn(logits, target_token_id)
        metric.backward()

    cache = ActivationCache(cache, model)
    grad_cache = ActivationCache(grad_cache, model)

    attn = torch.stack([cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0).to(torch.float32)
    grad = torch.stack([grad_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0).to(torch.float32)

    attr = grad * attn  # [layer, batch, head, dest, src] — positive value = positive contribution
    return einops.rearrange(attr, "layer batch head dest src -> batch layer head dest src")


def _get_span_mapping_matrix(span_ranges: List[List[Tuple[int, int]]], seq_len: int, device: torch.device):
    """Mask M[i, j] = 1 iff token j belongs to span i."""
    mask = torch.zeros((len(span_ranges), seq_len), dtype=torch.float32, device=device)
    for i, ranges in enumerate(span_ranges):
        for s, e in ranges:
            s, e = max(0, s), min(seq_len, e)
            if e > s:
                mask[i, s:e] = 1.0
    return mask


def aggregate_span_attribution(
    attr: Float[Tensor, "batch layer head dest src"],
    spans: Dict,
    span_attr_buffer: Dict[str, torch.Tensor],
) -> List[str]:
    """Aggregate token-level attribution into span-level scores (Eq. 7). Accumulates in-place."""
    assert attr.shape[0] == 1, "batch size 1 only"
    attr = attr.squeeze(0)  # [layer, head, dest, src]

    _, _, dest_len, src_len = attr.shape
    device = attr.device

    span_types = list(spans.keys())
    span_ranges = [r["index"] for r in spans.values()]

    M_dest = _get_span_mapping_matrix(span_ranges, dest_len, device)
    M_src = M_dest if dest_len == src_len else _get_span_mapping_matrix(span_ranges, src_len, device)

    pos_mask = attr > 0
    neg_mask = attr < 0
    eq = "nd, lhds, ms -> lhnm"  # M_dest, attr, M_src -> aggregated

    span_attr_buffer["total"] += torch.einsum(eq, M_dest, attr, M_src)
    span_attr_buffer["abs"]   += torch.einsum(eq, M_dest, attr.abs(), M_src)
    span_attr_buffer["pos"]   += torch.einsum(eq, M_dest, attr * pos_mask, M_src)
    span_attr_buffer["neg"]   += torch.einsum(eq, M_dest, attr * neg_mask, M_src)

    return span_types


def span_attribute(data_list: List[Dict], base_model: BaseModel, metric_fn: Callable):
    model = base_model.model
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    span_n = base_model.get_span_num()

    aggregate_keys = ["total", "abs", "pos", "neg"]
    results = {
        "span_types": [],
        "attr": {k: torch.zeros(n_layers, n_heads, span_n, span_n, device=model.cfg.device) for k in aggregate_keys},
    }

    for item in tqdm(data_list):
        input_ids = base_model.to_input_ids(item["prompt"])
        target_token_id = torch.tensor(item["tool_call_token_id"]).unsqueeze(0).to(model.cfg.device)

        attr = attention_attribution(base_model, input_ids, target_token_id, metric_fn)  # [1, layer, head, dest, src]
        spans = base_model.split_span(item["prompt"], item["tool_schema"], item["meta_data"]["true_tool_derived_class"])
        span_types = aggregate_span_attribution(attr, spans, results["attr"])

        if not results["span_types"]:
            results["span_types"] = span_types
        del attr

    num_samples = len(data_list)
    if num_samples > 0:
        for k in aggregate_keys:
            results["attr"][k] /= num_samples
    return results


def main(args):
    print("-" * 60)
    print(f"CAA | model={args.model_name} | circuit={args.circuit_type} | add={args.add} | split={args.split}")
    print("-" * 60)

    out_dir = os.path.join(args.output_dir, args.model_name, "attr_analysis")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(
        out_dir,
        f"add_{''.join(map(str, args.add))}_trunc_{args.trunc}_{args.split}_{args.circuit_type}.pt",
    )
    if os.path.exists(output_path):
        print(f"Already exists, skipping: {output_path}")
        return

    base_model = BaseModel.create(args.model_name)
    data_list, cf_data_list = get_data_list(base_model, args.add, args.circuit_type, args.split, args.trunc)

    metric_fn = get_metric_fn(args.circuit_type, base_model)

    print("Example prompt:", data_list[0]["prompt"][:200], "...")
    print("Counterfactual example prompt:", cf_data_list[0]["prompt"][:200], "...")

    orig = span_attribute(data_list, base_model, metric_fn)
    cf   = span_attribute(cf_data_list, base_model, metric_fn)
    assert orig["span_types"] == cf["span_types"]

    results = {"span_types": orig["span_types"], "attr": orig["attr"], "cf_attr": cf["attr"]}
    torch.save(results, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive Attention Attribution over SABEval spans (Section 5.1).")
    parser.add_argument("--model_name", type=str, default="qwen3-8b")
    parser.add_argument("--add", type=int, nargs="+", default=[1], help="add_param_num values (dataset subsets D_k).")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--trunc", type=int, default=500)
    parser.add_argument("--circuit_type", type=str, default="structural", choices=["semantic", "structural"])
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
