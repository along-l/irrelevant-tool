import einops
import torch
from tqdm import tqdm

from transformer_lens import ActivationCache
from model_utils.base_model import BaseModel
from mech_analysis.attribute import aggregate_span_attribution
from mech_analysis.patch_edge import SpanEdge


def cache_attention(base_model, tokens):
    """Cache the raw attention weights of every layer for a single prompt."""
    model = base_model.model
    attn_filter = lambda name: "pattern" in name  # [batch, head, dest, src]

    cache = {}

    def fwd_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    with torch.inference_mode():
        with model.hooks(fwd_hooks=[(attn_filter, fwd_cache_hook)]):
            _ = model(tokens)

    cache = ActivationCache(cache, model)
    attn = torch.stack([cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0).to(torch.float32)
    attn = einops.rearrange(attn, "layer batch head dest src -> batch layer head dest src")
    return attn


def cache_attention_patterns(data_list: list[dict], base_model: BaseModel):
    """Aggregate span-level attention weights over a dataset (for the pathway figures)."""
    model = base_model.model
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    span_n = base_model.get_span_num()

    aggregate_keys = ["total", "abs", "pos", "neg"]

    results = {
        "span_types": [],
        "attn": {k: torch.zeros(n_layers, n_heads, span_n, span_n, device=model.cfg.device) for k in aggregate_keys},
    }

    for item in tqdm(data_list, leave=False):
        prompt = item["prompt"]
        tool_schema = item["tool_schema"]
        tool_derived_class = item["meta_data"]["true_tool_derived_class"]

        spans = base_model.split_span(prompt, tool_schema, tool_derived_class)
        attn = cache_attention(base_model, base_model.to_input_ids(prompt))

        span_types = aggregate_span_attribution(attr=attn, spans=spans, span_attr_buffer=results["attn"], mask_noise=False)

        if not results["span_types"]:
            results["span_types"] = span_types

        del attn

    return results


def get_attention_pattern(span_edges: list[SpanEdge], attn_results):
    """Look up the aggregated attention weight of each span-edge."""
    span_types = attn_results["span_types"]
    attn_buffer = attn_results["attn"]

    edge_attn_list = []
    for span_edge in span_edges:
        src_idx = span_types.index(span_edge.src_span_name)
        dst_idx = span_types.index(span_edge.dst_span_name)
        edge_attn_list.append(
            {"span_edge": span_edge, "attn": attn_buffer["total"][span_edge.layer, span_edge.head, dst_idx, src_idx].item()}
        )
    return edge_attn_list
