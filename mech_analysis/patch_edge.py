import torch
from jaxtyping import Float
from typing import Literal
from torch import Tensor
from dataclasses import dataclass, field


@dataclass
class SpanEdge:
    """An attention edge (head at a layer, from a source span to a destination span)."""
    layer: int
    head: int
    src_span_name: str
    dst_span_name: str
    score: float = field(default=0.0, compare=False)

    def get_raw_edge(self, spans):
        return Edge(
            layer=self.layer,
            head=self.head,
            src_span_name=self.src_span_name,
            dst_span_name=self.dst_span_name,
            src_slice=spans[self.src_span_name]["index"],
            dst_slice=spans[self.dst_span_name]["index"],
            score=self.score,
        )


@dataclass
class Edge:
    """A SpanEdge resolved to concrete token index ranges for a given prompt."""
    layer: int
    head: int
    src_span_name: str
    dst_span_name: str
    src_slice: list[tuple[int, int]]
    dst_slice: list[tuple[int, int]]
    score: float = field(default=0.0, compare=False)


def get_attention_hooks(edges: list[Edge], patch_type: Literal["scale", "add"], patch_value: float, reallocate: bool = False):
    """Build TransformerLens hooks that scale/add the attention weights of the given edges (Eq. 12)."""
    # group edges by layer so each layer gets a single aggregated hook
    edges_by_layer = {}
    for edge in edges:
        edges_by_layer.setdefault(edge.layer, []).append(edge)

    hook_list = []
    for layer, layer_edges in edges_by_layer.items():

        def hook_fn(act: Float[Tensor, "batch head dest src"], hook, current_edges=layer_edges):
            if act.shape[-2] <= 1:  # skip single-token (cached) positions
                return act
            act_clone = act.clone()
            for edge in current_edges:
                head = edge.head
                for dst_span in edge.dst_slice:
                    for src_span in edge.src_slice:
                        target_region = act_clone[:, head, dst_span[0]:dst_span[1], src_span[0]:src_span[1]]
                        if patch_type == "scale":
                            target_region *= patch_value
                        elif patch_type == "add":
                            target_region += patch_value
                            target_region.clamp_(min=0.0)
            if reallocate:
                row_sums = act_clone.sum(dim=-1, keepdim=True)
                act_clone = act_clone / (row_sums + 1e-9)
            return act_clone

        hook_list.append((f"blocks.{layer}.attn.hook_pattern", hook_fn))

    return hook_list


def get_top_edges(attr: Float[Tensor, "layer head dst_span src_span"], span_types: list[str], top_n: int = 50):
    """Return the top_n positive span-edges by attribution score."""
    assert attr.shape[2] == len(span_types)
    assert attr.shape[3] == len(span_types)

    flat_attr = attr.flatten()
    top_n_scores, top_n_indices = torch.topk(flat_attr, top_n)

    n_layers, n_heads, n_spans, _ = attr.shape

    top_span_edges = []
    for i in range(top_n):
        score = top_n_scores[i].item()
        if score <= 1e-6:
            print(f"Warning: not enough positive edges found. Stopping with {i} edges.")
            break

        flat_idx = top_n_indices[i].item()
        layer_idx = flat_idx // (n_heads * n_spans * n_spans)
        head_idx = (flat_idx // (n_spans * n_spans)) % n_heads
        dst_idx = (flat_idx // n_spans) % n_spans
        src_idx = flat_idx % n_spans

        top_span_edges.append(
            SpanEdge(layer=layer_idx, head=head_idx, dst_span_name=span_types[dst_idx], src_span_name=span_types[src_idx], score=score)
        )

    return top_span_edges


def get_top_contrast_edges_by_subtraction(
    attr_1: Float[Tensor, "layer head dst_span src_span"],
    attr_2: Float[Tensor, "layer head dst_span src_span"],
    span_types: list[str],
    top_n: int = 50,
    clamp=True,
    norm=True,
    epsilon: float = 1e-9,
):
    """Select top_n contrast edges: important in attr_1 but not in attr_2 (normalized difference)."""
    if clamp:
        attr_1 = attr_1.clamp(min=0.0)
        attr_2 = attr_2.clamp(min=0.0)

    if norm:
        attr_1 = attr_1 / attr_1.max().clamp(min=epsilon)
        attr_2 = attr_2 / attr_2.max().clamp(min=epsilon)

    contrast_attr = attr_1 - attr_2
    return get_top_edges(contrast_attr, span_types, top_n)
