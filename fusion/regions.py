# fusion/regions.py
from dataclasses import dataclass
from typing import List, Set
from graph.ir import GraphIR

# TODO: Expand on this
ELEMENTWISE: Set[str] = {
    "add", "subtract", "multiply", "divide", "maximum", "minimum",
    "exp", "log", "tanh", "sigmoid", "rsqrt", "sqrt",
    "where", "clip",
}
REDUCTIONS: Set[str] = {
    "softmax", "sum", "mean", "prod", "max", "min", "logsumexp",
}
BARRIERS: Set[str] = {
    "matmul", "addmm", "einsum",
    "conv1d", "conv2d", "conv3d",
    "sort", "argsort", "topk", "partition", "argpartition",
    "load", "save", "savez", "save_gguf", "save_safetensors",
}

@dataclass
class FusionRegion:
    op_indices: List[int]
    kind: str  # "barrier" | "generic" | "template"
    score: float = 0.0
    template: str | None = None

def is_fusible(op_name: str) -> bool:
    return (op_name in ELEMENTWISE) or (op_name in REDUCTIONS)

def is_barrier(op_name: str) -> bool:
    return op_name in BARRIERS

def outputs_exclusive(g: GraphIR, op_idx: int) -> bool:
    # For safe fusion: every output has <= 1 consumer
    op = g.ops[op_idx]
    for tid in op.outputs:
        if len(g.tensors[tid].consumers) > 1:
            return False
    return True

def can_chain_fuse(g: GraphIR, prev_idx: int, curr_idx: int) -> bool:
    prev = g.ops[prev_idx]
    curr = g.ops[curr_idx]
    if not is_fusible(prev.op) or not is_fusible(curr.op):
        return False
    if is_barrier(prev.op) or is_barrier(curr.op):
        return False

    # must be data-dependent (chain)
    if len(set(prev.outputs).intersection(curr.inputs)) == 0:
        return False

    # prevent escaping intermediates
    if not outputs_exclusive(g, prev_idx):
        return False

    return True

def find_regions(g: GraphIR) -> List[FusionRegion]:
    regions: List[FusionRegion] = []
    current: List[int] = []

    for i, op in enumerate(g.ops):
        if is_barrier(op.op) or not is_fusible(op.op):
            # flush current
            if current:
                regions.append(FusionRegion(op_indices=current, kind="generic"))
                current = []
            regions.append(FusionRegion(op_indices=[i], kind="barrier"))
            continue

        if not current:
            current = [i]
            continue

        if can_chain_fuse(g, current[-1], i):
            current.append(i)
        else:
            regions.append(FusionRegion(op_indices=current, kind="generic"))
            current = [i]

    if current:
        regions.append(FusionRegion(op_indices=current, kind="generic"))

    return regions
