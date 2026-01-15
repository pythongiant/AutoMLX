# fusion/regions.py

from dataclasses import dataclass
from typing import List, Set

from graph.ir import GraphIR
from graph.op_registry import get_op_kind, is_barrier, OpKind
from graph.graph_utils import (
    build_op_consumers,
    single_consumer_only,
    toposort_ops,
)


@dataclass
class FusionRegion:
    """
    Concrete fusion region.
    Public API: op_indices
    Internal storage: _ops
    """
    _ops: List[int]

    @property
    def op_indices(self) -> List[int]:
        # Backward-compatible public interface
        return self._ops

    @property
    def ops(self) -> List[int]:
        # Internal alias (optional)
        return self._ops


def can_fuse_pair(g: GraphIR, prev_idx: int, curr_idx: int) -> bool:
    prev = g.ops[prev_idx]
    curr = g.ops[curr_idx]

    # ---- barrier checks ----
    if is_barrier(prev.op) or is_barrier(curr.op):
        return False

    # ---- reduction boundary ----
    if get_op_kind(prev.op) == OpKind.REDUCTION:
        return False

    if get_op_kind(curr.op) == OpKind.REDUCTION:
        return False

    # ---- data dependency ----
    if not set(prev.outputs).intersection(curr.inputs):
        return False

    # ---- single-consumer invariant ----
    if not single_consumer_only(g, prev_idx):
        return False

    return True



def find_regions(g: GraphIR) -> List[FusionRegion]:
    """
    Phase-1 fusion region discovery (legality only).

    Design:
    - Each op starts as its own region
    - No greedy expansion
    - No BFS / DFS
    - No cost model
    - Deterministic

    Rationale:
    - Cost-aware fusion must *grow* regions, not split them
    - This mirrors XLA / TVM / TorchInductor architecture
    """
    regions: List[FusionRegion] = []

    for idx, op in enumerate(g.ops):
        regions.append(FusionRegion([idx]))

    return regions


def is_fusible(op_name: str) -> bool:
    kind = get_op_kind(op_name)
    return kind in (OpKind.ELEMENTWISE, OpKind.REDUCTION)
