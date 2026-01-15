# graph/graph_utils.py

from collections import defaultdict, deque
from typing import Dict, List, Set

from graph.ir import GraphIR, OpIR


def build_producer_map(g: GraphIR) -> Dict[int, int]:
    """
    tid -> producing op index
    """
    prod = {}
    for i, op in enumerate(g.ops):
        for tid in op.outputs:
            prod[tid] = i
    return prod


def build_consumer_map(g: GraphIR) -> Dict[int, List[int]]:
    """
    tid -> list of consuming op indices
    """
    cons = defaultdict(list)
    for i, op in enumerate(g.ops):
        for tid in op.inputs:
            cons[tid].append(i)
    return cons


def build_op_consumers(g: GraphIR) -> Dict[int, Set[int]]:
    """
    op index -> set of op indices that consume its outputs
    """
    prod = build_producer_map(g)
    cons = build_consumer_map(g)

    op_cons = defaultdict(set)
    for tid, producer in prod.items():
        for c in cons.get(tid, []):
            op_cons[producer].add(c)
    return op_cons


def toposort_ops(g: GraphIR) -> List[int]:
    """
    Topologically sort ops using Kahn's algorithm.
    """
    indeg = [0] * len(g.ops)
    op_cons = build_op_consumers(g)

    for u, vs in op_cons.items():
        for v in vs:
            indeg[v] += 1

    q = deque([i for i, d in enumerate(indeg) if d == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in op_cons.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    assert len(order) == len(g.ops), "Graph has cycles"
    return order


def single_consumer_only(g: GraphIR, op_idx: int) -> bool:
    """
    Returns True iff all outputs of op_idx have at most one consumer.
    """
    cons = build_consumer_map(g)
    for tid in g.ops[op_idx].outputs:
        if len(cons.get(tid, [])) > 1:
            return False
    return True
