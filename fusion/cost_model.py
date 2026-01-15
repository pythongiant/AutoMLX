# fusion/cost_model.py
"""
Production-grade, research-informed fusion cost model.

This model is designed to be:
- Conservative (avoid bad fusions)
- Unit-consistent (benefit and penalty comparable)
- Deterministic
- Extensible to hardware-aware tuning later

Key principles (XLA / TVM / TorchInductor inspired):
- Benefit ≈ eliminated off-chip memory traffic + saved kernel launches
- Penalty ≈ peak live memory pressure + reduction/broadcast complexity
- Fusion is profitable iff benefit > penalty
"""

from typing import Iterable, List, Tuple, Union

from graph.ir import GraphIR
from fusion.regions import FusionRegion

# -----------------------------------------------------------------------------
# Tunable constants (safe, conservative defaults)
# -----------------------------------------------------------------------------

BYTES_PER_ELEMENT_DEFAULT = 4        # fp32 fallback
KERNEL_LAUNCH_EQUIV_BYTES = 1_000_000  # ~1MB memory-equivalent launch cost
LIVE_BYTES_SCALE = 0.4               # fraction of peak-live counted as penalty

REDUCTION_PENALTY_BYTES = 600_000    # per reduction op
BROADCAST_PENALTY_BYTES = 200_000    # per broadcast/view-like op

LARGE_LIVE_BYTES_THRESHOLD = 256 * 1024   # 256 KB (register/shared-memory cap)
LARGE_LIVE_BYTES_HARD_PENALTY = 4_000_000


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _iter_ops(region: Union[FusionRegion, Iterable[int]]) -> List[int]:
    if isinstance(region, FusionRegion):
        return list(region.ops)
    return list(region)


def _numel(shape) -> int:
    if not shape:
        return 0
    n = 1
    for d in shape:
        if d is None:
            return 0
        n *= int(d)
    return int(n)


def _dtype_bytes(dtype) -> int:
    if dtype is None:
        return BYTES_PER_ELEMENT_DEFAULT
    name = getattr(dtype, "name", str(dtype)).lower()
    if "16" in name:
        return 2
    if "64" in name:
        return 8
    if "8" in name:
        return 1
    return 4


def _tensor_bytes(g: GraphIR, tid: int) -> int:
    t = g.tensors.get(tid)
    if t is None or not t.shape:
        return 0
    return _numel(t.shape) * _dtype_bytes(t.dtype)


# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------

def _is_eliminated_tensor(g: GraphIR, tid: int, region_ops: List[int]) -> bool:
    """
    Tensor eliminated by fusion if:
    - produced inside region
    - all consumers inside region
    - not a graph output
    """
    producer = None
    for i in region_ops:
        if tid in g.ops[i].outputs:
            producer = i
            break

    if producer is None:
        return False

    if tid in getattr(g, "outputs", []):
        return False

    info = g.tensors.get(tid)
    if info is None:
        return False

    for c in getattr(info, "consumers", []):
        if c not in region_ops:
            return False

    return True


def _count_special_ops(g: GraphIR, ops: List[int]) -> Tuple[int, int]:
    reductions = 0
    broadcasts = 0
    for i in ops:
        name = g.ops[i].op
        if name in {"sum", "mean", "max", "min", "argmax", "argmin", "logsumexp", "softmax"}:
            reductions += 1
        if name in {"reshape", "view", "broadcast_to", "expand_dims", "tile", "repeat"}:
            broadcasts += 1
    return reductions, broadcasts


def _peak_live_bytes(g: GraphIR, ops: List[int]) -> int:
    """
    Conservative upper bound: sum of all outputs in region.
    """
    total = 0
    for i in ops:
        for tid in g.ops[i].outputs:
            total += _tensor_bytes(g, tid)
    return total


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def cost_model(
    g: GraphIR,
    region: Union[FusionRegion, Iterable[int]],
) -> Tuple[float, float]:
    """
    Returns (benefit_bytes_equiv, penalty_bytes_equiv).

    Fusion should proceed iff benefit > penalty.
    """
    ops = _iter_ops(region)
    if not ops:
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Benefit
    # ------------------------------------------------------------------

    # (1) Eliminated memory traffic (write + read)
    eliminated_bytes = 0
    for i in ops:
        for tid in g.ops[i].outputs:
            if _is_eliminated_tensor(g, tid, ops):
                eliminated_bytes += 2 * _tensor_bytes(g, tid)

    # (2) Kernel launch savings
    launch_saved = max(0, len(ops) - 1)
    launch_benefit = launch_saved * KERNEL_LAUNCH_EQUIV_BYTES

    benefit = eliminated_bytes + launch_benefit

    # ------------------------------------------------------------------
    # Penalty
    # ------------------------------------------------------------------

    peak_live = _peak_live_bytes(g, ops)
    penalty = peak_live * LIVE_BYTES_SCALE

    n_reductions, n_broadcasts = _count_special_ops(g, ops)
    penalty += n_reductions * REDUCTION_PENALTY_BYTES
    penalty += n_broadcasts * BROADCAST_PENALTY_BYTES

    if peak_live > LARGE_LIVE_BYTES_THRESHOLD:
        penalty += LARGE_LIVE_BYTES_HARD_PENALTY * (
            peak_live / LARGE_LIVE_BYTES_THRESHOLD
        )

    return float(benefit), float(penalty)
