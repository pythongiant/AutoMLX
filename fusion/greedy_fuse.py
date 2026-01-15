# fusion/greedy_fuse.py
"""
Greedy forward fusion pass.

Deterministic, legality-first, cost-gated fusion.

Key invariants:
- Never crosses barriers (GEMM, IO, sort, etc.)
- Never crosses reduction boundaries
- Enforces SSA single-consumer safety
- Fusion accepted iff incremental benefit > incremental penalty

This matches XLA / TVM / TorchInductor phase-1 fusion behavior.
"""

from dataclasses import dataclass
from typing import List

from graph.ir import GraphIR
from fusion.regions import FusionRegion, can_fuse_pair
from fusion.cost_model import cost_model


# -----------------------------------------------------------------------------
# Output structure
# -----------------------------------------------------------------------------

@dataclass
class FusedRegion:
    """
    Final fusion decision.

    These ops will be executed / lowered as a single kernel.
    """
    op_indices: List[int]
    benefit: float
    penalty: float



# -----------------------------------------------------------------------------
# Greedy fusion
# -----------------------------------------------------------------------------

def greedy_fuse(g: GraphIR) -> List[FusedRegion]:
    """
    Greedy forward fusion over legality-only base regions.

    Algorithm:
    - Iterate regions left-to-right
    - Attempt to merge adjacent regions
    - Accept merge iff incremental benefit > incremental penalty
    - Never backtrack
    """
    MAX_ALLOWED_PENALTY = 50_000_000  # 50MB-equivalent (conservative)

    base_regions: List[FusionRegion] = g.regions
    fused: List[FusedRegion] = []

    i = 0
    while i < len(base_regions):
        # start new fused region
        cur_ops = list(base_regions[i].op_indices)
        cur_benefit, cur_penalty = cost_model(g, cur_ops)

        j = i + 1
        while j < len(base_regions):
            next_ops = base_regions[j].op_indices

            # Hard legality check: boundary ops only
            if not can_fuse_pair(g, cur_ops[-1], next_ops[0]):
                break

            merged_ops = cur_ops + next_ops
            merged_benefit, merged_penalty = cost_model(g, merged_ops)
            if merged_penalty > MAX_ALLOWED_PENALTY:
                break
            # Incremental profitability gate
            delta_benefit = merged_benefit - cur_benefit
            delta_penalty = merged_penalty - cur_penalty

            if delta_benefit <= delta_penalty:
                break

            # Accept merge
            cur_ops = merged_ops
            cur_benefit = merged_benefit
            cur_penalty = merged_penalty
            j += 1

        fused.append(
            FusedRegion(
                op_indices=cur_ops,
                benefit=cur_benefit,
                penalty=cur_penalty,
            )
        )

        i = j

    return fused
