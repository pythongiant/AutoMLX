# runtime/execute_fused.py

import mlx.core as mx
from typing import Dict, List

from graph.ir import GraphIR, OpIR
from fusion.regions import FusionRegion


def run_fused_regions(
    g: GraphIR,
    regions: List[FusionRegion],
    inputs: Dict[int, mx.array],
) -> Dict[int, mx.array]:
    """
    Execute fused regions as single units (fallback execution).

    Semantics:
    - Regions are executed sequentially in graph order
    - Inside a region, ops are executed in program order
    - No kernel fusion yet: this is a correctness-preserving fallback
    """

    # Environment: tid -> array
    env: Dict[int, mx.array] = dict(inputs)

    # Map op_idx -> region
    op_to_region = {}
    for r in regions:
        for op_idx in r.ops:
            op_to_region[op_idx] = r

    executed_regions = set()

    for r in regions:
        if id(r) in executed_regions:
            continue

        # Execute ops in region order
        for op_idx in r.ops:
            op: OpIR = g.ops[op_idx]

            fn = getattr(mx, op.op)

            # rebuild args (respect const_args)
            args = []
            tensor_iter = iter(op.inputs)

            for a in op.const_args:
                if a is None:
                    tid = next(tensor_iter)
                    args.append(env[tid])
                else:
                    args.append(a)

            out = fn(*args, **(op.attrs or {}))

            # handle single or multi-output
            if len(op.outputs) == 1:
                env[op.outputs[0]] = out
            else:
                assert len(op.outputs) == len(out)
                for tid, val in zip(op.outputs, out):
                    env[tid] = val

        executed_regions.add(id(r))

    return env
