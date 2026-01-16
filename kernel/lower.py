# kernel/lower.py
"""
GraphIR / FusionRegion → KernelIR lowering.
"""

from typing import List, Set

from graph.ir import GraphIR
from fusion.regions import FusionRegion
from kernel.ir import KernelIR, KernelOp, KernelTensor


def lower_region_to_kernel(g: GraphIR, region: FusionRegion) -> KernelIR:
    region_ops = region.op_indices

    produced: Set[int] = set()
    consumed: Set[int] = set()

    for i in region_ops:
        produced.update(g.ops[i].outputs)
        consumed.update(g.ops[i].inputs)

    # Kernel inputs = consumed but not produced
    kernel_inputs = sorted(consumed - produced)

    # Kernel outputs = produced that escape region OR graph outputs
    kernel_outputs: Set[int] = set()
    for tid in produced:
        consumers = g.tensors[tid].consumers
        if any(c not in region_ops for c in consumers):
            kernel_outputs.add(tid)
        if tid in g.outputs:
            kernel_outputs.add(tid)

    kernel_outputs = sorted(kernel_outputs)

    # Temps = produced but not outputs
    kernel_temps = sorted(produced - set(kernel_outputs))

    # Tensors table
    tensors = {}
    for tid in kernel_inputs:
        t = g.tensors[tid]
        tensors[tid] = KernelTensor(tid, t.shape, t.dtype, "input")

    for tid in kernel_outputs:
        t = g.tensors[tid]
        tensors[tid] = KernelTensor(tid, t.shape, t.dtype, "output")

    for tid in kernel_temps:
        t = g.tensors[tid]
        tensors[tid] = KernelTensor(tid, t.shape, t.dtype, "temp")

    # Lower ops
    kernel_ops: List[KernelOp] = []
    for i in region_ops:
        op = g.ops[i]
        kernel_ops.append(
            KernelOp(
                op=op.op,
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs=dict(op.attrs or {}),
                # kind intentionally omitted → derived later
            )
        )

    return KernelIR(
        inputs=kernel_inputs,
        outputs=kernel_outputs,
        temps=kernel_temps,
        tensors=tensors,
        ops=kernel_ops,
    )
