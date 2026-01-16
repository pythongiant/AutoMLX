# kernel/validate.py
"""
Structural validation for KernelIR.
Fails fast on malformed kernels.
"""

from kernel.ir import KernelIR


def validate_kernel(kernel: KernelIR) -> None:
    # Inputs / outputs disjoint
    assert not set(kernel.inputs) & set(kernel.outputs)

    # Ops must exist
    assert kernel.ops, "Kernel has no ops"

    # Every op output must be declared
    declared = set(kernel.inputs) | set(kernel.outputs) | set(kernel.temps)

    for op in kernel.ops:
        for tid in op.inputs:
            assert tid in declared, f"Input tid {tid} not declared"
        for tid in op.outputs:
            assert tid in declared, f"Output tid {tid} not declared"

    # Outputs must be produced
    produced = set()
    for op in kernel.ops:
        produced.update(op.outputs)

    for out in kernel.outputs:
        assert out in produced, f"Kernel output {out} not produced"
