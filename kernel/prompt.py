# kernel/prompt.py
"""
Deterministic KernelIR → LLM prompt serialization.

NO abstractions.
NO KernelArg.
NO execution.
NO generation.

Pure string serialization of KernelIR state.
"""

from kernel.ir import KernelIR, KernelTensor


SYSTEM_PROMPT = """
You are generating a PURE MLX kernel.

Rules:
- Do NOT import anything except mlx.core as mx
- Do NOT mutate global state
- Do NOT allocate new Python objects except arrays
- Do NOT change shapes or dtypes
- Inputs and outputs MUST exactly match the contract
- Return outputs in the declared order
""".strip()


def build_kernel_prompt(kernel: KernelIR) -> str:
    lines: list[str] = []

    # ------------------------------------------------------------------
    # SYSTEM
    # ------------------------------------------------------------------
    lines.append("### SYSTEM")
    lines.append(SYSTEM_PROMPT)
    lines.append("")

    # ------------------------------------------------------------------
    # KERNEL
    # ------------------------------------------------------------------
    lines.append("### KERNEL")
    name = kernel.name or "anonymous_kernel"
    lines.append(f"name: {name}")
    lines.append("")

    # ------------------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------------------
    lines.append("### INPUTS")
    if not kernel.inputs:
        lines.append("(none)")
    else:
        for tid in kernel.inputs:
            t: KernelTensor = kernel.tensors[tid]
            lines.append(
                f"tid={tid} shape={t.shape} dtype={t.dtype}"
            )
    lines.append("")

    # ------------------------------------------------------------------
    # TEMPS
    # ------------------------------------------------------------------
    lines.append("### TEMPS")
    if not kernel.temps:
        lines.append("(none)")
    else:
        for tid in kernel.temps:
            t: KernelTensor = kernel.tensors[tid]
            lines.append(
                f"tid={tid} shape={t.shape} dtype={t.dtype}"
            )
    lines.append("")

    # ------------------------------------------------------------------
    # OPS
    # ------------------------------------------------------------------
    lines.append("### OPS")
    for idx, op in enumerate(kernel.ops):
        lines.append(
            f"{idx}: {op.op}("
            + ", ".join(f"t{tid}" for tid in op.inputs)
            + ") -> "
            + ", ".join(f"t{tid}" for tid in op.outputs)
        )
        if op.attrs:
            for k, v in sorted(op.attrs.items()):
                lines.append(f"    attr {k} = {v}")
    lines.append("")

    # ------------------------------------------------------------------
    # OUTPUTS
    # ------------------------------------------------------------------
    lines.append("### OUTPUTS")
    for tid in kernel.outputs:
        t: KernelTensor = kernel.tensors[tid]
        lines.append(
            f"tid={tid} shape={t.shape} dtype={t.dtype}"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # REQUIRED OUTPUT
    # ------------------------------------------------------------------
    lines.append("### REQUIRED OUTPUT")
    lines.append("Write a function with the exact signature:")
    lines.append("")
    args = ", ".join(f"t{tid}" for tid in kernel.inputs)
    lines.append(f"def kernel({args}):")
    lines.append("    # compute ops in order")
    if len(kernel.outputs) == 1:
        lines.append(f"    return t{kernel.outputs[0]}")
    else:
        outs = ", ".join(f"t{tid}" for tid in kernel.outputs)
        lines.append(f"    return {outs}")

    return "\n".join(lines)
