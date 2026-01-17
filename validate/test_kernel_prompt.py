# validate/test_kernel_prompt.py
"""
Tests for KernelIR → prompt serialization.

These tests verify:
- determinism
- completeness
- correctness w.r.t *actual KernelIR contents*
- no execution or side-effect language
"""

import mlx.core as mx

from compile.pipeline import trace_and_compile
from kernel.lower import lower_region_to_kernel
from kernel.prompt import build_kernel_prompt


def test_kernel_prompt_is_deterministic_and_complete():
    def fn(x):
        return mx.exp(mx.sqrt(mx.abs(x) + 1))

    x = mx.random.normal((8, 8))
    g = trace_and_compile(fn, [x])

    # Use first region only (legality-based, single-op)
    region = g.regions[0]
    kernel = lower_region_to_kernel(g, region)

    prompt1 = build_kernel_prompt(kernel)
    prompt2 = build_kernel_prompt(kernel)

    # Deterministic
    assert prompt1 == prompt2

    # Required sections
    for section in [
        "### SYSTEM",
        "### INPUTS",
        "### TEMPS",
        "### OPS",
        "### OUTPUTS",
        "### REQUIRED OUTPUT",
    ]:
        assert section in prompt1

    # All ops in kernel must appear
    for op in kernel.ops:
        assert op.op in prompt1

    # Kernel inputs must appear
    for tid in kernel.inputs:
        assert f"t{tid}" in prompt1

    # Kernel outputs must appear
    for tid in kernel.outputs:
        assert f"t{tid}" in prompt1


def test_prompt_includes_multi_output_ops():
    def fn(a, b):
        X, Y = mx.meshgrid(a, b)
        return X, Y

    a = mx.arange(3)
    b = mx.arange(4)
    g = trace_and_compile(fn, [a, b])

    # Find meshgrid region
    region = None
    for r in g.regions:
        if any(g.ops[i].op == "meshgrid" for i in r.op_indices):
            region = r
            break

    assert region is not None

    kernel = lower_region_to_kernel(g, region)
    prompt = build_kernel_prompt(kernel)

    # Op present
    assert "meshgrid" in prompt

    # Exactly one return statement
    assert prompt.count("return") == 1

    # Multi-output return must be tuple-like
    return_line = prompt.split("return")[-1]
    assert "," in return_line

    # All output tids must be referenced
    for tid in kernel.outputs:
        assert f"t{tid}" in prompt


def test_prompt_contains_no_execution_or_side_effects():
    def fn(x):
        return x + 1

    x = mx.ones((4,))
    g = trace_and_compile(fn, [x])
    kernel = lower_region_to_kernel(g, g.regions[0])

    prompt = build_kernel_prompt(kernel)

    forbidden = [
        "import os",
        "import sys",
        "open(",
        "exec(",
        "eval(",
        "__import__",
        "subprocess",
        "while True",
    ]

    for bad in forbidden:
        assert bad not in prompt
