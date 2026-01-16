# kernel/ir.py
"""
KernelIR — a low-level, backend-agnostic intermediate representation.

Position in the stack:
    GraphIR  →  FusionRegion  →  KernelIR  →  Backend codegen (MLX / generated)

Design goals:
- Explicit dataflow
- Explicit ABI (inputs / outputs / temporaries)
- No framework imports (NO mlx / torch / triton here)
- Safe target for AI-generated kernels
- Deterministic, serializable, verifiable

NOTE:
This IR currently supports *both*:
- ID-based kernels (compiler-generated)
- Name-based kernels (AI / ABI-generated)

This file is declarative only.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum, auto


# -----------------------------------------------------------------------------
# Core enums
# -----------------------------------------------------------------------------

class KernelOpKind(Enum):
    ELEMENTWISE = auto()
    REDUCTION = auto()
    BROADCAST = auto()
    GEMM = auto()
    LOAD = auto()
    STORE = auto()
    VIEW = auto()
    CONSTANT = auto()
    OTHER = auto()


class MemorySpace(Enum):
    INPUT = auto()
    OUTPUT = auto()
    TEMP = auto()
    CONSTANT = auto()


# -----------------------------------------------------------------------------
# ABI objects (future-facing)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelArg:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    space: MemorySpace  # INPUT or OUTPUT

    def __post_init__(self):
        if self.space not in (MemorySpace.INPUT, MemorySpace.OUTPUT):
            raise ValueError("KernelArg must be INPUT or OUTPUT")


@dataclass
class KernelTensor:
    """
    Compiler-facing tensor descriptor (ID-based).
    """
    tid: int
    role: str          # "input" | "output" | "temp"
    shape: tuple | None
    dtype: Any | None


@dataclass(frozen=True)
class KernelTemp:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    space: MemorySpace = MemorySpace.TEMP


# -----------------------------------------------------------------------------
# Kernel operations
# -----------------------------------------------------------------------------

@dataclass
class KernelOp:
    """
    Single operation inside a kernel.

    Compatible with:
    - ID-based lowering (inputs/outputs are ints)
    - Name-based ABI kernels (inputs/outputs are strings)
    """

    op: str
    inputs: List[Union[int, str]]
    outputs: List[Union[int, str]]
    attrs: Dict[str, object] = field(default_factory=dict)

    # Optional semantic kind (derived later if needed)
    kind: Optional[KernelOpKind] = None

    def __post_init__(self):
        if not self.outputs:
            raise ValueError("KernelOp must produce at least one output")


# -----------------------------------------------------------------------------
# Launch / scheduling metadata
# -----------------------------------------------------------------------------

@dataclass
class LaunchConfig:
    grid: Optional[Tuple[int, ...]] = None
    block: Optional[Tuple[int, ...]] = None
    num_warps: Optional[int] = None
    shared_mem_bytes: Optional[int] = None


# -----------------------------------------------------------------------------
# KernelIR container
# -----------------------------------------------------------------------------

@dataclass
class KernelIR:
    """
    Fully self-contained kernel definition.

    This supports:
    - Compiler-generated kernels (ID-based)
    - AI-generated kernels (name-based)
    """

    # Optional human-readable name
    name: Optional[str] = None

    # Compiler-style ABI (ID-based)
    inputs: List[int] = field(default_factory=list)
    outputs: List[int] = field(default_factory=list)
    temps: List[int] = field(default_factory=list)

    # Tensor table (ID-based)
    tensors: Dict[int, KernelTensor] = field(default_factory=dict)

    # Program
    ops: List[KernelOp] = field(default_factory=list)

    # Optional launch hints
    launch: Optional[LaunchConfig] = None

    # Provenance
    source_ops: Optional[List[int]] = None

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def validate(self) -> None:
        """
        Structural validation.

        Works for ID-based kernels (current compiler path).
        """
        defined = set(self.tensors.keys())
        written = set()

        for op in self.ops:
            for i in op.inputs:
                if isinstance(i, int) and i not in defined:
                    raise AssertionError(f"Read-before-write: tensor {i}")

            for o in op.outputs:
                if isinstance(o, int):
                    if o not in defined:
                        raise AssertionError(f"Write to undeclared tensor {o}")
                    written.add(o)

        for out in self.outputs:
            if out not in written:
                raise AssertionError(f"Kernel output {out} is never written")

        for t in self.temps:
            if t not in written:
                raise AssertionError(f"Dead temporary {t}")

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def num_ops(self) -> int:
        return len(self.ops)

    def num_temps(self) -> int:
        return len(self.temps)

    def summary(self) -> str:
        return (
            f"KernelIR("
            f"inputs={len(self.inputs)}, "
            f"outputs={len(self.outputs)}, "
            f"temps={len(self.temps)}, "
            f"ops={len(self.ops)})"
        )
