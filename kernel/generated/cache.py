# kernel/generated/cache.py
"""
Generated kernel cache.

Maps a deterministic kernel signature to a generated kernel callable.
This cache is:
- in-memory only
- deterministic
- explicitly invalidatable
"""

from typing import Callable, Dict
# kernel/generated/cache.py

from typing import Callable, Dict, Tuple
from kernel.ir import KernelIR

# -----------------------------------------------------------------------------
# Global in-memory cache (process-local, deterministic)
# -----------------------------------------------------------------------------

_KERNEL_CACHE: Dict[str, Callable] = {}


# -----------------------------------------------------------------------------
# Signature computation
# -----------------------------------------------------------------------------

def kernel_signature(kernel: KernelIR) -> str:
    """
    Compute a deterministic, collision-resistant signature
    for a KernelIR.

    This MUST be stable across runs.
    """

    parts = []

    # Ops (order matters)
    ops = ",".join(op.op for op in kernel.ops)
    parts.append(f"ops={ops}")

    # Inputs
    in_sig = []
    for tid in kernel.inputs:
        t = kernel.tensors[tid]
        in_sig.append(f"{t.shape}:{t.dtype}")
    parts.append(f"inputs={in_sig}")

    # Outputs
    out_sig = []
    for tid in kernel.outputs:
        t = kernel.tensors[tid]
        out_sig.append(f"{t.shape}:{t.dtype}")
    parts.append(f"outputs={out_sig}")

    return "|".join(parts)



# signature (str) -> kernel callable
_GENERATED_KERNEL_CACHE: Dict[str, Callable] = {}

def get_kernel(signature: str) -> Callable | None:
    """
    Retrieve a kernel if present.
    """
    return _KERNEL_CACHE.get(signature)

def put_kernel(signature: str, fn: Callable) -> None:
    """
    Register a generated kernel.

    Overwrites are allowed but explicit.
    """
    _KERNEL_CACHE[signature] = fn


def remove_kernel(signature: str):
    _GENERATED_KERNEL_CACHE.pop(signature, None)

def clear_cache() -> None:
    _KERNEL_CACHE.clear()

def cache_size() -> int:
    return len(_GENERATED_KERNEL_CACHE)
