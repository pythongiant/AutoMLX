# graph/op_registry.py
from enum import Enum, auto
from typing import Dict, Set


class OpKind(Enum):
    # Pure elementwise ops (map-style)
    ELEMENTWISE = auto()

    # Reductions (sum, max, etc.)
    REDUCTION = auto()

    # Matrix / tensor contractions
    GEMM = auto()

    # Shape-only ops (no data movement, views)
    RESHAPE_VIEW = auto()

    # Indexing / slicing / gather-style ops
    INDEXING = auto()

    # Sorting / selection
    SORT_SELECT = auto()

    # Randomness / nondeterministic
    RANDOM = auto()

    # IO / serialization
    IO = auto()

    # Misc / complex / unknown semantics
    MISC = auto()


# -----------------------------------------------------------------------------
# Op classification
# -----------------------------------------------------------------------------

OP_KIND: Dict[str, OpKind] = {
    # ---- elementwise unary ----
    "abs": OpKind.ELEMENTWISE,
    "negative": OpKind.ELEMENTWISE,
    "square": OpKind.ELEMENTWISE,
    "sqrt": OpKind.ELEMENTWISE,
    "rsqrt": OpKind.ELEMENTWISE,
    "exp": OpKind.ELEMENTWISE,
    "expm1": OpKind.ELEMENTWISE,
    "log": OpKind.ELEMENTWISE,
    "log1p": OpKind.ELEMENTWISE,
    "log2": OpKind.ELEMENTWISE,
    "log10": OpKind.ELEMENTWISE,
    "sin": OpKind.ELEMENTWISE,
    "cos": OpKind.ELEMENTWISE,
    "tan": OpKind.ELEMENTWISE,
    "sinh": OpKind.ELEMENTWISE,
    "cosh": OpKind.ELEMENTWISE,
    "tanh": OpKind.ELEMENTWISE,
    "sigmoid": OpKind.ELEMENTWISE,
    "sign": OpKind.ELEMENTWISE,
    "erf": OpKind.ELEMENTWISE,
    "erfinv": OpKind.ELEMENTWISE,
    "real": OpKind.ELEMENTWISE,
    "imag": OpKind.ELEMENTWISE,
    "conj": OpKind.ELEMENTWISE,
    "conjugate": OpKind.ELEMENTWISE,

    # ---- elementwise binary ----
    "add": OpKind.ELEMENTWISE,
    "subtract": OpKind.ELEMENTWISE,
    "multiply": OpKind.ELEMENTWISE,
    "divide": OpKind.ELEMENTWISE,
    "power": OpKind.ELEMENTWISE,
    "remainder": OpKind.ELEMENTWISE,
    "floor_divide": OpKind.ELEMENTWISE,
    "maximum": OpKind.ELEMENTWISE,
    "minimum": OpKind.ELEMENTWISE,
    "logaddexp": OpKind.ELEMENTWISE,
    "equal": OpKind.ELEMENTWISE,
    "not_equal": OpKind.ELEMENTWISE,
    "greater": OpKind.ELEMENTWISE,
    "greater_equal": OpKind.ELEMENTWISE,
    "less": OpKind.ELEMENTWISE,
    "less_equal": OpKind.ELEMENTWISE,
    "logical_and": OpKind.ELEMENTWISE,
    "logical_or": OpKind.ELEMENTWISE,
    "logical_not": OpKind.ELEMENTWISE,
    "where": OpKind.ELEMENTWISE,
    "clip": OpKind.ELEMENTWISE,
    "nan_to_num": OpKind.ELEMENTWISE,

    # ---- reductions ----
    "sum": OpKind.REDUCTION,
    "mean": OpKind.REDUCTION,
    "prod": OpKind.REDUCTION,
    "min": OpKind.REDUCTION,
    "max": OpKind.REDUCTION,
    "var": OpKind.REDUCTION,
    "std": OpKind.REDUCTION,
    "argmin": OpKind.REDUCTION,
    "argmax": OpKind.REDUCTION,
    "logsumexp": OpKind.REDUCTION,
    "logcumsumexp": OpKind.REDUCTION,
    "cumprod": OpKind.REDUCTION,
    "cumsum": OpKind.REDUCTION,
    "cummax": OpKind.REDUCTION,
    "cummin": OpKind.REDUCTION,

    # ---- GEMM / contractions ----
    "matmul": OpKind.GEMM,
    "addmm": OpKind.GEMM,
    "inner": OpKind.GEMM,
    "outer": OpKind.GEMM,
    "tensordot": OpKind.GEMM,
    "einsum": OpKind.GEMM,
    "quantized_matmul": OpKind.GEMM,
    "block_masked_mm": OpKind.GEMM,
    "gather_mm": OpKind.GEMM,
    "gather_qmm": OpKind.GEMM,

    # ---- attention primitive ----
    "softmax": OpKind.REDUCTION,  # treated as reduction until canonicalized

    # ---- convolutions ----
    "conv1d": OpKind.GEMM,
    "conv2d": OpKind.GEMM,
    "conv3d": OpKind.GEMM,
    "conv_transpose1d": OpKind.GEMM,
    "conv_transpose2d": OpKind.GEMM,
    "conv_transpose3d": OpKind.GEMM,
    "conv_general": OpKind.GEMM,

    # ---- shape / view ----
    "reshape": OpKind.RESHAPE_VIEW,
    "view": OpKind.RESHAPE_VIEW,
    "transpose": OpKind.RESHAPE_VIEW,
    "swapaxes": OpKind.RESHAPE_VIEW,
    "moveaxis": OpKind.RESHAPE_VIEW,
    "squeeze": OpKind.RESHAPE_VIEW,
    "expand_dims": OpKind.RESHAPE_VIEW,
    "flatten": OpKind.RESHAPE_VIEW,
    "unflatten": OpKind.RESHAPE_VIEW,
    "broadcast_to": OpKind.RESHAPE_VIEW,
    "broadcast_arrays": OpKind.RESHAPE_VIEW,
    "tile": OpKind.RESHAPE_VIEW,
    "repeat": OpKind.RESHAPE_VIEW,

    # ---- indexing / slicing ----
    "slice": OpKind.INDEXING,
    "slice_update": OpKind.INDEXING,
    "take": OpKind.INDEXING,
    "take_along_axis": OpKind.INDEXING,
    "put_along_axis": OpKind.INDEXING,
    "gather": OpKind.INDEXING,

    # ---- sorting / selection ----
    "sort": OpKind.SORT_SELECT,
    "argsort": OpKind.SORT_SELECT,
    "partition": OpKind.SORT_SELECT,
    "argpartition": OpKind.SORT_SELECT,
    "topk": OpKind.SORT_SELECT,

    # ---- creation ----
    "zeros": OpKind.MISC,
    "zeros_like": OpKind.MISC,
    "ones": OpKind.MISC,
    "ones_like": OpKind.MISC,
    "full": OpKind.MISC,
    "eye": OpKind.MISC,
    "identity": OpKind.MISC,
    "arange": OpKind.MISC,
    "linspace": OpKind.MISC,
    "meshgrid": OpKind.MISC,
    "diag": OpKind.MISC,
    "diagonal": OpKind.MISC,
    "tri": OpKind.MISC,
    "tril": OpKind.MISC,
    "triu": OpKind.MISC,

    # ---- IO ----
    "load": OpKind.IO,
    "save": OpKind.IO,
    "savez": OpKind.IO,
    "savez_compressed": OpKind.IO,
    "save_gguf": OpKind.IO,
    "save_safetensors": OpKind.IO,
}


# -----------------------------------------------------------------------------
# Fusion barriers (conservative, Phase 1)
# -----------------------------------------------------------------------------

BARRIERS: Set[str] = {
    # IO
    "load", "save", "savez", "savez_compressed", "save_gguf", "save_safetensors",

    # Random / nondeterministic
    "random.normal",

    # Sorting / selection
    "sort", "argsort", "partition", "argpartition", "topk",

    # Multi-output shape ops (treat as barrier initially)
    "split", "meshgrid",

    # Reductions that change rank (until canonicalized)
    "argmin", "argmax",

    # Unknown / unsafe
    "einsum",  # relaxed later once legality is explicit
}


# -----------------------------------------------------------------------------
# Helper APIs
# -----------------------------------------------------------------------------

def get_op_kind(op: str) -> OpKind:
    return OP_KIND.get(op, OpKind.MISC)


def is_barrier(op: str) -> bool:
    return op in BARRIERS
