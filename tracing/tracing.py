import mlx.core as mx

from tracing.tensors import *
from tracing.trace_op import make_traced_op

OPS_TO_TRACE = [
    # ---- elementwise unary ----
    "abs", "negative", "square", "sqrt", "rsqrt",
    "exp", "expm1",
    "log", "log1p", "log2", "log10",
    "sin", "cos", "tan",
    "sinh", "cosh", "tanh",
    "arcsin", "arccos", "arctan", "arctan2",
    "arcsinh", "arccosh", "arctanh",
    "sigmoid", "sign", "erf", "erfinv",

    # ---- elementwise binary ----
    "add", "subtract", "multiply", "divide",
    "power", "remainder", "floor_divide", "divmod",
    "maximum", "minimum", "logaddexp",

    # ---- comparisons / logical ----
    "equal", "not_equal",
    "greater", "greater_equal",
    "less", "less_equal",
    "logical_and", "logical_or", "logical_not",
    "isfinite", "isinf", "isnan", "isposinf", "isneginf",

    # ---- reductions ----
    "sum", "mean", "prod",
    "min", "max",
    "var", "std",
    "argmin", "argmax",
    "logsumexp", "logcumsumexp",
    "cumprod", "cumsum", "cummax", "cummin",

    # ---- linear algebra ----
    "matmul", "addmm",
    "inner", "outer",
    "tensordot",
    "einsum",

    # ---- attention / specialized matmul ----
    "softmax",
    "quantized_matmul",
    "block_masked_mm",
    "gather_mm",
    "gather_qmm",

    # ---- convolution ----
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "conv_general",

    # ---- shape / view ----
    "reshape", "view",
    "transpose", "swapaxes", "moveaxis",
    "squeeze", "expand_dims",
    "flatten", "unflatten",
    "broadcast_to", "broadcast_arrays",
    "tile", "repeat",
    "split", "stack", "concatenate",

    # ---- indexing / slicing ----
    "slice", "slice_update",
    "take", "take_along_axis",
    "put_along_axis",

    # ---- sorting / selection ----
    "sort", "argsort",
    "partition", "argpartition",
    "topk",

    # ---- creation ----
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full",
    "eye", "identity",
    "arange", "linspace",
    "meshgrid",
    "diag", "diagonal",
    "tri", "tril", "triu",

    # ---- misc math ----
    "clip", "where", "nan_to_num",
    "degrees", "radians",
    "real", "imag", "conj", "conjugate",
    "hadamard_transform",
]


# tracing/tracing.py (extend enable_tracing)

_ARRAY_OPS = {
    "__add__": "add",
    "__radd__": "add",
    "__sub__": "subtract",
    "__rsub__": "subtract",
    "__mul__": "multiply",
    "__rmul__": "multiply",
    "__truediv__": "divide",
    "__rtruediv__": "divide",
    "__neg__": "negative",
}

_original_ops = {}
_original_array_ops = {}

def enable_tracing():
    TraceContext.enabled = True
    TraceContext.next_tid = 0
    TraceContext.tensor_ids.clear()
    TraceContext.tensors.clear()
    TraceContext.ops.clear()

    # wrap mx functional ops
    for name in OPS_TO_TRACE:
        _original_ops[name] = getattr(mx, name)
        setattr(mx, name, make_traced_op(name, _original_ops[name]))

    # wrap array magic ops
    for meth, opname in _ARRAY_OPS.items():
        if hasattr(mx.array, meth):
            _original_array_ops[meth] = getattr(mx.array, meth)
            setattr(
                mx.array,
                meth,
                make_traced_op(opname, _original_array_ops[meth])
            )

def disable_tracing():
    TraceContext.enabled = False

    for name, fn in _original_ops.items():
        setattr(mx, name, fn)
    _original_ops.clear()

    for meth, fn in _original_array_ops.items():
        setattr(mx.array, meth, fn)
    _original_array_ops.clear()


