import mlx.core as mx

from tracing.tracing import enable_tracing, disable_tracing
from tracing.tensors import TraceContext

# ---- Create test tensors ----
x = mx.random.normal((2, 4))
w = mx.random.normal((4, 8))
b = mx.random.normal((8,))

# ---- Run traced computation ----
enable_tracing()

y = mx.softmax(mx.matmul(x, w) + b)

disable_tracing()

# ---- Inspect trace ----
print("=== OPS ===")
for i, op in enumerate(TraceContext.ops):
    print(f"{i}: {op.op} | inputs={op.inputs} | outputs={op.outputs}")

print("\n=== TENSORS ===")
for tid, t in TraceContext.tensors.items():
    print(
        f"tid={tid}, shape={t.shape}, dtype={t.dtype}, "
        f"producer={t.producer}, consumers={t.consumers}"
    )
