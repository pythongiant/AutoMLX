# validate/test_fusion.py
import mlx.core as mx
from tracing.tracing import enable_tracing, disable_tracing
from tracing.tensors import TraceContext
from compile.pipeline import compile_from_trace
from runtime.execute import run_graph

def main():
    x = mx.random.normal((2, 4))
    w = mx.random.normal((4, 8))
    b = mx.random.normal((8,))

    # eager
    y_ref = mx.softmax(mx.matmul(x, w) + b)

    # trace
    enable_tracing()
    y = mx.softmax(mx.matmul(x, w) + b)
    disable_tracing()

    g, regions = compile_from_trace(TraceContext)

    print("Regions:")
    for r in regions:
        print(r.kind, r.template, [g.ops[i].op for i in r.op_indices])

    # Provide inputs by tid: you need to map tids for x,w,b (you can do this by reading TraceContext.tensor_ids during tracing)
    # For now, simplest: treat producer=None as inputs and provide in order.
    input_tids = [tid for tid, t in g.tensors.items() if t.producer is None]
    # This ordering is not guaranteed; you should store a mapping in tracing later.
    env = run_graph(g, {input_tids[0]: x, input_tids[1]: w, input_tids[2]: b})
    y_out = env[g.outputs[0]]

    print("allclose:", mx.allclose(y_ref, y_out).item())

if __name__ == "__main__":
    main()
