import argparse
import os
import onnx
from onnxruntime_tools import optimizer
from onnxconverter_common import float16
from onnx import helper, TensorProto, checker

BINOPS = {'Add','Mul','Sub','Div','Pow'}
VARIADIC_OPS = {'Concat'}

def build_dtype_maps(model):
    init_dtype = {init.name: init.data_type for init in model.graph.initializer}
    vi_dtype = {}
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        for vi in list(inferred.graph.value_info)+list(inferred.graph.input)+list(inferred.graph.output):
            t = vi.type.tensor_type
            if t.HasField('elem_type'):
                vi_dtype[vi.name] = t.elem_type
    except Exception:
        pass
    return init_dtype, vi_dtype

def get_dtype(name, init_dtype, vi_dtype):
    return vi_dtype.get(name) or init_dtype.get(name)

def insert_cast_before(graph, node, input_idx, to_dtype=TensorProto.FLOAT16):
    src = node.input[input_idx]
    cast_out = f"{src}_cast_fp16__{node.name}_in{input_idx}"
    cast = helper.make_node(
        'Cast',
        name=f'Cast_for_{node.name}_in{input_idx}',
        inputs=[src],
        outputs=[cast_out],
        to=to_dtype
    )
    node_idx = list(graph.node).index(node)
    graph.node.insert(node_idx, cast)
    node.input[input_idx] = cast_out

def fix_mixed_precision(path_in, path_out):
    m = onnx.load(path_in)
    g = m.graph
    init_dtype, vi_dtype = build_dtype_maps(m)

    fixed = 0
    for n in list(g.node):

        if n.op_type in BINOPS and len(n.input) >= 2:
            dtypes = [get_dtype(n.input[i], init_dtype, vi_dtype) for i in range(2)]
            if set(dtypes) == {TensorProto.FLOAT, TensorProto.FLOAT16}:
                for i, dt in enumerate(dtypes):
                    if dt == TensorProto.FLOAT:
                        insert_cast_before(g, n, i, TensorProto.FLOAT16)
                        fixed += 1

        if n.op_type in VARIADIC_OPS and len(n.input) >= 2:
            dts = [get_dtype(inp, init_dtype, vi_dtype) for inp in n.input]
            s = set(dts) - {None}
            if TensorProto.FLOAT in s and TensorProto.FLOAT16 in s:
                for i, dt in enumerate(dts):
                    if dt == TensorProto.FLOAT:
                        insert_cast_before(g, n, i, TensorProto.FLOAT16)
                        fixed += 1

    checker.check_model(m)
    onnx.save(m, path_out)
    print(f"[Mixed-Precision Fix] Cast inserted: {fixed}")

def main():
    parser = argparse.ArgumentParser(description="Convert FP32 ONNX model to FP16 and fix mixed precision")
    parser.add_argument("--model", required=True, help="Path to FP32 ONNX model")
    args = parser.parse_args()

    path_fp32 = os.path.abspath(args.model)
    base, ext = os.path.splitext(path_fp32)

    path_fp16_in  = f"{base}_fp16.onnx"
    path_fp16_out = f"{base}_fp16_mixed.onnx"

    print(f"Input FP32 Model: {path_fp32}")
    print(f"Output FP16 Model: {path_fp16_in}")
    print(f"Output Mixed Model: {path_fp16_out}")

    model = onnx.load(path_fp32)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
        op_block_list=['Cast','Mul','Add','Sub','Div','Pow']
    )

    onnx.save(model_fp16, path_fp16_in)
    print("[FP32 → FP16] Done")

    fix_mixed_precision(path_fp16_in, path_fp16_out)
    print("All conversions completed successfully!")

if __name__ == "__main__":
    main()
