"""Rewrite a .tflite file's attention matmuls from [B, M, K] @ [B, K, N] to
[1, B, M, K] @ [1, B, K, N].

The converter lowers every matmul to rank 3 with RESHAPEs around it. The TFLite GPU
delegate runs a matmul with batch 1 directly (one weight matrix per row of H), but
any other batch goes through extra reshapes that copy the operands and the result:
for attention that is three copies of the scores. This drops the RESHAPEs and makes
the matmuls rank 4.

Usage:  python -I tflite_matmul4d.py IN.tflite OUT.tflite     (or import `rewrite`)
"""
import sys

import flatbuffers
from ai_edge_litert import schema_py_generated as sch


def rewrite(src, dst):
    model = sch.ModelT.InitFromPackedBuf(open(src, "rb").read(), 0)
    (g,) = model.subgraphs
    code = lambda op: model.operatorCodes[op.opcodeIndex].builtinCode
    shape = lambda t: [int(d) for d in g.tensors[t].shape]
    producer = {int(t): op for op in g.operators for t in op.outputs}
    replace, n = {}, 0  # tensor -> the tensor to use instead
    for op in g.operators:
        if code(op) != sch.BuiltinOperator.BATCH_MATMUL or len(shape(op.outputs[0])) != 3:
            continue
        ins = []
        for t in op.inputs:
            r = producer.get(int(t))
            if r is None or code(r) != sch.BuiltinOperator.RESHAPE or shape(r.inputs[0]) != [1] + shape(t):
                break
            ins.append(r.inputs[0])
        else:
            op.inputs = ins
            out = g.tensors[op.outputs[0]]
            out.shape = [1] + shape(op.outputs[0])
            out.shapeSignature = None
            n += 1
    # A RESHAPE of a rewritten result back to [1, ...] is now the identity.
    for op in g.operators:
        if code(op) == sch.BuiltinOperator.RESHAPE and shape(op.inputs[0]) == shape(op.outputs[0]):
            replace[int(op.outputs[0])] = int(op.inputs[0])
    for op in g.operators:
        op.inputs = [replace.get(int(t), int(t)) for t in op.inputs]
    g.outputs = [replace.get(int(t), int(t)) for t in g.outputs]
    # Drop operators nobody reads any more (the bypassed RESHAPEs), repeatedly.
    while True:
        used = {int(t) for op in g.operators for t in op.inputs} | {int(t) for t in g.outputs}
        keep = [op for op in g.operators if any(int(t) in used for t in op.outputs)]
        if len(keep) == len(g.operators):
            break
        g.operators = keep
    b = flatbuffers.Builder(1 << 20)
    b.Finish(model.Pack(b), file_identifier=b"TFL3")
    open(dst, "wb").write(b.Output())
    return n


if __name__ == "__main__":
    print("matmuls rewritten:", rewrite(sys.argv[1], sys.argv[2]))
