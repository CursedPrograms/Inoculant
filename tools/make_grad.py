# Inoculant - development tool (not needed to run Inoculant): the reference for gradgraph.cpp. Builds the same
# "gradient graph" from a face-recognition ONNX model in Python, and checks it against finite differences, so the
# C++ builder can be compared with it.
#
#   python make_grad.py <w600k_r50.onnx> <out_grad.onnx>
#
# The graph: inputs x [N,3,112,112] (ArcFace's (pixel - 127.5) / 127.5) and t [1,512] (a unit target embedding);
# outputs the embedding e, cos = <e/|e|, t> [N,1], and grad = d cos / d x [N,3,112,112]. Reverse mode is written out
# by hand for the six ops ArcFace (IResNet) uses: Conv, BatchNormalization, PRelu, Add, Flatten, Gemm.
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper as nh, TensorProto


def build(src, dst):
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: nh.to_array(i) for i in g.initializer}
    x_name = [i.name for i in g.input if i.name not in inits][0]
    e_name = g.output[0].name
    # shapes of every activation, for ConvTranspose's output padding
    fixed = onnx.load(src)
    fixed.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    inf = onnx.shape_inference.infer_shapes(fixed)
    shapes = {}
    for vi in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]

    nodes, new_inits = [], []
    count = [0]

    def nm(base):
        count[0] += 1
        return 'g%d_%s' % (count[0], base)

    def const(arr, base):
        n = nm(base)
        new_inits.append(nh.from_array(np.asarray(arr, dtype=np.float32), n))
        return n

    def i64(arr, base):
        n = nm(base)
        new_inits.append(nh.from_array(np.asarray(arr, dtype=np.int64), n))
        return n

    # the loss: cos = <e/|e|, t>; d cos / d e = (t - cos * e^) / |e|
    t = 't'
    norm = nm('norm'); nodes.append(helper.make_node('ReduceL2', [e_name], [norm], axes=[1], keepdims=1))
    eh = nm('eh'); nodes.append(helper.make_node('Div', [e_name, norm], [eh]))
    prod = nm('prod'); nodes.append(helper.make_node('Mul', [eh, t], [prod]))
    nodes.append(helper.make_node('ReduceSum', [prod], ['cos'], axes=[1], keepdims=1))
    ce = nm('ce'); nodes.append(helper.make_node('Mul', ['cos', eh], [ce]))
    diff = nm('diff'); nodes.append(helper.make_node('Sub', [t, ce], [diff]))
    ge = nm('ge'); nodes.append(helper.make_node('Div', [diff, norm], [ge]))

    grads = {e_name: [ge]}   # activation -> its gradient pieces (summed when used more than once)

    def grad_of(name):
        parts = grads.get(name)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        s = nm('sum'); nodes.append(helper.make_node('Sum', parts, [s]))
        grads[name] = [s]
        return s

    def add(name, gname):
        if name in inits:
            return
        grads.setdefault(name, []).append(gname)

    for n in reversed(list(g.node)):
        gy = grad_of(n.output[0])
        if gy is None:
            continue
        a = {at.name: helper.get_attribute_value(at) for at in n.attribute}
        op = n.op_type
        if op == 'Add':
            add(n.input[0], gy); add(n.input[1], gy)
        elif op == 'BatchNormalization':
            sc, b, mean, var = (inits[i] for i in n.input[1:5])
            k = sc / np.sqrt(var + a.get('epsilon', 1e-5))
            rank = len(shapes.get(n.input[0], [1, 1, 1, 1]))
            k = k.reshape([1, -1] + [1] * (rank - 2))
            o = nm('bn'); nodes.append(helper.make_node('Mul', [gy, const(k, 'bnk')], [o])); add(n.input[0], o)
        elif op == 'PRelu':
            slope = n.input[1]
            zero = const(np.zeros([1], np.float32), 'zero')
            pos = nm('pos'); nodes.append(helper.make_node('Greater', [n.input[0], zero], [pos]))
            neg = nm('neg'); nodes.append(helper.make_node('Mul', [gy, slope], [neg]))
            o = nm('prelu'); nodes.append(helper.make_node('Where', [pos, gy, neg], [o])); add(n.input[0], o)
        elif op == 'Conv':
            W = inits[n.input[1]]
            k = list(W.shape[2:])
            s = a.get('strides', [1, 1]); p = a.get('pads', [0, 0, 0, 0]); d = a.get('dilations', [1, 1]); grp = a.get('group', 1)
            ins, outs = shapes[n.input[0]], shapes[n.output[0]]
            op_ = []
            for i in range(2):
                full = (outs[2 + i] - 1) * s[i] - p[i] - p[i + 2] + (k[i] - 1) * d[i] + 1
                op_.append(ins[2 + i] - full)
            o = nm('convT')
            nodes.append(helper.make_node('ConvTranspose', [gy, n.input[1]], [o], strides=s, pads=p, dilations=d, group=grp, output_padding=op_, kernel_shape=k))
            add(n.input[0], o)
        elif op == 'Flatten':
            sh = nm('shape'); nodes.append(helper.make_node('Shape', [n.input[0]], [sh]))
            o = nm('unflat'); nodes.append(helper.make_node('Reshape', [gy, sh], [o])); add(n.input[0], o)
        elif op == 'Gemm':
            if a.get('transA', 0):
                raise SystemExit('Gemm with transA: not handled')
            B = inits[n.input[1]]
            # Y = alpha A op(B) + C; dA = alpha gY op(B)^T: B itself when transB, else B^T
            M = B if a.get('transB', 0) else B.T
            alpha = a.get('alpha', 1.0)
            o = nm('gemm'); nodes.append(helper.make_node('MatMul', [gy, const(M * alpha, 'gemmB') if (alpha != 1.0 or not a.get('transB', 0)) else n.input[1]], [o]))
            add(n.input[0], o)
        else:
            raise SystemExit('op not handled: ' + op)

    gx = grad_of(x_name)
    nodes.append(helper.make_node('Identity', [gx], ['grad']))
    g.node.extend(nodes)
    g.initializer.extend(new_inits)
    g.input.append(helper.make_tensor_value_info('t', TensorProto.FLOAT, [1, 512]))
    g.output.append(helper.make_tensor_value_info('cos', TensorProto.FLOAT, ['N', 1]))
    g.output.append(helper.make_tensor_value_info('grad', TensorProto.FLOAT, ['N', 3, 112, 112]))
    onnx.save(m, dst)
    return x_name, e_name


def check(path, x_name):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (1, 3, 112, 112)).astype(np.float32)
    t = rng.normal(size=(1, 512)).astype(np.float32); t /= np.linalg.norm(t)
    e, c, gr = s.run(None, {x_name: x, 't': t})
    for trial in range(3):
        v = rng.normal(size=x.shape).astype(np.float32); v /= np.linalg.norm(v)
        h = 1e-2
        cp = s.run(['cos'], {x_name: x + h * v, 't': t})[0][0, 0]
        cm = s.run(['cos'], {x_name: x - h * v, 't': t})[0][0, 0]
        fd = (cp - cm) / (2 * h)
        an = float((gr * v).sum())
        print('direction %d: finite difference %.6f, gradient graph %.6f' % (trial, fd, an))


if __name__ == '__main__':
    xn, _ = build(sys.argv[1], sys.argv[2])
    check(sys.argv[2], xn)
