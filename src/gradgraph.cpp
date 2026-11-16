// Inoculant - the gradient graph (see gradgraph.h). A small protobuf reader and writer: just enough of ONNX's
// ModelProto to read the forward graph's nodes, initializers and shapes, and to append the backward nodes. Protobuf
// merges repeated fields, so the new graph is the old graph's bytes, untouched, with the new nodes, initializers,
// input and outputs written after them.
#include "gradgraph.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace GradGraph {

// ---------- protobuf, the wire format ----------
struct Reader {
    const uint8_t *p, *end;
    bool bad = false;
    Reader(std::string_view s) : p((const uint8_t*)s.data()), end((const uint8_t*)s.data() + s.size()) {}
    bool more() const { return !bad && p < end; }
    uint64_t varint() {
        uint64_t v = 0;
        for (int sh = 0; sh < 64; sh += 7) {
            if (p >= end) { bad = true; return 0; }
            const uint8_t b = *p++;
            v |= (uint64_t)(b & 0x7f) << sh;
            if (!(b & 0x80)) return v;
        }
        bad = true;
        return v;
    }
    bool next(uint32_t& field, uint32_t& wire) {
        if (!more()) return false;
        const uint64_t t = varint();
        field = (uint32_t)(t >> 3); wire = (uint32_t)(t & 7);
        return !bad;
    }
    std::string_view bytes() {
        const uint64_t n = varint();
        if (bad || n > (uint64_t)(end - p)) { bad = true; return {}; }
        std::string_view s((const char*)p, (size_t)n);
        p += n;
        return s;
    }
    float f32() { float v = 0; if (end - p < 4) { bad = true; return 0; } memcpy(&v, p, 4); p += 4; return v; }
    void skip(uint32_t wire) {
        if (wire == 0) varint();
        else if (wire == 1) { if (end - p < 8) bad = true; else p += 8; }
        else if (wire == 2) bytes();
        else if (wire == 5) { if (end - p < 4) bad = true; else p += 4; }
        else bad = true;
    }
};
// a repeated int64 field, packed or not
static void readInts(Reader& r, uint32_t wire, std::vector<int64_t>& out) {
    if (wire == 2) { Reader q(r.bytes()); while (q.more()) out.push_back((int64_t)q.varint()); }
    else out.push_back((int64_t)r.varint());
}

struct Writer {
    std::string out;
    void varint(uint64_t v) { while (v >= 0x80) { out += (char)((v & 0x7f) | 0x80); v >>= 7; } out += (char)v; }
    void tag(uint32_t f, uint32_t w) { varint((uint64_t)f << 3 | w); }
    void str(uint32_t f, std::string_view s) { tag(f, 2); varint(s.size()); out.append(s.data(), s.size()); }
    void i64(uint32_t f, int64_t v) { tag(f, 0); varint((uint64_t)v); }
    void f32(uint32_t f, float v) { tag(f, 5); char b[4]; memcpy(b, &v, 4); out.append(b, 4); }
    void msg(uint32_t f, const Writer& w) { str(f, w.out); }
};

// ---------- ONNX, the parts read ----------
enum { ATTR_FLOAT = 1, ATTR_INT = 2, ATTR_INTS = 7 };
struct Attr { std::string name; int type = 0; float f = 0; int64_t i = 0; std::vector<int64_t> ints; };
struct Node {
    std::string op, name;
    std::vector<std::string> in, out;
    std::vector<Attr> attrs;
    const Attr* attr(const char* n) const { for (const auto& a : attrs) if (a.name == n) return &a; return nullptr; }
    int64_t geti(const char* n, int64_t d) const { const Attr* a = attr(n); return a ? a->i : d; }
    float getf(const char* n, float d) const { const Attr* a = attr(n); return a ? a->f : d; }
    std::vector<int64_t> getints(const char* n, std::vector<int64_t> d) const { const Attr* a = attr(n); return a && !a->ints.empty() ? a->ints : d; }
};
struct Init { std::vector<int64_t> dims; int type = 0; std::string_view raw; std::vector<float> floats; };

static Attr parseAttr(std::string_view s) {
    Attr a; Reader r(s); uint32_t f, w;
    while (r.next(f, w)) {
        if (f == 1 && w == 2) a.name = std::string(r.bytes());
        else if (f == 2 && w == 5) a.f = r.f32();
        else if (f == 3 && w == 0) a.i = (int64_t)r.varint();
        else if (f == 8) readInts(r, w, a.ints);
        else if (f == 20 && w == 0) a.type = (int)r.varint();
        else r.skip(w);
    }
    return a;
}
static Node parseNode(std::string_view s) {
    Node n; Reader r(s); uint32_t f, w;
    while (r.next(f, w)) {
        if (f == 1 && w == 2) n.in.emplace_back(r.bytes());
        else if (f == 2 && w == 2) n.out.emplace_back(r.bytes());
        else if (f == 3 && w == 2) n.name = std::string(r.bytes());
        else if (f == 4 && w == 2) n.op = std::string(r.bytes());
        else if (f == 5 && w == 2) n.attrs.push_back(parseAttr(r.bytes()));
        else r.skip(w);
    }
    return n;
}
static std::pair<std::string, Init> parseInit(std::string_view s) {
    std::string name; Init t; Reader r(s); uint32_t f, w;
    while (r.next(f, w)) {
        if (f == 1) readInts(r, w, t.dims);
        else if (f == 2 && w == 0) t.type = (int)r.varint();
        else if (f == 8 && w == 2) name = std::string(r.bytes());
        else if (f == 9 && w == 2) t.raw = r.bytes();
        else if (f == 4 && w == 2) { Reader q(r.bytes()); while (q.more()) t.floats.push_back(q.f32()); }
        else if (f == 4 && w == 5) t.floats.push_back(r.f32());
        else r.skip(w);
    }
    return {name, t};
}
static std::vector<float> floatsOf(const Init& t) {
    if (!t.floats.empty()) return t.floats;
    std::vector<float> v(t.raw.size() / 4);
    if (!v.empty()) memcpy(v.data(), t.raw.data(), v.size() * 4);
    return v;
}
// a value's name and its shape (unknown dims: 1)
static std::pair<std::string, std::vector<int64_t>> parseValueInfo(std::string_view s) {
    std::string name; std::vector<int64_t> shape; Reader r(s); uint32_t f, w;
    while (r.next(f, w)) {
        if (f == 1 && w == 2) name = std::string(r.bytes());
        else if (f == 2 && w == 2) {
            Reader ty(r.bytes()); uint32_t f2, w2;
            while (ty.next(f2, w2)) {
                if (f2 == 1 && w2 == 2) {   // tensor_type
                    Reader tt(ty.bytes()); uint32_t f3, w3;
                    while (tt.next(f3, w3)) {
                        if (f3 == 2 && w3 == 2) {   // shape
                            Reader sh(tt.bytes()); uint32_t f4, w4;
                            while (sh.next(f4, w4)) {
                                if (f4 == 1 && w4 == 2) {   // dim
                                    Reader d(sh.bytes()); uint32_t f5, w5; int64_t v = 1;
                                    while (d.next(f5, w5)) { if (f5 == 1 && w5 == 0) v = (int64_t)d.varint(); else d.skip(w5); }
                                    shape.push_back(v > 0 ? v : 1);
                                } else sh.skip(w4);
                            }
                        } else tt.skip(w3);
                    }
                } else ty.skip(w2);
            }
        } else r.skip(w);
    }
    return {name, shape};
}

// ---------- writing the new parts ----------
struct Out {
    Writer graphTail;   // nodes, initializers, inputs, outputs, appended to the graph
    int count = 0;
    std::string name(const char* base) { return "inoc" + std::to_string(++count) + "_" + base; }
    void node(const char* op, std::vector<std::string> in, std::vector<std::string> out, const std::vector<Attr>& attrs = {}) {
        Writer n;
        for (const auto& s : in) n.str(1, s);
        for (const auto& s : out) n.str(2, s);
        n.str(3, out[0] + "_node");
        n.str(4, op);
        for (const auto& a : attrs) {
            Writer w;
            w.str(1, a.name);
            if (a.type == ATTR_FLOAT) w.f32(2, a.f);
            else if (a.type == ATTR_INT) w.i64(3, a.i);
            else for (int64_t v : a.ints) w.i64(8, v);
            w.i64(20, a.type);
            n.msg(5, w);
        }
        graphTail.msg(1, n);
    }
    std::string floats(const std::vector<int64_t>& dims, const std::vector<float>& v, const char* base) {
        const std::string nm = name(base);
        Writer t;
        for (int64_t d : dims) t.i64(1, d);
        t.i64(2, 1);   // FLOAT
        t.str(8, nm);
        t.str(9, std::string_view((const char*)v.data(), v.size() * 4));
        graphTail.msg(5, t);
        return nm;
    }
    std::string ints(const std::vector<int64_t>& v, const char* base) {
        const std::string nm = name(base);
        Writer t;
        t.i64(1, (int64_t)v.size());
        t.i64(2, 7);   // INT64
        t.str(8, nm);
        t.str(9, std::string_view((const char*)v.data(), v.size() * 8));
        graphTail.msg(5, t);
        return nm;
    }
    void valueInfo(uint32_t field, const std::string& nm, const std::vector<int64_t>& dims, const std::vector<std::string>& params) {
        Writer shape;
        for (size_t i = 0; i < dims.size(); ++i) {
            Writer d;
            if (i < params.size() && !params[i].empty()) d.str(2, params[i]); else d.i64(1, dims[i]);
            shape.msg(1, d);
        }
        Writer tensor; tensor.i64(1, 1); tensor.msg(2, shape);
        Writer type; type.msg(1, tensor);
        Writer vi; vi.str(1, nm); vi.msg(2, type);
        graphTail.msg(field, vi);
    }
};
static Attr A(const char* n, int64_t i) { Attr a; a.name = n; a.type = ATTR_INT; a.i = i; return a; }
static Attr A(const char* n, std::vector<int64_t> v) { Attr a; a.name = n; a.type = ATTR_INTS; a.ints = std::move(v); return a; }

bool build(const std::string& src, const std::string& dst, std::string& err) {
    std::string bytes;
    {
        std::ifstream f(fs::u8path(src), std::ios::binary);
        if (!f) { err = "Can't read " + src; return false; }
        bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    // the model: the graph's place, the opset
    Reader mr(bytes);
    uint32_t f, w;
    std::string_view graph;
    const char* graphStart = nullptr; const char* graphEnd = nullptr;
    int64_t opset = 11;
    while (true) {
        const uint8_t* at = mr.p;
        if (!mr.next(f, w)) break;
        if (f == 7 && w == 2) { graph = mr.bytes(); graphStart = (const char*)at; graphEnd = (const char*)mr.p; }
        else if (f == 8 && w == 2) {
            Reader o(mr.bytes()); uint32_t f2, w2; std::string dom; int64_t v = 0;
            while (o.next(f2, w2)) { if (f2 == 1 && w2 == 2) dom = std::string(o.bytes()); else if (f2 == 2 && w2 == 0) v = (int64_t)o.varint(); else o.skip(w2); }
            if (dom.empty() || dom == "ai.onnx") opset = v;
        } else mr.skip(w);
    }
    if (mr.bad || !graphStart) { err = src + " isn't an ONNX model Inoculant can read"; return false; }

    // the graph
    std::vector<Node> nodes;
    std::map<std::string, Init> inits;
    std::vector<std::pair<std::string, std::vector<int64_t>>> inputs, outputs;
    Reader gr(graph);
    while (gr.next(f, w)) {
        if (f == 1 && w == 2) nodes.push_back(parseNode(gr.bytes()));
        else if (f == 5 && w == 2) inits.insert(parseInit(gr.bytes()));
        else if (f == 11 && w == 2) inputs.push_back(parseValueInfo(gr.bytes()));
        else if (f == 12 && w == 2) outputs.push_back(parseValueInfo(gr.bytes()));
        else gr.skip(w);
    }
    if (gr.bad) { err = "Couldn't read the graph in " + src; return false; }
    std::string xName; std::vector<int64_t> xShape;
    for (const auto& i : inputs) if (!inits.count(i.first)) { xName = i.first; xShape = i.second; break; }
    if (xName.empty() || xShape.size() != 4 || outputs.empty()) { err = src + ": expected one picture input and an embedding output"; return false; }
    xShape[0] = 1;
    const std::string eName = outputs[0].first;

    // every activation's shape (batch 1), for ConvTranspose's output padding and Flatten's way back
    std::map<std::string, std::vector<int64_t>> shapes;
    shapes[xName] = xShape;
    for (const auto& n : nodes) {
        if (n.in.empty() || n.out.empty() || !shapes.count(n.in[0])) continue;
        const auto& s = shapes[n.in[0]];
        std::vector<int64_t> o = s;
        if (n.op == "Conv" && inits.count(n.in[1]) && s.size() == 4) {
            const auto& W = inits[n.in[1]].dims;
            const auto st = n.getints("strides", {1, 1}), pd = n.getints("pads", {0, 0, 0, 0}), dl = n.getints("dilations", {1, 1});
            o = {s[0], W[0], 0, 0};
            for (int i = 0; i < 2; ++i) o[2 + i] = (s[2 + i] + pd[i] + pd[i + 2] - dl[i] * (W[2 + i] - 1) - 1) / st[i] + 1;
        } else if (n.op == "Flatten") {
            const int64_t ax = n.geti("axis", 1); int64_t a = 1, b = 1;
            for (size_t i = 0; i < s.size(); ++i) ((int64_t)i < ax ? a : b) *= s[i];
            o = {a, b};
        } else if (n.op == "Gemm" && inits.count(n.in[1])) {
            const auto& B = inits[n.in[1]].dims;
            o = {s[0], n.geti("transB", 0) ? B[0] : B[1]};
        } else if (n.op == "MatMul" && inits.count(n.in[1])) {
            o = s; o.back() = inits[n.in[1]].dims.back();
        } else if (n.op == "Reshape" && inits.count(n.in[1])) {
            std::vector<int64_t> t; const Init& R = inits[n.in[1]];
            std::vector<int64_t> v(R.raw.size() / 8); if (!v.empty()) memcpy(v.data(), R.raw.data(), v.size() * 8);
            int64_t total = 1, known = 1; for (auto d : s) total *= d;
            for (size_t i = 0; i < v.size(); ++i) { int64_t d = v[i] == 0 && i < s.size() ? s[i] : v[i]; t.push_back(d); if (d > 0) known *= d; }
            for (auto& d : t) if (d < 0) d = total / std::max<int64_t>(1, known);
            o = t;
        }
        shapes[n.out[0]] = o;
    }

    Out out;
    // the loss: cos = <e/|e|, t>; d cos / d e = (t - cos e^) / |e|
    auto reduce = [&](const char* op, const std::string& in, const std::string& res, int sinceInput) {
        if (opset >= sinceInput) out.node(op, {in, out.ints({1}, "axes")}, {res}, {A("keepdims", 1)});
        else out.node(op, {in}, {res}, {A("axes", std::vector<int64_t>{1}), A("keepdims", 1)});
    };
    const std::string norm = out.name("norm"), eh = out.name("eh"), prod = out.name("prod"), ce = out.name("ce"), diff = out.name("diff"), ge = out.name("ge");
    reduce("ReduceL2", eName, norm, 18);
    out.node("Div", {eName, norm}, {eh});
    out.node("Mul", {eh, "t"}, {prod});
    reduce("ReduceSum", prod, "cos", 13);
    out.node("Mul", {"cos", eh}, {ce});
    out.node("Sub", {"t", ce}, {diff});
    out.node("Div", {diff, norm}, {ge});

    std::map<std::string, std::vector<std::string>> grads;   // an activation's gradient, in pieces
    grads[eName] = {ge};
    auto gradOf = [&](const std::string& nm) -> std::string {
        auto it = grads.find(nm);
        if (it == grads.end() || it->second.empty()) return "";
        if (it->second.size() > 1) { const std::string s = out.name("sum"); out.node("Sum", it->second, {s}); it->second = {s}; }
        return it->second[0];
    };
    auto add = [&](const std::string& nm, const std::string& g) { if (!inits.count(nm)) grads[nm].push_back(g); };
    std::string zero;
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        const Node& n = *it;
        if (n.out.empty()) continue;
        const std::string gy = gradOf(n.out[0]);
        if (gy.empty()) continue;
        const std::string& op = n.op;
        if (op == "Add") {
            add(n.in[0], gy); add(n.in[1], gy);
        } else if (op == "Identity") {
            add(n.in[0], gy);
        } else if (op == "BatchNormalization") {
            if (n.in.size() < 5 || !inits.count(n.in[1]) || !inits.count(n.in[4])) { err = "BatchNormalization without its numbers"; return false; }
            const auto sc = floatsOf(inits[n.in[1]]), var = floatsOf(inits[n.in[4]]);
            const float eps = n.getf("epsilon", 1e-5f);
            std::vector<float> k(sc.size());
            for (size_t i = 0; i < k.size(); ++i) k[i] = sc[i] / std::sqrt(var[i] + eps);
            const size_t rank = shapes.count(n.in[0]) ? shapes[n.in[0]].size() : 4;
            std::vector<int64_t> dims = {1, (int64_t)k.size()};
            while (dims.size() < rank) dims.push_back(1);
            const std::string o = out.name("bn");
            out.node("Mul", {gy, out.floats(dims, k, "bnk")}, {o});
            add(n.in[0], o);
        } else if (op == "PRelu" || op == "Relu") {
            if (zero.empty()) zero = out.floats({1}, {0.0f}, "zero");
            const std::string pos = out.name("pos"), o = out.name("act");
            out.node("Greater", {n.in[0], zero}, {pos});
            std::string other = zero;
            if (op == "PRelu") { other = out.name("neg"); out.node("Mul", {gy, n.in[1]}, {other}); }
            else { other = out.name("zeros"); out.node("Mul", {gy, zero}, {other}); }
            out.node("Where", {pos, gy, other}, {o});
            add(n.in[0], o);
        } else if (op == "Conv") {
            if (!inits.count(n.in[1]) || !shapes.count(n.in[0]) || !shapes.count(n.out[0])) { err = "a Conv Inoculant can't follow"; return false; }
            const auto& W = inits[n.in[1]].dims;
            const auto st = n.getints("strides", {1, 1}), pd = n.getints("pads", {0, 0, 0, 0}), dl = n.getints("dilations", {1, 1});
            const auto &is = shapes[n.in[0]], &os = shapes[n.out[0]];
            std::vector<int64_t> opad(2);
            for (int i = 0; i < 2; ++i) opad[i] = is[2 + i] - ((os[2 + i] - 1) * st[i] - pd[i] - pd[i + 2] + (W[2 + i] - 1) * dl[i] + 1);
            const std::string o = out.name("convT");
            out.node("ConvTranspose", {gy, n.in[1]}, {o}, {A("strides", st), A("pads", pd), A("dilations", dl), A("group", n.geti("group", 1)),
                                                       A("output_padding", opad), A("kernel_shape", std::vector<int64_t>{W[2], W[3]})});
            add(n.in[0], o);
        } else if (op == "Flatten" || op == "Reshape") {
            const std::string sh = out.name("shape"), o = out.name("unflat");
            out.node("Shape", {n.in[0]}, {sh});
            out.node("Reshape", {gy, sh}, {o});
            add(n.in[0], o);
        } else if (op == "Gemm" || op == "MatMul") {
            if (!inits.count(n.in[1])) { err = op + " between two activations: not handled"; return false; }
            if (op == "Gemm" && n.geti("transA", 0)) { err = "Gemm with transA: not handled"; return false; }
            const Init& B = inits[n.in[1]];
            const bool transB = op == "Gemm" && n.geti("transB", 0);
            const float alpha = op == "Gemm" ? n.getf("alpha", 1.0f) : 1.0f;
            std::string M = n.in[1];
            if (!transB || alpha != 1.0f) {   // dA = alpha gY op(B)^T: B itself when transB, else B^T
                const auto v = floatsOf(B);
                const int64_t r = B.dims[0], c = B.dims[1];
                std::vector<float> m(v.size());
                if (transB) for (size_t i = 0; i < v.size(); ++i) m[i] = v[i] * alpha;
                else for (int64_t i = 0; i < r; ++i) for (int64_t j = 0; j < c; ++j) m[(size_t)j * r + i] = v[(size_t)i * c + j] * alpha;
                M = out.floats(transB ? std::vector<int64_t>{r, c} : std::vector<int64_t>{c, r}, m, "gemmB");
            }
            const std::string o = out.name("gemm");
            out.node("MatMul", {gy, M}, {o});
            add(n.in[0], o);
        } else {
            err = "The model uses " + op + ", which Inoculant can't run backwards yet";
            return false;
        }
    }
    const std::string gx = gradOf(xName);
    if (gx.empty()) { err = "The picture input doesn't reach the embedding"; return false; }
    out.node("Identity", {gx}, {"grad"});
    const int64_t D = shapes.count(eName) && shapes[eName].size() == 2 ? shapes[eName][1] : 512;
    out.valueInfo(11, "t", {1, D}, {});
    out.valueInfo(12, "cos", {1, 1}, {"N", ""});
    out.valueInfo(12, "grad", xShape, {"N", "", "", ""});

    // the model again, the graph field replaced by the old graph + the new tail
    Writer g; g.out.reserve(graph.size() + out.graphTail.out.size() + 16);
    g.out.append(graph.data(), graph.size());
    g.out += out.graphTail.out;
    Writer model;
    model.out.append(bytes.data(), graphStart - bytes.data());
    model.str(7, g.out);
    model.out.append(graphEnd, bytes.data() + bytes.size() - graphEnd);
    std::error_code ec;
    fs::create_directories(fs::u8path(dst).parent_path(), ec);
    const std::string tmp = dst + ".part";
    {
        std::ofstream o(fs::u8path(tmp), std::ios::binary);
        if (!o) { err = "Can't write " + dst; return false; }
        o.write(model.out.data(), (std::streamsize)model.out.size());
        if (!o) { err = "Can't write " + dst; return false; }
    }
    fs::rename(fs::u8path(tmp), fs::u8path(dst), ec);
    if (ec) { err = "Can't write " + dst + ": " + ec.message(); return false; }
    return true;
}

}  // namespace GradGraph
