// Inoculant - the identity cloak (see cloak.h).
//
// One step, for each face:
//   1. B views of the face as the recogniser sees it: the 112 x 112 ArcFace crop of the picture-so-far, the first
//      exactly aligned, the others jittered (a small random shift, turn, scale, and sometimes a blur);
//   2. the gradient graph gives d similarity / d crop pixel for all of them in one run;
//   3. each view's gradient is carried back through its warp (the exact adjoint of the bilinear sampling) onto the
//      picture's own pixels, and summed;
//   4. MI-FGSM: the sum, normalised, joins a running momentum; the picture moves one step against (Hide) or toward
//      (Decoy) the target by the momentum's sign, each pixel's step scaled by its budget, and the change is clipped
//      to the budget (strength x the texture and coverage maps).
#include "cloak.h"
#include "gradgraph.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

namespace Cloak {

static const char* RECOGNISER = "arcface_w600k_r50.onnx";
static const char* GRADFILE = "w600k_r50_grad.onnx";
static const int N = 112;

const char* goalName(int g) {
    static const char* n[] = {"Hide (away from you)", "Decoy (toward another face)"};
    return g >= 0 && g < G_COUNT ? n[g] : "?";
}

DetectSettings detectSettings() {
    DetectSettings st;
    st.model = DET_RETINAFACE;
    st.size = 640;
    st.score = 0.5f;
    for (int& m : st.margin) m = 25;   // room round the frame: close-ups that fill the picture are found too
    return st;
}

std::string gradModelPath() {
    const std::string have = Ort::findModel(GRADFILE);
    if (!have.empty()) return have;
    return Ort::modelStore("face") + "/" + GRADFILE;
}

bool ready(std::string& err, Ort::Say say) {
    std::error_code ec;
    std::string grad = Ort::findModel(GRADFILE);
    if (grad.empty()) {
        const std::string src = Ort::findModel(RECOGNISER, say);
        if (src.empty()) { err = "Can't find or download ArcFace (w600k_r50)"; return false; }
        grad = Ort::modelStore("face") + "/" + GRADFILE;
        if (say) say("Making ArcFace's gradient graph, once...");
        if (!GradGraph::build(src, grad, err)) return false;
    }
    if (!Ort::open(RECOGNISER, err, say)) return false;
    return Ort::open(GRADFILE, err, say) != nullptr;
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0;
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return (float)(d / std::sqrt(std::max(1e-12, na * nb)));
}

bool faces(const Img& im, std::vector<Face>& out, std::string& err) {
    return Faces::analyse(im, detectSettings(), out, err, false);
}

int sameFace(const Face& f, const std::vector<Face>& b) {
    int best = -1; float bestD = 1e30f;
    const float cx = (f.box[0] + f.box[2]) / 2, cy = (f.box[1] + f.box[3]) / 2, size = std::max(8.0f, f.box[2] - f.box[0]);
    for (size_t i = 0; i < b.size(); ++i) {
        const float x = (b[i].box[0] + b[i].box[2]) / 2, y = (b[i].box[1] + b[i].box[3]) / 2;
        const float d = std::hypot(x - cx, y - cy);
        if (d < size * 0.5f && d < bestD) { bestD = d; best = (int)i; }
    }
    return best;
}

std::vector<float> compare(const std::vector<Face>& a, const std::vector<Face>& b) {
    std::vector<float> r;
    for (const auto& f : a) { const int j = sameFace(f, b); r.push_back(j < 0 ? -1.0f : cosine(f.embNorm, b[j].embNorm)); }
    return r;
}

// ---------- warping a float picture, and carrying a gradient back through it ----------
struct FImg { int w = 0, h = 0; std::vector<float> px; };

// a: after b (a(b(p)))
static Affine compose(const Affine& a, const Affine& b) {
    Affine r;
    r.m[0] = a.m[0] * b.m[0] + a.m[1] * b.m[3]; r.m[1] = a.m[0] * b.m[1] + a.m[1] * b.m[4]; r.m[2] = a.m[0] * b.m[2] + a.m[1] * b.m[5] + a.m[2];
    r.m[3] = a.m[3] * b.m[0] + a.m[4] * b.m[3]; r.m[4] = a.m[3] * b.m[1] + a.m[4] * b.m[4]; r.m[5] = a.m[3] * b.m[2] + a.m[4] * b.m[5] + a.m[5];
    return r;
}

// the crop M (picture -> crop) takes: n x n samples a crop pixel (n by how much it shrinks), bilinear, edges repeated
static int samplesPerSide(const Affine& M) { return std::clamp((int)std::ceil(1.0 / std::max(1e-9, M.scale()) - 0.01), 1, 4); }

static void warp(const FImg& im, const Affine& M, float* out /* N*N*3 */) {
    const Affine inv = M.inverse();
    const int n = samplesPerSide(M);
    parallelRows(0, N, [&](int y) {
        for (int x = 0; x < N; ++x) {
            float acc[3] = {0, 0, 0};
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    double px, py;
                    inv.apply(x + (i + 0.5) / n - 0.5, y + (j + 0.5) / n - 0.5, px, py);
                    px = std::clamp(px, 0.0, im.w - 1.0); py = std::clamp(py, 0.0, im.h - 1.0);
                    const int x0 = (int)px, y0 = (int)py, x1 = std::min(im.w - 1, x0 + 1), y1 = std::min(im.h - 1, y0 + 1);
                    const float fx = (float)(px - x0), fy = (float)(py - y0);
                    const float* a = &im.px[((size_t)y0 * im.w + x0) * 3]; const float* b = &im.px[((size_t)y0 * im.w + x1) * 3];
                    const float* c = &im.px[((size_t)y1 * im.w + x0) * 3]; const float* d = &im.px[((size_t)y1 * im.w + x1) * 3];
                    for (int k = 0; k < 3; ++k) acc[k] += (a[k] * (1 - fx) + b[k] * fx) * (1 - fy) + (c[k] * (1 - fx) + d[k] * fx) * fy;
                }
            float* o = &out[((size_t)y * N + x) * 3];
            for (int k = 0; k < 3; ++k) o[k] = acc[k] / (n * n);
        }
    });
}

// the adjoint of warp: each crop pixel's gradient spread back onto the four picture pixels each sample read, into
// g (the region rx, ry, rw x rh of the picture; samples outside it are dropped)
static void unwarp(const float* gcrop, const Affine& M, int imW, int imH, int rx, int ry, int rw, int rh, std::vector<float>& g) {
    const Affine inv = M.inverse();
    const int n = samplesPerSide(M);
    const float share = 1.0f / (n * n);
    auto put = [&](int x, int y, float wgt, const float* v) {
        x -= rx; y -= ry;
        if (x < 0 || y < 0 || x >= rw || y >= rh || wgt == 0) return;
        float* p = &g[((size_t)y * rw + x) * 3];
        p[0] += v[0] * wgt; p[1] += v[1] * wgt; p[2] += v[2] * wgt;
    };
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const float* v = &gcrop[((size_t)y * N + x) * 3];
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    double px, py;
                    inv.apply(x + (i + 0.5) / n - 0.5, y + (j + 0.5) / n - 0.5, px, py);
                    px = std::clamp(px, 0.0, imW - 1.0); py = std::clamp(py, 0.0, imH - 1.0);
                    const int x0 = (int)px, y0 = (int)py, x1 = std::min(imW - 1, x0 + 1), y1 = std::min(imH - 1, y0 + 1);
                    const float fx = (float)(px - x0), fy = (float)(py - y0);
                    put(x0, y0, share * (1 - fx) * (1 - fy), v); put(x1, y0, share * fx * (1 - fy), v);
                    put(x0, y1, share * (1 - fx) * fy, v); put(x1, y1, share * fx * fy, v);
                }
        }
}

// a Gaussian blur of an interleaved RGB float buffer (the same blur is its own adjoint, near enough)
static void blurRGB(std::vector<float>& px, int w, int h, float sigma) {
    if (sigma < 0.3f) return;
    std::vector<float> ch((size_t)w * h);
    for (int k = 0; k < 3; ++k) {
        for (size_t i = 0; i < ch.size(); ++i) ch[i] = px[i * 3 + k];
        gaussianBlur(ch, w, h, sigma);
        for (size_t i = 0; i < ch.size(); ++i) px[i * 3 + k] = ch[i];
    }
}

// the coverage mask in crop space: an ellipse over the inner face, growing to the whole crop at coverage 1
static float coverageAt(double cx, double cy, float coverage) {
    if (cx < 0 || cy < 0 || cx >= N || cy >= N) return 0.0f;
    if (coverage >= 0.999f) return 1.0f;
    const double rx = 30 + 40 * coverage, ry = 40 + 40 * coverage;
    const double d = std::hypot((cx - 56) / rx, (cy - 62) / ry);
    return (float)std::clamp((1.15 - d) / 0.3, 0.0, 1.0);
}

bool protect(const Img& clean, const Settings& st, const std::vector<float>& decoy, Result& res, std::string& err, Progress progress) {
    const auto t0 = std::chrono::steady_clock::now();
    res = Result();
    if (!ready(err)) return false;
    OrtSession* gs = Ort::open(GRADFILE, err);
    if (!gs) return false;
    if (st.goal == G_DECOY && decoy.size() != 512) { err = "Decoy: choose a picture of the decoy face first"; return false; }
    if (progress) progress(0.0f, "Finding the faces...");
    std::vector<Face> found;
    if (!faces(clean, found, err)) return false;
    if (found.empty()) { err = "No face found in this picture"; return false; }

    const int W = clean.w, H = clean.h;
    // each face's alignment, and the region of the picture the cloak may touch (every crop, with room for jitter)
    std::vector<Affine> align;
    double x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9, pxPerCrop = 0;
    for (const auto& f : found) {
        const Affine M = alignFace(f.kps, T_ARCFACE_112_V2, N);
        align.push_back(M);
        const Affine inv = M.inverse();
        const double m = N * 0.2;
        const double cs[4][2] = {{-m, -m}, {N + m, -m}, {N + m, N + m}, {-m, N + m}};
        for (const auto& c : cs) { double x, y; inv.apply(c[0], c[1], x, y); x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y); }
        pxPerCrop += 1.0 / std::max(1e-9, M.scale());
    }
    pxPerCrop /= found.size();
    const int rx = std::clamp((int)std::floor(x0), 0, W - 1), ry = std::clamp((int)std::floor(y0), 0, H - 1);
    const int rw = std::clamp((int)std::ceil(x1), rx + 1, W) - rx, rh = std::clamp((int)std::ceil(y1), ry + 1, H) - ry;
    const size_t RN = (size_t)rw * rh;

    // the budget map: strength x coverage (the faces' areas) x texture (more where there's detail to hide it in)
    std::vector<float> budget(RN, 0.0f);
    {
        std::vector<double> sum((size_t)(rw + 1) * (rh + 1), 0.0), sum2(sum.size(), 0.0);
        for (int y = 0; y < rh; ++y)
            for (int x = 0; x < rw; ++x) {
                const uint8_t* p = &clean.px[((size_t)(y + ry) * W + x + rx) * 3];
                const double l = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
                const size_t i = (size_t)(y + 1) * (rw + 1) + x + 1;
                sum[i] = l + sum[i - 1] + sum[i - (rw + 1)] - sum[i - (rw + 1) - 1];
                sum2[i] = l * l + sum2[i - 1] + sum2[i - (rw + 1)] - sum2[i - (rw + 1) - 1];
            }
        const int r = std::max(1, (int)std::lround(pxPerCrop * 1.5));
        parallelRows(0, rh, [&](int y) {
            for (int x = 0; x < rw; ++x) {
                float cov = 0;
                for (const auto& M : align) { double cx, cy; M.apply(x + rx, y + ry, cx, cy); cov = std::max(cov, coverageAt(cx, cy, st.coverage)); }
                if (cov <= 0) continue;
                const int ax = std::max(0, x - r), ay = std::max(0, y - r), bx = std::min(rw, x + r + 1), by = std::min(rh, y + r + 1);
                const double n = (double)(bx - ax) * (by - ay);
                auto S = [&](const std::vector<double>& s) { return s[(size_t)by * (rw + 1) + bx] - s[(size_t)ay * (rw + 1) + bx] - s[(size_t)by * (rw + 1) + ax] + s[(size_t)ay * (rw + 1) + ax]; };
                const double mean = S(sum) / n, sd = std::sqrt(std::max(0.0, S(sum2) / n - mean * mean));
                const float tex = (float)std::clamp(sd / 12.0, 0.15, 1.0);
                budget[(size_t)y * rw + x] = st.strength * cov * (1.0f - st.texture * (1.0f - tex));
            }
        });
    }

    // the targets: Hide pushes away from each face's own embedding, Decoy pulls toward the decoy's
    std::vector<std::vector<float>> targets;
    for (const auto& f : found) targets.push_back(st.goal == G_DECOY ? decoy : f.embNorm);

    FImg cur; cur.w = W; cur.h = H; cur.px.resize((size_t)W * H * 3);
    for (size_t i = 0; i < cur.px.size(); ++i) cur.px[i] = clean.px[i];
    std::vector<float> delta(RN * 3, 0.0f), mom(RN * 3, 0.0f), g(RN * 3);
    std::mt19937 rng((uint32_t)st.seed * 7919u + 17u);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    const int B = std::clamp(st.samples, 1, 16);
    const float r = std::clamp(st.robustness, 0.0f, 1.0f);
    std::vector<float> crops((size_t)B * N * N * 3);
    std::vector<float> lastCos(found.size(), 1.0f);
    int step = 0;
    for (; step < st.steps; ++step) {
        std::fill(g.begin(), g.end(), 0.0f);
        for (size_t fi = 0; fi < found.size(); ++fi) {
            std::vector<Affine> Ms(B);
            std::vector<float> blurs(B, 0.0f);
            for (int b = 0; b < B; ++b) {
                Affine J;
                if (b > 0 && r > 0) {   // a jitter round the crop's centre
                    const double a = U(rng) * 6.0 * r * 3.14159265 / 180.0, s = 1.0 + U(rng) * 0.08 * r;
                    const double tx = U(rng) * 4.0 * r, ty = U(rng) * 4.0 * r;
                    const double ca = std::cos(a) * s, sa = std::sin(a) * s, c = N / 2.0;
                    J.m[0] = ca; J.m[1] = -sa; J.m[2] = c - ca * c + sa * c + tx;
                    J.m[3] = sa; J.m[4] = ca;  J.m[5] = c - sa * c - ca * c + ty;
                    if (U(rng) > 0) blurs[b] = (U(rng) * 0.5f + 0.5f) * 1.2f * r;
                }
                Ms[b] = compose(J, align[fi]);
                float* c = &crops[(size_t)b * N * N * 3];
                warp(cur, Ms[b], c);
                if (blurs[b] > 0) { std::vector<float> t(c, c + N * N * 3); blurRGB(t, N, N, blurs[b]); memcpy(c, t.data(), t.size() * sizeof(float)); }
            }
            // one view per run: DirectML refuses ArcFace's last BatchNormalization with a batch above 1
            Ort::Tensor x, t;
            for (const auto& nm : Ort::inputNames(gs)) if (nm != "t") x.name = nm;
            x.shape = {1, 3, N, N};
            x.f.resize((size_t)3 * N * N);
            t.name = "t"; t.shape = {1, 512}; t.f = targets[fi];
            // Hide: down the similarity; Decoy: up it
            const float dir = st.goal == G_HIDE ? -1.0f : 1.0f;
            std::vector<float> gc((size_t)N * N * 3);
            for (int b = 0; b < B; ++b) {
                for (int i = 0; i < N * N; ++i)
                    for (int k = 0; k < 3; ++k) x.f[(size_t)k * N * N + i] = crops[((size_t)b * N * N + i) * 3 + k] / 127.5f - 1.0f;
                std::vector<Ort::Tensor> ins = {x, t}, outs;
                if (!Ort::run(gs, ins, outs, err)) return false;
                const Ort::Tensor *cosT = nullptr, *gradT = nullptr;
                for (const auto& o : outs) { if (o.name == "cos") cosT = &o; else if (o.name == "grad") gradT = &o; }
                if (!cosT || !gradT || gradT->f.size() < x.f.size() || cosT->f.empty()) { err = "The gradient graph gave nothing"; return false; }
                if (b == 0) lastCos[fi] = cosT->f[0];
                for (int i = 0; i < N * N; ++i)
                    for (int k = 0; k < 3; ++k) gc[(size_t)i * 3 + k] = dir * gradT->f[(size_t)k * N * N + i];
                if (blurs[b] > 0) blurRGB(gc, N, N, blurs[b]);
                unwarp(gc.data(), Ms[b], W, H, rx, ry, rw, rh, g);
            }
        }
        if (st.goal == G_HIDE && st.stopAt > -1.0f) {
            bool all = true;
            for (float c : lastCos) if (c > st.stopAt) all = false;
            if (all && step > 0) break;
        }
        // the noise's grain, then MI-FGSM: momentum of the L1-normalised gradient, a signed step within the budget
        if (st.smoothness > 0.01f) blurRGB(g, rw, rh, st.smoothness * 2.5f * (float)pxPerCrop);
        double l1 = 0;
        for (float v : g) l1 += std::fabs(v);
        const float inv = l1 > 0 ? (float)(g.size() / l1) : 0.0f;
        for (size_t i = 0; i < g.size(); ++i) mom[i] = st.momentum * mom[i] + g[i] * inv;
        parallelRows(0, rh, [&](int y) {
            for (int x = 0; x < rw; ++x) {
                const size_t p = (size_t)y * rw + x;
                const float b = budget[p];
                if (b <= 0) continue;
                // colour: 0 moves R, G and B together (light and dark only), 1 each on its own
                const float grey = (mom[p * 3] + mom[p * 3 + 1] + mom[p * 3 + 2]) / 3.0f;
                for (int k = 0; k < 3; ++k) {
                    const size_t i = p * 3 + k;
                    const float m = st.colour * mom[i] + (1.0f - st.colour) * grey;
                    float d = delta[i] + (m > 0 ? 1.0f : m < 0 ? -1.0f : 0.0f) * st.stepSize * (b / std::max(0.01f, st.strength));
                    d = std::clamp(d, -b, b);
                    const float c = clean.px[((size_t)(y + ry) * W + x + rx) * 3 + k];
                    d = std::clamp(c + d, 0.0f, 255.0f) - c;
                    delta[i] = d;
                    cur.px[((size_t)(y + ry) * W + x + rx) * 3 + k] = c + d;
                }
            }
        });
        if (progress) {
            float worst = -1;
            for (float c : lastCos) worst = std::max(worst, c);
            char msg[160];
            snprintf(msg, sizeof msg, st.goal == G_HIDE ? "Step %d / %d: similarity to you %.2f" : "Step %d / %d: similarity to the decoy %.2f", step + 1, st.steps, worst);
            if (!progress((float)(step + 1) / std::max(1, st.steps), msg)) { ++step; break; }
        }
    }
    res.steps = step;

    // the picture, to 8 bits, and what it did
    res.out = clean;
    double se = 0, sumAbs = 0, mx = 0;
    for (int y = 0; y < rh; ++y)
        for (int x = 0; x < rw; ++x)
            for (int k = 0; k < 3; ++k) {
                const size_t o = ((size_t)(y + ry) * W + x + rx) * 3 + k;
                const int v = (int)std::lround(std::clamp(clean.px[o] + delta[((size_t)y * rw + x) * 3 + k], 0.0f, 255.0f));
                res.out.px[o] = (uint8_t)v;
                const double d = (double)v - clean.px[o];
                se += d * d; sumAbs += std::fabs(d); mx = std::max(mx, std::fabs(d));
            }
    const double total = (double)W * H * 3;
    res.psnr = se > 0 ? 10.0 * std::log10(255.0 * 255.0 / (se / total)) : 99.0;
    res.maxDiff = mx;
    res.meanDiff = sumAbs / total;
    if (progress) progress(1.0f, "Checking: finding the faces again...");
    std::vector<Face> again;
    if (!faces(res.out, again, err)) return false;
    for (const auto& f : found) {
        FaceReport fr;
        memcpy(fr.box, f.box, sizeof fr.box);
        const int j = sameFace(f, again);
        fr.found = j >= 0;
        fr.after = j >= 0 ? cosine(f.embNorm, again[j].embNorm) : -1.0f;
        if (st.goal == G_DECOY && j >= 0) fr.decoy = cosine(decoy, again[j].embNorm);
        res.faces.push_back(fr);
    }
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace Cloak
