// Inoculant - the antidotes (see antidote.h).
#include "antidote.h"
#include "cloak.h"
#include "picture.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace Antidote {

const char* filterName(int f) {
    static const char* n[] = {"JPEG", "Blur", "Median", "Bit depth", "Shrink and enlarge", "Noise"};
    return f >= 0 && f < F_COUNT ? n[f] : "?";
}
const char* filterHint(int f) {
    static const char* n[] = {
        "Save as a JPEG at this quality and load it again: what most sites do to an upload.",
        "A Gaussian blur of this radius: softens the fine detail a cloak lives in.",
        "A median filter (ART's spatial smoothing): each pixel the middle value of its neighbours.",
        "Fewer levels per colour (feature squeezing): small changes round away.",
        "Shrink to this size and enlarge back: what a thumbnail or a resize does.",
        "Add random noise of this strength (Gaussian augmentation), to drown the cloak's pattern."};
    return f >= 0 && f < F_COUNT ? n[f] : "";
}

static Img fromFloat(const std::vector<float>& px, int w, int h) {
    Img o(w, h);
    for (size_t i = 0; i < o.px.size(); ++i) o.px[i] = (uint8_t)std::clamp(px[i] + 0.5f, 0.0f, 255.0f);
    return o;
}

static Img blur(const Img& im, float sigma) {
    std::vector<float> ch((size_t)im.w * im.h);
    std::vector<float> all(im.px.size());
    for (int k = 0; k < 3; ++k) {
        for (size_t i = 0; i < ch.size(); ++i) ch[i] = im.px[i * 3 + k];
        gaussianBlur(ch, im.w, im.h, sigma);
        for (size_t i = 0; i < ch.size(); ++i) all[i * 3 + k] = ch[i];
    }
    return fromFloat(all, im.w, im.h);
}

static Img median(const Img& im, int r) {
    if (r < 1) return im;
    Img o(im.w, im.h);
    parallelRows(0, im.h, [&](int y) {
        std::vector<uint8_t> v;
        v.reserve((size_t)(2 * r + 1) * (2 * r + 1));
        for (int x = 0; x < im.w; ++x)
            for (int k = 0; k < 3; ++k) {
                v.clear();
                for (int j = -r; j <= r; ++j)
                    for (int i = -r; i <= r; ++i)
                        v.push_back(im.px[((size_t)std::clamp(y + j, 0, im.h - 1) * im.w + std::clamp(x + i, 0, im.w - 1)) * 3 + k]);
                std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
                o.px[((size_t)y * im.w + x) * 3 + k] = v[v.size() / 2];
            }
    });
    return o;
}

static Img resizeTo(const Img& im, int w, int h) {
    Affine M; M.m[0] = (double)w / im.w; M.m[4] = (double)h / im.h;
    M.m[2] = 0.5 * M.m[0] - 0.5; M.m[5] = 0.5 * M.m[4] - 0.5;   // pixel centres line up
    const Crop c = warpCrop(im, M, w, h);
    return fromFloat(c.px, w, h);
}

Img apply(const Img& im, int f, const Settings& st) {
    switch (f) {
        case F_JPEG: return jpegRoundTrip(im, std::clamp((int)std::lround(st.jpeg), 1, 100));
        case F_BLUR: return blur(im, st.blur);
        case F_MEDIAN: return median(im, std::clamp((int)std::lround(st.median), 0, 5));
        case F_BITS: {
            const int levels = (1 << std::clamp((int)std::lround(st.bits), 1, 8)) - 1;
            Img o = im;
            for (auto& v : o.px) v = (uint8_t)std::lround(std::round(v / 255.0 * levels) * 255.0 / levels);
            return o;
        }
        case F_RESIZE: {
            const double s = std::clamp(st.resize, 5.0f, 100.0f) / 100.0;
            if (s >= 0.999) return im;
            const Img small = resizeTo(im, std::max(1, (int)std::lround(im.w * s)), std::max(1, (int)std::lround(im.h * s)));
            return resizeTo(small, im.w, im.h);
        }
        case F_NOISE: {
            std::mt19937 rng((uint32_t)st.seed);
            std::normal_distribution<float> n(0.0f, st.noise);
            Img o = im;
            for (auto& v : o.px) v = (uint8_t)std::clamp(v + n(rng) + 0.5f, 0.0f, 255.0f);
            return o;
        }
        default: return im;
    }
}

Img purify(const Img& im, const Settings& st) {
    Img o = im;
    for (int f = 0; f < F_COUNT; ++f) if (st.on[f]) o = apply(o, f, st);
    return o;
}

// the largest face's embedding in a picture, near where ref is when given
static bool faceOf(const Img& im, const Face* ref, Face& out, std::string& err) {
    std::vector<Face> fs;
    if (!Cloak::faces(im, fs, err)) return false;
    if (fs.empty()) return false;
    if (!ref) { out = fs[0]; return true; }
    const int j = Cloak::sameFace(*ref, fs);
    if (j < 0) return false;
    out = fs[j];
    return true;
}

bool test(const Img& clean, const Img& prot, const Settings& st, Report& rep, std::string& err) {
    rep = Report();
    Face c;
    if (!faceOf(clean, nullptr, c, err)) { if (err.empty()) err = "No face found in the original"; return false; }
    Face p;
    if (faceOf(prot, &c, p, err)) rep.protectedSim = Cloak::cosine(c.embNorm, p.embNorm);
    else if (!err.empty()) return false;
    auto row = [&](const std::string& name, const Img& pp, const Img& cc) {
        Row r; r.name = name;
        Face f;
        if (faceOf(pp, &c, f, err)) r.purifiedSim = Cloak::cosine(c.embNorm, f.embNorm);
        if (faceOf(cc, &c, f, err)) r.baselineSim = Cloak::cosine(c.embNorm, f.embNorm);
        const float protection = 1.0f - rep.protectedSim;
        if (protection > 1e-3f && r.purifiedSim > -1 && r.baselineSim > -1) r.kept = std::max(0.0f, (r.baselineSim - r.purifiedSim) / protection);
        rep.rows.push_back(r);
    };
    int count = 0;
    for (int f = 0; f < F_COUNT; ++f) {
        if (!st.on[f]) continue;
        ++count;
        row(filterName(f), apply(prot, f, st), apply(clean, f, st));
    }
    if (count > 1) row("All of them, in order", purify(prot, st), purify(clean, st));
    err.clear();
    return true;
}

// ---------- the inspector ----------
static bool embedCrop(const Img& crop, std::vector<float>& e, std::string& err) {
    OrtSession* s = Ort::open("arcface_w600k_r50.onnx", err);
    if (!s) return false;
    Ort::Tensor in;
    in.name = Ort::inputNames(s)[0];
    in.shape = {1, 3, 112, 112};
    in.f.resize((size_t)3 * 112 * 112);
    for (int i = 0; i < 112 * 112; ++i) for (int k = 0; k < 3; ++k) in.f[(size_t)k * 112 * 112 + i] = crop.px[(size_t)i * 3 + k] / 127.5f - 1.0f;
    std::vector<Ort::Tensor> ins = {in}, outs;
    if (!Ort::run(s, ins, outs, err) || outs.empty() || outs[0].f.size() < 512) { if (err.empty()) err = "ArcFace gave nothing"; return false; }
    e.assign(outs[0].f.begin(), outs[0].f.begin() + 512);
    return true;
}

bool inspect(const Img& im, Inspection& out, std::string& err) {
    out = Inspection();
    std::vector<Face> fs;
    if (!Cloak::faces(im, fs, err)) return false;
    out.faces = (int)fs.size();
    if (fs.empty()) { out.verdict = "No face found: Inoculant's cloak works on faces, so there's nothing to check"; return true; }
    const Face& f = fs[0];
    // 1. feature squeezing on the face's own crop: the most its embedding moves
    const Affine M = alignFace(f.kps, T_ARCFACE_112_V2, 112);
    const Crop cc = warpCrop(im, M, 112, 112);
    Img crop(112, 112);
    for (size_t i = 0; i < crop.px.size(); ++i) crop.px[i] = (uint8_t)std::clamp(cc.px[i] + 0.5f, 0.0f, 255.0f);
    std::vector<float> e0;
    if (!embedCrop(crop, e0, err)) return false;
    Settings sq; sq.jpeg = 75; sq.median = 1; sq.bits = 5; sq.blur = 0.8f;
    for (int filt : {F_JPEG, F_MEDIAN, F_BITS, F_BLUR}) {
        std::vector<float> e;
        if (!embedCrop(apply(crop, filt, sq), e, err)) return false;
        out.squeeze = std::max(out.squeeze, 1.0f - Cloak::cosine(e0, e));
    }
    // 2. fine noise on the smooth parts of the face, at the scale the recogniser (and so a cloak) works at: the
    // face's crop minus its 3 x 3 median, where the median itself is flat (skin, not edges)
    const Img& region = crop;
    const Img med = median(region, 1);
    double se = 0; long n = 0;
    for (int y = 2; y < region.h - 2; ++y)
        for (int x = 2; x < region.w - 2; ++x) {
            // flatness of the median picture: its largest step to a neighbour 2 pixels away
            int step = 0;
            for (int k = 0; k < 3; ++k) {
                const int c = med.px[((size_t)y * region.w + x) * 3 + k];
                step = std::max({step, std::abs(c - med.px[((size_t)y * region.w + x + 2) * 3 + k]), std::abs(c - med.px[((size_t)(y + 2) * region.w + x) * 3 + k])});
            }
            if (step > 6) continue;
            for (int k = 0; k < 3; ++k) {
                const double d = (double)region.px[((size_t)y * region.w + x) * 3 + k] - med.px[((size_t)y * region.w + x) * 3 + k];
                se += d * d; ++n;
            }
        }
    out.residual = n > 100 ? (float)std::sqrt(se / n) : 0.0f;
    // the score is the squeeze alone: on Inoculant's test faces (README) clean pictures, JPEGs among them, moved
    // 0.04-0.07; cloaks of strength 4 moved 0.09, of 8 about 0.17. The fine noise is shown, not scored: a cloak made
    // with EOT is smooth at the face's scale and hides in it as well as real skin texture does.
    out.score = std::clamp((out.squeeze - 0.06f) / 0.08f, 0.0f, 1.0f);
    out.strength = out.score;   // 0..1: none, weak, strong
    out.verdict = out.score > 0.5f ? "Likely cloaked: the face's identity jumps when the picture is squeezed"
                : out.score > 0.25f ? "Maybe: a weak cloak, or a heavily processed picture"
                                    : "Looks clean: no sign of an identity cloak (a very light cloak can hide from this)";
    return true;
}

}  // namespace Antidote
