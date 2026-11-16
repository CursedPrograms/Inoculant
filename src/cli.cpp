// Inoculant - the command line.
//
//   inoculant-cli protect in.jpg out.png [--strength 4] [--steps 40] [--step 0.5] [--momentum 0.9] [--samples 4]
//                 [--robustness 0.5] [--smoothness 0] [--texture 0.6] [--colour 0.5] [--coverage 1] [--stop-at -1]
//                 [--decoy face.jpg] [--seed 1] [--cpu]
//   inoculant-cli test clean.jpg protected.png [--jpeg 75] [--blur 1] [--median 1] [--bits 5] [--resize 50]
//                 [--noise 4] [--all]
//   inoculant-cli inspect picture.jpg
//   inoculant-cli purify in.png out.png [the test's filters]
//   inoculant-cli grad-check                     the gradient graph against finite differences
#include "antidote.h"
#include "cloak.h"
#include "picture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "shell32.lib")
#endif

namespace fs = std::filesystem;

static int usage() {
    std::fprintf(stderr,
        "Inoculant: cloak the faces in your own pictures against face recognition and face swaps, and test it.\n"
        "  inoculant-cli protect in.jpg out.png [--strength 4] [--steps 40] [--step 0.5] [--momentum 0.9] [--samples 4]\n"
        "                [--robustness 0.5] [--smoothness 0] [--texture 0.6] [--colour 0.5] [--coverage 1] [--stop-at -1]\n"
        "                [--decoy face.jpg] [--seed 1] [--cpu]\n"
        "  inoculant-cli test clean.jpg protected.png [--jpeg 75] [--blur 1] [--median 1] [--bits 5] [--resize 50] [--noise 4] [--all]\n"
        "  inoculant-cli purify in.png out.png [the same filters]\n"
        "  inoculant-cli inspect picture.jpg\n"
        "  inoculant-cli grad-check\n");
    return 2;
}

int main(int argc, char** argv) {
    std::vector<std::string> a;
#ifdef _WIN32
    { int n; LPWSTR* wa = CommandLineToArgvW(GetCommandLineW(), &n); if (wa) { for (int i = 1; i < n; ++i) a.push_back(fs::path(wa[i]).u8string()); LocalFree(wa); } }
    SetConsoleOutputCP(CP_UTF8);
#else
    for (int i = 1; i < argc; ++i) a.push_back(argv[i]);
#endif
    (void)argc; (void)argv;
    if (a.empty()) return usage();
    const std::string cmd = a[0];
    std::vector<std::string> files;
    Cloak::Settings cs;
    Antidote::Settings as;
    bool anyFilter = false, cpu = false;
    std::string decoyPath;
    auto num = [&](size_t& i) { return i + 1 < a.size() ? (float)std::atof(a[++i].c_str()) : 0.0f; };
    for (size_t i = 1; i < a.size(); ++i) {
        const std::string& s = a[i];
        if (s == "--strength") cs.strength = num(i);
        else if (s == "--steps") cs.steps = (int)num(i);
        else if (s == "--step") cs.stepSize = num(i);
        else if (s == "--momentum") cs.momentum = num(i);
        else if (s == "--samples") cs.samples = (int)num(i);
        else if (s == "--robustness") cs.robustness = num(i);
        else if (s == "--smoothness") cs.smoothness = num(i);
        else if (s == "--texture") cs.texture = num(i);
        else if (s == "--colour" || s == "--color") cs.colour = num(i);
        else if (s == "--coverage") cs.coverage = num(i);
        else if (s == "--stop-at") cs.stopAt = num(i);
        else if (s == "--seed") cs.seed = (int)num(i);
        else if (s == "--decoy" && i + 1 < a.size()) { decoyPath = a[++i]; cs.goal = Cloak::G_DECOY; }
        else if (s == "--cpu") cpu = true;
        else if (s == "--jpeg") { as.jpeg = num(i); as.on[Antidote::F_JPEG] = anyFilter = true; }
        else if (s == "--blur") { as.blur = num(i); as.on[Antidote::F_BLUR] = anyFilter = true; }
        else if (s == "--median") { as.median = num(i); as.on[Antidote::F_MEDIAN] = anyFilter = true; }
        else if (s == "--bits") { as.bits = num(i); as.on[Antidote::F_BITS] = anyFilter = true; }
        else if (s == "--resize") { as.resize = num(i); as.on[Antidote::F_RESIZE] = anyFilter = true; }
        else if (s == "--noise") { as.noise = num(i); as.on[Antidote::F_NOISE] = anyFilter = true; }
        else if (s == "--all") { for (bool& b : as.on) b = true; anyFilter = true; }
        else files.push_back(s);
    }
    // filters named: only those (JPEG is on by default otherwise)
    if (anyFilter && std::find(a.begin(), a.end(), "--jpeg") == a.end() && std::find(a.begin(), a.end(), "--all") == a.end()) as.on[Antidote::F_JPEG] = false;
    std::string err;
    if (!Ort::init(!cpu, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto say = [](const std::string& s) { std::fprintf(stderr, "%s\n", s.c_str()); };
    if (!Cloak::ready(err, say)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "Running on the %s\n", Ort::device().c_str());

    if (cmd == "grad-check") {
        OrtSession* s = Ort::open("w600k_r50_grad.onnx", err);
        if (!s) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> U(-1, 1);
        std::normal_distribution<float> G(0, 1);
        Ort::Tensor x, t;
        for (const auto& nm : Ort::inputNames(s)) if (nm != "t") x.name = nm;
        x.shape = {1, 3, 112, 112}; x.f.resize(3 * 112 * 112);
        for (auto& v : x.f) v = U(rng);
        t.name = "t"; t.shape = {1, 512}; t.f.resize(512);
        double nn = 0; for (auto& v : t.f) { v = G(rng); nn += v * v; }
        for (auto& v : t.f) v = (float)(v / std::sqrt(nn));
        auto run = [&](const std::vector<float>& xf, std::vector<float>* grad) {
            Ort::Tensor xx = x; xx.f = xf;
            std::vector<Ort::Tensor> ins = {xx, t}, outs;
            if (!Ort::run(s, ins, outs, err)) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(1); }
            float c = 0;
            for (auto& o : outs) { if (o.name == "cos") c = o.f[0]; if (o.name == "grad" && grad) *grad = o.f; }
            return c;
        };
        std::vector<float> grad;
        run(x.f, &grad);
        for (int k = 0; k < 3; ++k) {
            std::vector<float> v(x.f.size()); double vn = 0;
            for (auto& e : v) { e = G(rng); vn += e * e; }
            for (auto& e : v) e = (float)(e / std::sqrt(vn));
            const float h = 1e-2f;
            std::vector<float> p = x.f, m = x.f;
            for (size_t i = 0; i < v.size(); ++i) { p[i] += h * v[i]; m[i] -= h * v[i]; }
            const double fd = (run(p, nullptr) - run(m, nullptr)) / (2 * h);
            double an = 0; for (size_t i = 0; i < v.size(); ++i) an += (double)grad[i] * v[i];
            std::printf("direction %d: finite difference %+.6f, gradient graph %+.6f\n", k, fd, an);
        }
        double gn = 0; for (float g : grad) gn += (double)g * g;
        gn = std::sqrt(gn);
        std::vector<float> p = x.f, m = x.f;
        const float h = 1e-2f;
        for (size_t i = 0; i < p.size(); ++i) { p[i] += h * grad[i] / (float)gn; m[i] -= h * grad[i] / (float)gn; }
        std::printf("along the gradient: finite difference %+.6f, |gradient| %+.6f\n", (run(p, nullptr) - run(m, nullptr)) / (2 * h), gn);
        return 0;
    }
    if (cmd == "protect" && files.size() == 2) {
        Img in; if (!loadPicture(files[0], in, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::vector<float> decoy;
        if (cs.goal == Cloak::G_DECOY) {
            Img d; std::vector<Face> df;
            if (!loadPicture(decoyPath, d, err) || !Cloak::faces(d, df, err) || df.empty()) { std::fprintf(stderr, "decoy: %s\n", err.empty() ? "no face" : err.c_str()); return 1; }
            decoy = df[0].embNorm;
        }
        Cloak::Result r;
        int last = -1;
        if (!Cloak::protect(in, cs, decoy, r, err, [&](float f, const std::string& m) { const int p = (int)(f * 10); if (p != last) { last = p; std::fprintf(stderr, "%s\n", m.c_str()); } return true; })) {
            std::fprintf(stderr, "%s\n", err.c_str()); return 1;
        }
        if (!savePicture(files[1], r.out, err, 95)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::printf("%d step(s), %.1f s; PSNR %.1f dB, largest change %.0f, mean change %.2f levels\n", r.steps, r.seconds, r.psnr, r.maxDiff, r.meanDiff);
        for (size_t i = 0; i < r.faces.size(); ++i) {
            const auto& f = r.faces[i];
            if (!f.found) std::printf("face %zu: not found again (the detector misses it too)\n", i + 1);
            else if (cs.goal == Cloak::G_DECOY) std::printf("face %zu: similarity to you %.3f, to the decoy %.3f\n", i + 1, f.after, f.decoy);
            else std::printf("face %zu: similarity to you %.3f (was 1.000)\n", i + 1, f.after);
        }
        return 0;
    }
    if ((cmd == "test" && files.size() == 2) || (cmd == "purify" && files.size() == 2)) {
        Img a1, a2;
        if (!loadPicture(files[0], a1, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        if (cmd == "purify") {
            if (!savePicture(files[1], Antidote::purify(a1, as), err, 95)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
            return 0;
        }
        if (!loadPicture(files[1], a2, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        Antidote::Report rep;
        if (!Antidote::test(a1, a2, as, rep, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::printf("protected vs clean: similarity %.3f\n", rep.protectedSim);
        std::printf("%-24s %10s %10s %8s\n", "purifier", "protected", "clean", "kept");
        for (const auto& r : rep.rows) std::printf("%-24s %10.3f %10.3f %7.0f%%\n", r.name.c_str(), r.purifiedSim, r.baselineSim, r.kept * 100);
        return 0;
    }
    if (cmd == "inspect" && files.size() == 1) {
        Img im; if (!loadPicture(files[0], im, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        Antidote::Inspection in;
        if (!Antidote::inspect(im, in, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::printf("faces %d; squeeze %.3f; fine noise %.2f levels; score %.2f\n%s\n", in.faces, in.squeeze, in.residual, in.score, in.verdict.c_str());
        return 0;
    }
    return usage();
}
