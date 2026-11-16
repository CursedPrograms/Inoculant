// Inoculant - the window: poisons and antidotes for your own pictures. Part of Atelier.
// Open a picture of yourself (or of someone who agreed); Poison cloaks the faces in it so face recognisers and the
// face swappers built on them stop reading it as that person; Antidote washes it the ways others might, and says how
// much of the cloak is left, or inspects any picture for signs of one. The controls are on the left, the picture in
// the middle: a before/after wipe, the difference as a heat-map, or the noise itself, enlarged.
//
//   Inoculant [picture]
//   Inoculant --capture out.bmp [--protect] [--test] [--inspect] [--purify] [--tab antidote] [--view 0..4]
//             [--lang code] [picture]      a screenshot from a hidden window, then quit
#include "antidote.h"
#include "cloak.h"
#include "lang.h"
#include "picture.h"
#include "splash.h"

#include <SDL.h>
#include <SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "shell32.lib")
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef INOCULANT_VERSION
#define INOCULANT_VERSION "0.1.0"
#endif

namespace fs = std::filesystem;

// ---------- the Atelier look: Changeling greys, Atelier blue on the controls; solid, no gradients, no shadows ----------
static const ImU32 C_BG = IM_COL32(0x1a, 0x1a, 0x1a, 255), C_PANEL = IM_COL32(0x24, 0x24, 0x24, 255),
                   C_LINE = IM_COL32(0x3a, 0x3a, 0x3a, 255), C_TEXT = IM_COL32(0xe6, 0xe6, 0xe6, 255),
                   C_DIM = IM_COL32(0x9a, 0x9a, 0x9a, 255), C_BLUE = IM_COL32(0x00, 0x7b, 0xb7, 255),
                   C_DEEP = IM_COL32(0x00, 0x51, 0x84, 255), C_LIGHT = IM_COL32(0x34, 0xa9, 0xe3, 255),
                   C_WELL = IM_COL32(0x14, 0x14, 0x14, 255), C_GOOD = IM_COL32(0x60, 0xd0, 0x80, 255),
                   C_BAD = IM_COL32(0xf0, 0x70, 0x70, 255), C_MID = IM_COL32(0xe8, 0xc0, 0x50, 255);
static ImVec4 col4(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }

static void applyStyle() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 0; s.FrameRounding = 2; s.GrabRounding = 2; s.PopupRounding = 2; s.ScrollbarRounding = 2; s.TabRounding = 2;
    s.WindowBorderSize = 0; s.FrameBorderSize = 0; s.PopupBorderSize = 1;
    s.WindowPadding = ImVec2(12, 10); s.FramePadding = ImVec2(7, 4); s.ItemSpacing = ImVec2(6, 7);
    ImVec4* c = s.Colors;
    c[ImGuiCol_Text] = col4(C_TEXT); c[ImGuiCol_TextDisabled] = col4(C_DIM);
    c[ImGuiCol_WindowBg] = col4(C_PANEL); c[ImGuiCol_ChildBg] = col4(C_PANEL); c[ImGuiCol_PopupBg] = col4(IM_COL32(0x21, 0x21, 0x21, 255));
    c[ImGuiCol_Border] = col4(C_LINE); c[ImGuiCol_Separator] = col4(C_LINE);
    c[ImGuiCol_FrameBg] = col4(C_WELL); c[ImGuiCol_FrameBgHovered] = col4(IM_COL32(0x2e, 0x2e, 0x2e, 255)); c[ImGuiCol_FrameBgActive] = col4(IM_COL32(0x33, 0x33, 0x33, 255));
    c[ImGuiCol_MenuBarBg] = col4(C_BG);
    c[ImGuiCol_CheckMark] = col4(C_BLUE); c[ImGuiCol_SliderGrab] = col4(C_BLUE); c[ImGuiCol_SliderGrabActive] = col4(C_LIGHT);
    c[ImGuiCol_Button] = col4(IM_COL32(0x33, 0x33, 0x33, 255)); c[ImGuiCol_ButtonHovered] = col4(C_DEEP); c[ImGuiCol_ButtonActive] = col4(C_BLUE);
    c[ImGuiCol_Header] = col4(IM_COL32(0x33, 0x33, 0x33, 255)); c[ImGuiCol_HeaderHovered] = col4(C_DEEP); c[ImGuiCol_HeaderActive] = col4(C_BLUE);
    c[ImGuiCol_Tab] = col4(IM_COL32(0x2c, 0x2c, 0x2c, 255)); c[ImGuiCol_TabHovered] = col4(C_DEEP); c[ImGuiCol_TabActive] = col4(C_BLUE);
    c[ImGuiCol_TabUnfocused] = col4(IM_COL32(0x2c, 0x2c, 0x2c, 255)); c[ImGuiCol_TabUnfocusedActive] = col4(C_DEEP);
    c[ImGuiCol_PlotHistogram] = col4(C_BLUE); c[ImGuiCol_TableRowBgAlt] = col4(IM_COL32(0x29, 0x29, 0x29, 255));
    c[ImGuiCol_TableHeaderBg] = col4(IM_COL32(0x2c, 0x2c, 0x2c, 255)); c[ImGuiCol_TableBorderLight] = col4(C_LINE); c[ImGuiCol_TableBorderStrong] = col4(C_LINE);
}

static bool fileExists(const char* f) { if (FILE* t = fopen(f, "rb")) { fclose(t); return true; } return false; }
static std::string exeDir() { char* b = SDL_GetBasePath(); std::string s = b ? b : ""; if (b) SDL_free(b); return s; }
static std::string prefsFile() { char* p = SDL_GetPrefPath("Cursed Entertainment", "Inoculant"); std::string s = p ? std::string(p) + "inoculant.ini" : ""; if (p) SDL_free(p); return s; }

static void loadFonts() {
    ImGuiIO& io = ImGui::GetIO();
    const char* ui = "C:/Windows/Fonts/segoeui.ttf";
    if (!fileExists(ui)) ui = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    if (!fileExists(ui)) { io.Fonts->AddFontDefault(); return; }
    static const ImWchar ranges[] = {0x0020, 0x024F, 0x0400, 0x04FF, 0x2010, 0x2027, 0x2190, 0x21FF, 0x2600, 0x26FF, 0};
    auto add = [&](const char* file, float size) {
        ImFont* f = io.Fonts->AddFontFromFileTTF(file, size, nullptr, ranges);
        ImFontConfig m; m.MergeMode = true;
        // Segoe UI has no stars (the credit's): Windows' symbol font fills them in
        if (fileExists("C:/Windows/Fonts/seguisym.ttf")) { static const ImWchar sym[] = {0x2190, 0x21FF, 0x2600, 0x26FF, 0}; io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/seguisym.ttf", size, &m, sym); }
        if (Lang::needsCJK()) {   // Japanese, Chinese or Korean: their font, for that language only
            const char* cjk = Lang::current() == "ko" ? "C:/Windows/Fonts/malgun.ttf" : Lang::current() == "zh" ? "C:/Windows/Fonts/msyh.ttc" : "C:/Windows/Fonts/YuGothM.ttc";
            const ImWchar* r = Lang::current() == "ko" ? io.Fonts->GetGlyphRangesKorean() : Lang::current() == "zh" ? io.Fonts->GetGlyphRangesChineseSimplifiedCommon() : io.Fonts->GetGlyphRangesJapanese();
            if (fileExists(cjk)) io.Fonts->AddFontFromFileTTF(cjk, size, &m, r);
        }
        return f;
    };
    add(ui, 16.0f);
    ImFont* title = add(fileExists("C:/Windows/Fonts/segoeuil.ttf") ? "C:/Windows/Fonts/segoeuil.ttf" : ui, 50.0f);
    ImFont* small = add(ui, 13.0f);
    Splash::setFonts(title, small);
}

static std::string fileDialog(bool save, const char* filter, const char* defExt, const std::string& start) {
#ifdef _WIN32
    std::wstring f;
    for (const char* p = filter; *p; ++p) f += (*p == '|') ? L'\0' : (wchar_t)*p;
    f += L'\0'; f += L'\0';
    std::vector<wchar_t> buf(32768, 0);
    if (!start.empty()) { std::wstring s = fs::u8path(start).wstring(); wcsncpy(buf.data(), s.c_str(), buf.size() - 1); }
    OPENFILENAMEW o{};
    o.lStructSize = sizeof(o);
    o.lpstrFilter = f.c_str(); o.lpstrFile = buf.data(); o.nMaxFile = (DWORD)buf.size();
    std::wstring ext = defExt ? std::wstring(defExt, defExt + strlen(defExt)) : L"";
    o.lpstrDefExt = defExt ? ext.c_str() : nullptr;
    o.Flags = save ? (OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR) : (OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR);
    if (!(save ? GetSaveFileNameW(&o) : GetOpenFileNameW(&o))) return "";
    return fs::path(buf.data()).u8string();
#else
    (void)filter; (void)start;
    std::string cmd = save ? "zenity --file-selection --save --confirm-overwrite 2>/dev/null" : "zenity --file-selection 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return {};
    char line[4096] = {0};
    std::string r = fgets(line, sizeof line, p) ? line : "";
    pclose(p);
    while (!r.empty() && (r.back() == '\n' || r.back() == '\r')) r.pop_back();
    if (save && !r.empty() && defExt && r.find('.') == std::string::npos) r += std::string(".") + defExt;
    return r;
#endif
}

static const char* PICTURES = "Pictures|*.png;*.jpg;*.jpeg;*.bmp;*.tga|All files|*.*|";

static void upload(GLuint& tex, const Img& im) {
    if (im.empty()) return;
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);   // close up, the noise's own pixels
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, im.w, im.h, 0, GL_RGB, GL_UNSIGNED_BYTE, im.px.data());
}

// a face's square, for the decoy's thumbnail
static Img faceThumb(const Img& im, const Face& f, int size) {
    const float cx = (f.box[0] + f.box[2]) / 2, cy = (f.box[1] + f.box[3]) / 2, side = std::max(f.box[2] - f.box[0], f.box[3] - f.box[1]) * 1.3f;
    Affine M; M.m[0] = M.m[4] = size / side; M.m[2] = -(cx - side / 2) * M.m[0]; M.m[5] = -(cy - side / 2) * M.m[4];
    const Crop c = warpCrop(im, M, size, size);
    Img t(size, size);
    for (size_t i = 0; i < t.px.size(); ++i) t.px[i] = (uint8_t)std::clamp(c.px[i] + 0.5f, 0.0f, 255.0f);
    return t;
}

// ---------- state ----------
static SDL_Window* window = nullptr;
static bool quit = false;

enum View { V_WIPE, V_AFTER, V_BEFORE, V_HEAT, V_NOISE, V_COUNT };
enum Kind { K_NONE, K_PROTECTED, K_PURIFIED, K_OPENED };
static std::string picturePath, testPath;
static Img original, result, protectedPic;   // the picture; what's shown after; the latest protected picture
static int resultKind = K_NONE;
static std::vector<Face> originalFaces;
static GLuint originalTex = 0, resultTex = 0, heatTex = 0, noiseTex = 0, decoyTex = 0;
static bool originalStale = false, resultStale = false, diffStale = false, decoyStale = false;
static float wipe = 0.5f, amplify = 8.0f, matchAt = 0.40f;
static int view = V_WIPE, tab = 0, jpegQuality = 95;
static bool showAbout = false, forceTab = false;

static Cloak::Settings cloakSt;
static Cloak::Result cloakRes; static bool haveCloak = false;
static std::string decoyPath; static std::vector<float> decoyEmb; static Img decoyThumb;
static Antidote::Settings antiSt;
static Antidote::Report report; static bool haveReport = false;
static Antidote::Inspection inspection; static bool haveInspection = false;

// the worker: model work, one job at a time
static std::mutex mtx;
static std::deque<std::function<void()>> jobs;
static std::thread worker;
static std::atomic<bool> busy{false}, stopJob{false}, workerQuit{false}, modelsReady{false};
static std::atomic<float> progress{-1.0f};
static std::string status;
static std::vector<std::function<void()>> onMain;   // results handed back to the main thread (textures live there)

static void say(const std::string& s) { std::lock_guard<std::mutex> g(mtx); status = s; }
static void post(std::function<void()> f) { std::lock_guard<std::mutex> g(mtx); onMain.push_back(std::move(f)); }
static void queue(std::function<void()> f) { std::lock_guard<std::mutex> g(mtx); jobs.push_back(std::move(f)); }
static bool idle() { std::lock_guard<std::mutex> g(mtx); return jobs.empty() && !busy; }

static void workerLoop() {
    while (!workerQuit) {
        std::function<void()> job;
        {
            std::lock_guard<std::mutex> g(mtx);
            if (!jobs.empty()) { job = std::move(jobs.front()); jobs.pop_front(); busy = true; }
        }
        if (!job) { std::this_thread::sleep_for(std::chrono::milliseconds(15)); continue; }
        job();
        busy = false;
    }
}

static std::string base(const std::string& p) { return fs::u8path(p).filename().u8string(); }
static std::string fmt(const char* f, double a) { char b[256]; snprintf(b, sizeof b, f, a); return b; }

static void setResult(const Img& im, int kind) { result = im; resultKind = kind; resultStale = true; diffStale = true; }

static void loadPicture(const std::string& path) {
    queue([path] {
        Img im; std::string err;
        if (!::loadPicture(path, im, err)) { say(err); return; }
        std::vector<Face> faces;
        if (!Cloak::faces(im, faces, err)) { say(err); faces.clear(); }
        post([path, im, faces] {
            picturePath = path; original = im; originalFaces = faces; originalStale = true;
            result = Img(); resultKind = K_NONE; protectedPic = Img(); haveCloak = haveReport = haveInspection = false; diffStale = true;
            testPath.clear();
        });
        say(base(path) + " (" + std::to_string(im.w) + " x " + std::to_string(im.h) + "), " + std::to_string(faces.size()) + " " + T("face(s)"));
    });
}

// a picture to test in the Antidote tab (a protected one from elsewhere): the original stays the clean reference
static void loadTestPicture(const std::string& path) {
    queue([path] {
        Img im; std::string err;
        if (!::loadPicture(path, im, err)) { say(err); return; }
        post([path, im] {
            if (original.empty()) { picturePath = path; original = im; originalStale = true; }
            else if (im.w != original.w || im.h != original.h) { say(T("That picture isn't the original's size: open it as the picture instead")); return; }
            testPath = path; protectedPic = im; setResult(im, K_OPENED); haveReport = haveInspection = false;
        });
        say(std::string(T("Picture to test: ")) + base(path));
    });
}

static void chooseDecoy(const std::string& path) {
    queue([path] {
        Img im; std::string err; std::vector<Face> faces;
        if (!::loadPicture(path, im, err)) { say(err); return; }
        if (!Cloak::faces(im, faces, err) || faces.empty()) { say(base(path) + ": " + (err.empty() ? T("no face in it") : err)); return; }
        const Img thumb = faceThumb(im, faces[0], 96);
        const std::vector<float> e = faces[0].embNorm;
        post([path, thumb, e] { decoyPath = path; decoyThumb = thumb; decoyEmb = e; decoyStale = true; cloakSt.goal = Cloak::G_DECOY; });
        say(std::string(T("Decoy face: ")) + base(path));
    });
}

static void runProtect() {
    if (original.empty() || progress >= 0) return;
    if (cloakSt.goal == Cloak::G_DECOY && decoyEmb.empty()) { say(T("Decoy: choose a picture of the decoy face first")); return; }
    const Img im = original; const auto st = cloakSt; const auto decoy = decoyEmb;
    stopJob = false; progress = 0.0f;
    queue([im, st, decoy] {
        Cloak::Result r; std::string err;
        const bool ok = Cloak::protect(im, st, decoy, r, err, [](float f, const std::string& m) { progress = f; say(m); return !stopJob; });
        progress = -1.0f;
        if (!ok) { say(err); return; }
        post([r] { cloakRes = r; haveCloak = true; protectedPic = r.out; testPath.clear(); setResult(r.out, K_PROTECTED); haveReport = haveInspection = false; });
        say(fmt(T("Protected in %.1f s"), r.seconds) + fmt(", PSNR %.1f dB", r.psnr) + (stopJob ? std::string(" (") + T("stopped early") + ")" : ""));
    });
}

static void runTest() {
    if (original.empty() || protectedPic.empty()) { say(T("Protect the picture first (or open a protected one)")); return; }
    const Img clean = original, prot = protectedPic; const auto st = antiSt;
    queue([clean, prot, st] {
        say(T("Testing the purifiers..."));
        Antidote::Report r; std::string err;
        if (!Antidote::test(clean, prot, st, r, err)) { say(err); return; }
        const Img shown = Antidote::purify(prot, st);
        post([r, shown] { report = r; haveReport = true; setResult(shown, K_PURIFIED); });
        say(T("Robustness test done"));
    });
}

static void runPurify() {
    const Img src = !protectedPic.empty() ? protectedPic : original;
    if (src.empty()) return;
    const auto st = antiSt;
    queue([src, st] {
        say(T("Purifying..."));
        const Img out = Antidote::purify(src, st);
        post([out] { setResult(out, K_PURIFIED); });
        say(T("Purified: the wipe shows the original and the purified picture"));
    });
}

static void runInspect() {
    const Img src = !protectedPic.empty() ? protectedPic : original;
    if (src.empty()) return;
    queue([src] {
        say(T("Inspecting..."));
        Antidote::Inspection in; std::string err;
        if (!Antidote::inspect(src, in, err)) { say(err); return; }
        post([in] { inspection = in; haveInspection = true; });
        say(T("Inspection done"));
    });
}

static std::string defaultOut(const char* ending) {
    const fs::path p = fs::u8path(picturePath);
    return (p.parent_path() / (p.stem().u8string() + ending + ".png")).u8string();
}

static void saveResultAs() {
    if (result.empty()) return;
    const std::string p = fileDialog(true, "PNG (keeps the cloak exactly)|*.png|JPEG|*.jpg|", "png", defaultOut(resultKind == K_PURIFIED ? "_purified" : "_inoculated"));
    if (p.empty()) return;
    std::string err;
    if (savePicture(p, result, err, jpegQuality)) say(std::string(T("Saved ")) + p); else say(err);
}

// the difference pictures: a heat-map of how much each pixel changed, and the change itself enlarged round grey
static Img heatImg, noiseImg;
static void makeDiffs() {
    heatImg = Img(); noiseImg = Img();
    if (original.empty() || result.empty() || result.w != original.w || result.h != original.h) return;
    heatImg = Img(original.w, original.h); noiseImg = Img(original.w, original.h);
    // the heat-map's colours: black, deep blue, blue, light blue, white
    static const float stops[5][3] = {{0x14, 0x14, 0x14}, {0x00, 0x51, 0x84}, {0x00, 0x7b, 0xb7}, {0x34, 0xa9, 0xe3}, {0xff, 0xff, 0xff}};
    for (size_t i = 0; i < original.px.size(); i += 3) {
        int m = 0;
        for (int k = 0; k < 3; ++k) {
            const int d = (int)result.px[i + k] - original.px[i + k];
            m = std::max(m, std::abs(d));
            noiseImg.px[i + k] = (uint8_t)std::clamp(128.0f + d * amplify, 0.0f, 255.0f);
        }
        const float t = std::clamp(m * amplify / 255.0f, 0.0f, 1.0f) * 4.0f;
        const int s = std::min(3, (int)t);
        const float f = t - s;
        for (int k = 0; k < 3; ++k) heatImg.px[i + k] = (uint8_t)(stops[s][k] * (1 - f) + stops[s + 1][k] * f);
    }
}

// ---------- widgets ----------
static void heading(const char* t) { ImGui::PushStyleColor(ImGuiCol_Text, col4(C_LIGHT)); ImGui::TextUnformatted(t); ImGui::PopStyleColor(); }
static void tip(const char* t) { if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) { ImGui::BeginTooltip(); ImGui::PushTextWrapPos(380); ImGui::TextUnformatted(t); ImGui::PopTextWrapPos(); ImGui::EndTooltip(); } }
static bool sliderF(const char* id, const char* label, float& v, float lo, float hi, const char* f, const char* help) {
    const std::string format = std::string(T(label)) + ": " + f;
    const bool ch = ImGui::SliderFloat(id, &v, lo, hi, format.c_str());
    tip(T(help));
    return ch;
}
static bool sliderI(const char* id, const char* label, int& v, int lo, int hi, const char* help) {
    const std::string format = std::string(T(label)) + ": %d";
    const bool ch = ImGui::SliderInt(id, &v, lo, hi, format.c_str());
    tip(T(help));
    return ch;
}
static bool bigButton(const char* label, float w, bool primary = true) {
    if (primary) { ImGui::PushStyleColor(ImGuiCol_Button, col4(C_BLUE)); ImGui::PushStyleColor(ImGuiCol_ButtonHovered, col4(C_LIGHT)); ImGui::PushStyleColor(ImGuiCol_ButtonActive, col4(C_DEEP)); }
    const bool r = ImGui::Button(label, ImVec2(w, 30));
    if (primary) ImGui::PopStyleColor(3);
    return r;
}
static ImU32 simColour(float s) { return s < 0 ? C_DIM : s < matchAt * 0.75f ? C_GOOD : s < matchAt ? C_MID : C_BAD; }

// ---------- the frame ----------
static void menuBar() {
    if (!ImGui::BeginMainMenuBar()) return;
    if (ImGui::BeginMenu(T("File"))) {
        if (ImGui::MenuItem(T("Open a picture..."), "Ctrl+O")) { const std::string p = fileDialog(false, PICTURES, nullptr, ""); if (!p.empty()) loadPicture(p); }
        if (ImGui::MenuItem(T("Open a protected picture to test..."), nullptr, false, !original.empty())) { const std::string p = fileDialog(false, PICTURES, nullptr, ""); if (!p.empty()) { loadTestPicture(p); tab = 1; forceTab = true; } }
        ImGui::Separator();
        if (ImGui::MenuItem(T("Save the result as..."), "Ctrl+S", false, !result.empty())) saveResultAs();
        ImGui::Separator();
        if (ImGui::MenuItem(T("Quit"), "Ctrl+Q")) quit = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu(T("View"))) {
        const char* names[V_COUNT] = {T("Before | After"), T("After"), T("Before"), T("Difference (heat-map)"), T("Noise (enlarged)")};
        for (int i = 0; i < V_COUNT; ++i) { char k[4]; snprintf(k, sizeof k, "%d", i + 1); if (ImGui::MenuItem(names[i], k, view == i)) view = i; }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu(T("Language"))) {
        for (const auto& p : Lang::packs())
            if (ImGui::MenuItem(p.name.c_str(), nullptr, Lang::current() == p.code)) {
                Lang::load(exeDir() + "lang", p.code);
                std::ofstream(fs::u8path(prefsFile())) << "language=" << p.code << "\n";
                if (Lang::needsCJK()) say(T("Restart Inoculant for this language's letters"));
            }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu(T("Help"))) { if (ImGui::MenuItem(T("About Inoculant"))) showAbout = true; ImGui::EndMenu(); }
    ImGui::EndMainMenuBar();
}

static void keys() {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return;
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O)) { const std::string p = fileDialog(false, PICTURES, nullptr, ""); if (!p.empty()) loadPicture(p); }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) saveResultAs();
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Q)) quit = true;
    for (int i = 0; i < V_COUNT; ++i) if (!io.KeyCtrl && ImGui::IsKeyPressed((ImGuiKey)(ImGuiKey_1 + i))) view = i;
}

static void poisonTab(float w) {
    heading(T("GOAL"));
    const char* goals[Cloak::G_COUNT] = {T("Hide (away from you)"), T("Decoy (toward another face)")};
    ImGui::SetNextItemWidth(w);
    if (ImGui::BeginCombo("##goal", goals[cloakSt.goal])) {
        for (int i = 0; i < Cloak::G_COUNT; ++i) if (ImGui::Selectable(goals[i], cloakSt.goal == i)) cloakSt.goal = i;
        ImGui::EndCombo();
    }
    tip(T("Hide pushes what a face recogniser sees away from you: it, and the face swappers that use it, no longer find you. Decoy pulls it toward another face (use an AI-generated one), so a swap made from your picture comes out as that face."));
    if (cloakSt.goal == Cloak::G_DECOY) {
        if (decoyTex) { ImGui::Image((ImTextureID)(intptr_t)decoyTex, ImVec2(56, 56)); ImGui::SameLine(); }
        ImGui::BeginGroup();
        ImGui::TextWrapped("%s", decoyPath.empty() ? T("No decoy face yet") : base(decoyPath).c_str());
        if (ImGui::Button(T("Choose the decoy face..."))) { const std::string p = fileDialog(false, PICTURES, nullptr, ""); if (!p.empty()) chooseDecoy(p); }
        ImGui::EndGroup();
        ImGui::TextDisabled("%s", T("Use an AI-generated face, not a real person's."));
    }
    ImGui::PushItemWidth(w);
    ImGui::Spacing();
    heading(T("STRENGTH"));
    sliderF("##eps", "Strength", cloakSt.strength, 1.0f, 16.0f, "%.1f / 255", "The most any pixel may change, in levels of 255. 2-4 is hard to see; 8 and up is strong but shows on smooth skin.");
    sliderI("##steps", "Steps", cloakSt.steps, 5, 200, "How many times it looks and adjusts. More: a stronger cloak for the same strength, slower.");
    sliderF("##step", "Step size", cloakSt.stepSize, 0.1f, 4.0f, "%.2f", "How far each step moves the pixels, in levels. Smaller steps and more of them find a finer cloak.");
    sliderF("##mom", "Momentum", cloakSt.momentum, 0.0f, 1.0f, "%.2f", "MI-FGSM: each step keeps this much of the last ones' direction. Higher carries over better to face recognisers Inoculant hasn't seen.");
    sliderF("##stop", "Stop at similarity", cloakSt.stopAt, -1.0f, 1.0f, cloakSt.stopAt <= -1.0f ? T("never") : "%.2f", "Hide: stop as soon as every face is this unlike you (the lightest cloak that does it). -1: run every step.");
    ImGui::Spacing();
    heading(T("ROBUSTNESS"));
    sliderI("##views", "Views per step", cloakSt.samples, 1, 12, "EOT: how many slightly shifted, turned, scaled or blurred views of the face each step looks at. More views: a cloak that survives re-alignment and resizing better.");
    sliderF("##jit", "Jitter", cloakSt.robustness, 0.0f, 1.0f, "%.2f", "EOT: how far those views are moved (up to 4 px, 6 degrees, 8% scale and a blur). 0: only the exact alignment.");
    ImGui::Spacing();
    heading(T("LOOK"));
    sliderF("##tex", "Texture masking", cloakSt.texture, 0.0f, 1.0f, "%.2f", "Put the change where there's detail to hide it (hair, brows, lashes, fabric) and less on smooth skin. 0: the same everywhere.");
    sliderF("##col", "Colour", cloakSt.colour, 0.0f, 1.0f, "%.2f", "0: only lighter and darker (R, G and B move together: least visible as colour blotches). 1: each colour on its own (stronger).");
    sliderF("##grain", "Grain", cloakSt.smoothness, 0.0f, 1.0f, "%.2f", "0: the finest grain. Higher: softer, larger blotches that survive blur and JPEG a little better but show more.");
    sliderF("##cov", "Coverage", cloakSt.coverage, 0.0f, 1.0f, "%.2f", "How much round each face may change: 0 the inner face only, 1 the whole area the recogniser looks at.");
    sliderI("##seed", "Seed", cloakSt.seed, 1, 999, "The random views' seed: another seed, another cloak.");
    ImGui::PopItemWidth();
    if (ImGui::Button(T("Defaults"), ImVec2(w, 0))) { const int goal = cloakSt.goal; cloakSt = Cloak::Settings(); cloakSt.goal = goal; }
    ImGui::Spacing();
    if (progress >= 0) {
        ImGui::ProgressBar(progress, ImVec2(w, 8), "");
        if (bigButton(T("Stop"), w, false)) stopJob = true;
    } else {
        ImGui::BeginDisabled(original.empty() || originalFaces.empty() || !modelsReady);
        if (bigButton(T("Protect"), w)) runProtect();
        ImGui::EndDisabled();
        if (!original.empty() && originalFaces.empty()) ImGui::TextDisabled("%s", T("No face in this picture: the cloak works on faces."));
    }
    if (haveCloak) {
        ImGui::Spacing();
        heading(T("RESULT"));
        for (size_t i = 0; i < cloakRes.faces.size(); ++i) {
            const auto& f = cloakRes.faces[i];
            if (!f.found) { ImGui::TextColored(col4(C_GOOD), "%s %zu: %s", T("Face"), i + 1, T("not found again")); continue; }
            ImGui::Text("%s %zu: %s", T("Face"), i + 1, T("similarity to you"));
            ImGui::SameLine(); ImGui::TextColored(col4(simColour(f.after)), "%.2f", f.after);
            ImGui::SameLine(); ImGui::TextDisabled("(%s 1.00)", T("was"));
            if (cloakSt.goal == Cloak::G_DECOY) { ImGui::Text("   %s", T("to the decoy")); ImGui::SameLine(); ImGui::TextColored(col4(f.decoy > matchAt ? C_GOOD : C_DIM), "%.2f", f.decoy); }
        }
        bool all = true;
        for (const auto& f : cloakRes.faces) if (f.found && f.after >= matchAt) all = false;
        ImGui::PushTextWrapPos(0);
        ImGui::TextColored(col4(all ? C_GOOD : C_MID), "%s", all ? T("ArcFace no longer takes it for the same person.") : T("ArcFace still matches it: try more strength or steps."));
        ImGui::PopTextWrapPos();
        ImGui::TextDisabled("PSNR %.1f dB, %s %.0f, %s %.2f", cloakRes.psnr, T("largest change"), cloakRes.maxDiff, T("mean"), cloakRes.meanDiff);
        ImGui::TextDisabled("%d %s, %.1f s", cloakRes.steps, T("steps"), cloakRes.seconds);
    }
}

static void antidoteTab(float w) {
    heading(T("PICTURE TO TEST"));
    ImGui::PushTextWrapPos(0);
    if (!testPath.empty()) ImGui::TextUnformatted(base(testPath).c_str());
    else if (!protectedPic.empty()) ImGui::TextUnformatted(T("The protected picture (from Poison)"));
    else if (!original.empty()) ImGui::TextDisabled("%s", T("The picture itself (nothing protected yet): Inspect works on it"));
    else ImGui::TextDisabled("%s", T("Open a picture first"));
    ImGui::PopTextWrapPos();
    ImGui::BeginDisabled(original.empty());
    if (ImGui::Button(T("Open a protected picture..."), ImVec2(w, 0))) { const std::string p = fileDialog(false, PICTURES, nullptr, ""); if (!p.empty()) loadTestPicture(p); }
    ImGui::EndDisabled();
    tip(T("A protected copy of the open picture, from Inoculant or another tool: the open picture stays the clean original it's compared with."));
    ImGui::Spacing();
    heading(T("PURIFIERS"));
    ImGui::PushItemWidth(w - 34);
    struct P { int f; const char* id; float* v; float lo, hi; const char* fmt; };
    const P ps[] = {{Antidote::F_JPEG, "##jq", &antiSt.jpeg, 10, 100, "%.0f"}, {Antidote::F_BLUR, "##bl", &antiSt.blur, 0.3f, 6, "%.1f px"},
                    {Antidote::F_MEDIAN, "##md", &antiSt.median, 1, 5, "%.0f px"}, {Antidote::F_BITS, "##bt", &antiSt.bits, 1, 8, "%.0f bits"},
                    {Antidote::F_RESIZE, "##rs", &antiSt.resize, 5, 100, "%.0f%%"}, {Antidote::F_NOISE, "##nz", &antiSt.noise, 0.5f, 32, "%.1f / 255"}};
    for (const auto& p : ps) {
        ImGui::PushID(p.f);
        ImGui::Checkbox("##on", &antiSt.on[p.f]);
        tip(T(Antidote::filterHint(p.f)));
        ImGui::SameLine();
        ImGui::BeginDisabled(!antiSt.on[p.f]);
        sliderF(p.id, Antidote::filterName(p.f), *p.v, p.lo, p.hi, p.fmt, Antidote::filterHint(p.f));
        ImGui::EndDisabled();
        ImGui::PopID();
    }
    ImGui::PopItemWidth();
    ImGui::PushItemWidth(w);
    sliderF("##match", "Match threshold", matchAt, 0.1f, 0.8f, "%.2f", "ArcFace similarity above which a face counts as the same person. 0.40 is about where face swappers and face search start to match.");
    ImGui::PopItemWidth();
    ImGui::Spacing();
    const bool can = !original.empty() && modelsReady && progress < 0;
    ImGui::BeginDisabled(!can || protectedPic.empty());
    if (bigButton(T("Robustness test"), w)) runTest();
    ImGui::EndDisabled();
    tip(T("Each purifier switched on, then all of them in a row, on the protected picture and on the original: how alike ArcFace finds them to the original face, and how much of the protection is left."));
    ImGui::BeginDisabled(!can);
    if (ImGui::Button(T("Purify and show"), ImVec2(w * 0.5f - 3, 0))) runPurify();
    tip(T("The purifiers switched on, in a row: the result is shown after the wipe, ready to save."));
    ImGui::SameLine();
    if (ImGui::Button(T("Inspect"), ImVec2(-1, 0))) runInspect();
    tip(T("No original needed: does this picture carry an identity cloak? Feature squeezing: a clean face's identity hardly moves when the picture is squeezed; a cloaked one jumps."));
    ImGui::EndDisabled();

    if (haveReport) {
        ImGui::Spacing();
        heading(T("HOW MUCH SURVIVES"));
        ImGui::Text("%s", T("Protected, before purifying:")); ImGui::SameLine(); ImGui::TextColored(col4(simColour(report.protectedSim)), "%.2f", report.protectedSim);
        if (ImGui::BeginTable("##rep", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn(T("Purifier"), ImGuiTableColumnFlags_WidthStretch, 2.2f);
            ImGui::TableSetupColumn(T("Similarity"), ImGuiTableColumnFlags_WidthStretch, 1.1f);
            ImGui::TableSetupColumn(T("Kept"), ImGuiTableColumnFlags_WidthStretch, 1.3f);
            ImGui::TableHeadersRow();
            for (const auto& r : report.rows) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(T(r.name.c_str()));
                ImGui::TableNextColumn(); ImGui::TextColored(col4(simColour(r.purifiedSim)), "%.2f", r.purifiedSim);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s %.3f\n%s %.3f", T("Protected, purified, vs the original:"), r.purifiedSim, T("The original, purified, vs itself:"), r.baselineSim);
                ImGui::TableNextColumn();
                char k[32]; snprintf(k, sizeof k, "%.0f%%", r.kept * 100);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col4(r.purifiedSim < matchAt ? C_BLUE : C_BAD));
                ImGui::ProgressBar(std::min(1.0f, r.kept), ImVec2(-1, 0), k);
                ImGui::PopStyleColor();
            }
            ImGui::EndTable();
        }
        ImGui::PushTextWrapPos(0);
        ImGui::TextDisabled("%s", T("Kept: how much of the drop in similarity is still there after the purifier (its own effect on the clean original taken off). Red: the face matches again."));
        ImGui::PopTextWrapPos();
    }
    if (haveInspection) {
        ImGui::Spacing();
        heading(T("INSPECTION"));
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col4(inspection.score > 0.5f ? C_BLUE : inspection.score > 0.25f ? C_MID : C_DIM));
        char k[32]; snprintf(k, sizeof k, "%.0f%%", inspection.score * 100);
        ImGui::ProgressBar(inspection.score, ImVec2(w, 0), k);
        ImGui::PopStyleColor();
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(T(inspection.verdict.c_str()));
        ImGui::PopTextWrapPos();
        ImGui::TextDisabled("%s %.3f, %s %.2f", T("squeeze"), inspection.squeeze, T("fine noise"), inspection.residual);
        tip(T("Squeeze: the most the face's identity moved under JPEG, median, bit-depth and blur squeezing (clean pictures: about 0.04-0.07). Fine noise: the face crop's high-frequency noise on smooth skin, in levels."));
    }
}

static void leftPanel(ImVec2 pos, ImVec2 size) {
    ImGui::SetNextWindowPos(pos); ImGui::SetNextWindowSize(size);
    ImGui::Begin("##left", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove);
    const float w = ImGui::GetContentRegionAvail().x;
    heading(T("PICTURE"));
    if (picturePath.empty()) ImGui::TextDisabled("%s", T("Open or drop a picture of yourself"));
    else {
        ImGui::TextWrapped("%s", base(picturePath).c_str());
        tip(picturePath.c_str());
        ImGui::TextDisabled("%d x %d, %zu %s", original.w, original.h, originalFaces.size(), T("face(s)"));
    }
    if (ImGui::Button(T("Open..."), ImVec2(w * 0.5f - 3, 0))) { const std::string p = fileDialog(false, PICTURES, nullptr, ""); if (!p.empty()) loadPicture(p); }
    ImGui::SameLine();
    ImGui::BeginDisabled(result.empty());
    if (ImGui::Button(T("Save..."), ImVec2(-1, 0))) saveResultAs();
    ImGui::EndDisabled();
    tip(T("Save what's shown after the wipe. PNG keeps the cloak exactly; a JPEG wears it down a little (see Antidote)."));
    ImGui::Spacing();
    const float footH = 74;
    ImGui::BeginChild("##tabs", ImVec2(w, size.y - ImGui::GetCursorPosY() - footH), false);
    const float cw = ImGui::GetContentRegionAvail().x - 4;
    if (ImGui::BeginTabBar("##tb")) {
        const int want = forceTab ? tab : -1;   // a tab asked for (the command line, File > Open a protected picture)
        if (ImGui::BeginTabItem(T("Poison"), nullptr, want == 0 ? ImGuiTabItemFlags_SetSelected : 0)) { tab = 0; ImGui::Spacing(); poisonTab(cw); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem(T("Antidote"), nullptr, want == 1 ? ImGuiTabItemFlags_SetSelected : 0)) { tab = 1; ImGui::Spacing(); antidoteTab(cw); ImGui::EndTabItem(); }
        forceTab = false;
        ImGui::EndTabBar();
    }
    ImGui::EndChild();
    ImGui::SetCursorPosY(size.y - footH + 6);
    ImGui::Separator();
    { std::lock_guard<std::mutex> g(mtx); ImGui::PushTextWrapPos(0); ImGui::TextDisabled("%s", status.c_str()); ImGui::PopTextWrapPos(); }
    ImGui::SetCursorPosY(size.y - 26);
    ImGui::TextDisabled("Inoculant " INOCULANT_VERSION " - %s", T("part of Atelier"));
    ImGui::End();
}

static void canvas(ImVec2 pos, ImVec2 size) {
    ImGui::SetNextWindowPos(pos); ImGui::SetNextWindowSize(size);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, col4(C_BG));
    ImGui::Begin("##canvas", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (original.empty()) {
        const ImVec2 c(pos.x + size.x / 2, pos.y + size.y / 2);
        const float bw = std::min(560.0f, size.x - 80), bh = 240;
        const ImVec2 a(c.x - bw / 2, c.y - bh / 2), b(c.x + bw / 2, c.y + bh / 2);
        for (float x = a.x; x < b.x; x += 12) { dl->AddLine(ImVec2(x, a.y), ImVec2(std::min(x + 6, b.x), a.y), C_LINE, 1.5f); dl->AddLine(ImVec2(x, b.y), ImVec2(std::min(x + 6, b.x), b.y), C_LINE, 1.5f); }
        for (float y = a.y; y < b.y; y += 12) { dl->AddLine(ImVec2(a.x, y), ImVec2(a.x, std::min(y + 6, b.y)), C_LINE, 1.5f); dl->AddLine(ImVec2(b.x, y), ImVec2(b.x, std::min(y + 6, b.y)), C_LINE, 1.5f); }
        auto centred = [&](const char* t, float y, ImU32 col) { const ImVec2 ts = ImGui::CalcTextSize(t); dl->AddText(ImVec2(c.x - ts.x / 2, y), col, t); };
        centred(T("Drop a picture of yourself here"), c.y - 34, C_TEXT);
        centred(T("Inoculant cloaks the faces in it against face recognition and face swaps."), c.y - 6, C_DIM);
        centred(T("Only protect pictures of yourself, or of people who agreed."), c.y + 22, C_DIM);
        ImGui::End(); ImGui::PopStyleColor();
        return;
    }
    // the view's bar along the top
    const char* names[V_COUNT] = {T("Before | After"), T("After"), T("Before"), T("Difference"), T("Noise")};
    ImGui::SetCursorPos(ImVec2(16, 10));
    for (int i = 0; i < V_COUNT; ++i) {
        if (i) ImGui::SameLine();
        const bool on = view == i;
        if (on) ImGui::PushStyleColor(ImGuiCol_Button, col4(C_BLUE));
        if (ImGui::Button(names[i])) view = i;
        if (on) ImGui::PopStyleColor();
    }
    if (view == V_HEAT || view == V_NOISE) {
        ImGui::SameLine(); ImGui::SetNextItemWidth(200);
        if (sliderF("##amp", "Enlarge", amplify, 1.0f, 64.0f, "x%.0f", "How much the difference is enlarged so you can see it: a change of 4 levels at x8 shows as 32.")) diffStale = true;
    }
    const float top = 48, pad = 20;
    const float s = std::min((size.x - pad * 2) / original.w, (size.y - top - pad) / original.h);
    const float dw = original.w * s, dh = original.h * s;
    const ImVec2 a(pos.x + (size.x - dw) / 2, pos.y + top + (size.y - top - pad - dh) / 2), b(a.x + dw, a.y + dh);
    const bool after = !result.empty() && resultTex;
    auto label = [&](const char* t, float x) {   // on a dark tab, so it reads on bright pictures
        const ImVec2 ts = ImGui::CalcTextSize(t);
        dl->AddRectFilled(ImVec2(x - 6, a.y + 6), ImVec2(x + ts.x + 6, a.y + 10 + ts.y), IM_COL32(0x1a, 0x1a, 0x1a, 230), 2);
        dl->AddText(ImVec2(x, a.y + 8), C_TEXT, t);
    };
    const char* afterName = T("After");   // as Mimic: Before on the left of the wipe, After on the right (protected or purified)
    if (!after || view == V_BEFORE) { dl->AddImage((ImTextureID)(intptr_t)originalTex, a, b); if (after) label(T("Before"), a.x + 14); }
    else if (view == V_AFTER) { dl->AddImage((ImTextureID)(intptr_t)resultTex, a, b); label(afterName, a.x + 14); }
    else if ((view == V_HEAT || view == V_NOISE) && (view == V_HEAT ? heatTex : noiseTex)) {
        dl->AddImage((ImTextureID)(intptr_t)(view == V_HEAT ? heatTex : noiseTex), a, b);
        label(view == V_HEAT ? T("How much each pixel changed") : T("The change itself, round grey"), a.x + 14);
    } else {
        const float x = a.x + dw * wipe;
        dl->AddImage((ImTextureID)(intptr_t)originalTex, a, ImVec2(x, b.y), ImVec2(0, 0), ImVec2(wipe, 1));
        dl->AddImage((ImTextureID)(intptr_t)resultTex, ImVec2(x, a.y), b, ImVec2(wipe, 0), ImVec2(1, 1));
        dl->AddLine(ImVec2(x, a.y), ImVec2(x, b.y), C_BLUE, 2.0f);
        dl->AddCircleFilled(ImVec2(x, a.y + dh / 2), 9, C_BLUE);
        label(T("Before"), a.x + 14);
        label(afterName, b.x - 14 - ImGui::CalcTextSize(afterName).x);
    }
    // the faces found
    if (!after) for (const auto& f : originalFaces) dl->AddRect(ImVec2(a.x + f.box[0] * s, a.y + f.box[1] * s), ImVec2(a.x + f.box[2] * s, a.y + f.box[3] * s), IM_COL32(0x34, 0xa9, 0xe3, 120), 2, 0, 1.0f);
    ImGui::SetCursorScreenPos(a);
    ImGui::InvisibleButton("view", ImVec2(dw, dh));
    if (ImGui::IsItemActive() && view == V_WIPE) wipe = std::clamp((ImGui::GetIO().MousePos.x - a.x) / dw, 0.0f, 1.0f);
    if (busy) {
        const float t = (float)ImGui::GetTime() * 6;
        const ImVec2 c(b.x - 22, b.y - 22);
        dl->AddCircleFilled(c, 16, IM_COL32(0x1a, 0x1a, 0x1a, 220));
        dl->PathArcTo(c, 10, t, t + 4.2f, 24);
        dl->PathStroke(C_BLUE, 0, 2.5f);
    }
    ImGui::End(); ImGui::PopStyleColor();
}

static void about() {
    if (showAbout) { ImGui::OpenPopup("##about"); showAbout = false; }
    ImGui::SetNextWindowSize(ImVec2(Splash::W + 24, 0));
    if (ImGui::BeginPopupModal("##about", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar)) {
        const ImVec2 at = ImGui::GetCursorScreenPos();
        Splash::draw(ImGui::GetWindowDrawList(), at.x, at.y, nullptr, -1);
        ImGui::Dummy(ImVec2(Splash::W, Splash::H));
        ImGui::Spacing();
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(T("Poisons and antidotes for your own pictures. Poison cloaks the faces in a picture so face recognisers, and the face swappers and editors that rely on them, stop reading them as you. Antidote washes a protected picture the ways others might, to see how much survives, and inspects pictures for cloaks. Part of Atelier, Cursed Entertainment's creative suite."));
        ImGui::Spacing();
        ImGui::TextUnformatted(T("Methods: PGD (Madry et al.), MI-FGSM (Dong et al.), EOT (Athalye et al.), input-transformation defences and feature squeezing (as in IBM's Adversarial Robustness Toolbox, MIT). Recogniser: ArcFace w600k_r50 (InsightFace, non-commercial research use). Run by ONNX Runtime with DirectML."));
        ImGui::Spacing();
        ImGui::TextUnformatted(T("Only protect pictures of yourself or of people who agreed. A cloak is a speed bump, not a guarantee: new models may see through it."));
        ImGui::PopTextWrapPos();
        ImGui::TextDisabled("%s %s", T("Running on the"), Ort::device().c_str());
        ImGui::Spacing();
        ImGui::TextUnformatted("Farica\xe2\x98\x86Kimora \xc2\xa9 Cursed Entertainment");
        if (ImGui::Button(T("Close"), ImVec2(-1, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

static void frame() {
    std::vector<std::function<void()>> todo;
    { std::lock_guard<std::mutex> g(mtx); todo.swap(onMain); }
    for (auto& f : todo) f();
    if (originalStale) { upload(originalTex, original); originalStale = false; }
    if (resultStale) { upload(resultTex, result); resultStale = false; }
    if (diffStale) { makeDiffs(); upload(heatTex, heatImg); upload(noiseTex, noiseImg); diffStale = false; }
    if (decoyStale) { upload(decoyTex, decoyThumb); decoyStale = false; }
    keys();
    menuBar();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float side = 360, top = ImGui::GetFrameHeight();
    leftPanel(ImVec2(vp->Pos.x, vp->Pos.y + top), ImVec2(side, vp->Size.y - top));
    canvas(ImVec2(vp->Pos.x + side, vp->Pos.y + top), ImVec2(vp->Size.x - side, vp->Size.y - top));
    about();
}

int main(int argc, char** argv) {
    std::string file, capture;
    bool autoProtect = false, autoTest = false, autoInspect = false, autoPurify = false, aboutAtStart = false;
    std::vector<std::string> args;
#ifdef _WIN32
    { int n; LPWSTR* wa = CommandLineToArgvW(GetCommandLineW(), &n); if (wa) { for (int i = 1; i < n; ++i) args.push_back(fs::path(wa[i]).u8string()); LocalFree(wa); } }
#else
    for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
#endif
    (void)argc; (void)argv;
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) return 1;
    for (size_t i = 0; i < args.size(); ++i) {
        const bool more = i + 1 < args.size();
        if (args[i] == "--capture" && more) capture = args[++i];
        else if (args[i] == "--protect") autoProtect = true;
        else if (args[i] == "--test") autoTest = true;
        else if (args[i] == "--inspect") autoInspect = true;
        else if (args[i] == "--purify") autoPurify = true;
        else if (args[i] == "--about") aboutAtStart = true;
        else if (args[i] == "--tab" && more) { tab = args[++i] == "antidote" ? 1 : 0; forceTab = true; }
        else if (args[i] == "--view" && more) view = std::clamp(atoi(args[++i].c_str()), 0, V_COUNT - 1);
        else if (args[i] == "--strength" && more) cloakSt.strength = (float)atof(args[++i].c_str());
        else if (args[i] == "--steps" && more) cloakSt.steps = atoi(args[++i].c_str());
        else if (args[i] == "--lang" && more) Lang::load(exeDir() + "lang", args[++i]);
        else file = args[i];
    }
    if (Lang::current() == "en") {   // the language chosen last time
        std::ifstream p(fs::u8path(prefsFile())); std::string line;
        while (std::getline(p, line)) if (line.rfind("language=", 0) == 0) Lang::load(exeDir() + "lang", line.substr(9));
    }
    status = T("Open or drop a picture of yourself");
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    const Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_HIDDEN;   // shown after the splash
    window = SDL_CreateWindow("Inoculant", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1500, 920, flags);
    if (!window) return 1;
    SDL_SetWindowMinimumSize(window, 1000, 640);
    SDL_GLContext gl = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl);
    SDL_GL_SetSwapInterval(1);
    SDL_Surface* icon = SDL_LoadBMP((exeDir() + "assets/icon.bmp").c_str());
    if (icon) SDL_SetWindowIcon(window, icon);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    loadFonts();
    applyStyle();

    // the splash: a small borderless window sharing the main window's OpenGL context, up while the models load
    SDL_Window* splashWin = nullptr;
    if (capture.empty()) {
        splashWin = SDL_CreateWindow("Inoculant", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)Splash::W, (int)Splash::H, SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS);
        if (splashWin && SDL_GL_MakeCurrent(splashWin, gl) != 0) { SDL_DestroyWindow(splashWin); splashWin = nullptr; SDL_GL_MakeCurrent(window, gl); }
        if (splashWin && icon) SDL_SetWindowIcon(splashWin, icon);
    }
    if (icon) SDL_FreeSurface(icon);
    ImGui_ImplSDL2_InitForOpenGL(splashWin ? splashWin : window, gl);
    ImGui_ImplOpenGL3_Init("#version 330");
    SDL_EventState(SDL_DROPFILE, SDL_ENABLE);

    static std::atomic<int> loaded{0};
    const int toLoad = 4;   // the detector, the landmarkers, ArcFace and its gradient graph
    worker = std::thread(workerLoop);
    queue([] {
        std::string e;
        if (!Ort::init(true, e)) { say(e); modelsReady = true; return; }
        auto count = [](const std::string& s) { say(s); if (s.rfind("Loading ", 0) == 0) ++loaded; };
        if (!Cloak::ready(e, count)) { say(e); modelsReady = true; return; }
        // the face finder too, so the first picture opens at once
        for (const char* m : {"retinaface_10g.onnx", "fan_68_5.onnx", "2dfan4.onnx"}) if (!Ort::open(m, e, count)) { say(e); break; }
        say(std::string(T("Ready, on the ")) + Ort::device());
        modelsReady = true;
    });
    if (splashWin) {
        const Uint32 start = SDL_GetTicks();
        bool skip = false;
        for (;;) {
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                ImGui_ImplSDL2_ProcessEvent(&e);
                if (e.type == SDL_QUIT) quit = true;
                if (e.type == SDL_MOUSEBUTTONDOWN || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)) skip = true;
            }
            const Uint32 t = SDL_GetTicks() - start;
            if (skip || quit || (modelsReady && t >= 1500) || t >= 30000) break;
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL2_NewFrame();
            ImGui::NewFrame();
            std::string line;
            { std::lock_guard<std::mutex> g(mtx); line = status; }
            const float p = modelsReady ? 1.0f : 0.05f + 0.9f * std::min(1.0f, (float)loaded / toLoad);
            Splash::draw(ImGui::GetBackgroundDrawList(), 0, 0, modelsReady ? T("Ready") : line.c_str(), p);
            ImGui::Render();
            int sw, sh;
            SDL_GL_GetDrawableSize(splashWin, &sw, &sh);
            glViewport(0, 0, sw, sh);
            glClearColor(0.1f, 0.1f, 0.1f, 1);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            SDL_GL_SwapWindow(splashWin);
        }
        ImGui_ImplSDL2_Shutdown();
        SDL_DestroyWindow(splashWin);
        SDL_GL_MakeCurrent(window, gl);
        ImGui_ImplSDL2_InitForOpenGL(window, gl);
    }
    if (capture.empty()) { SDL_ShowWindow(window); SDL_MaximizeWindow(window); SDL_RaiseWindow(window); }
    if (!file.empty()) loadPicture(file);
    showAbout = aboutAtStart;

    int frames = 0, quietFrames = 0;
    std::vector<std::string> drops;
    while (!quit) {
        SDL_Event e;
        drops.clear();
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if (e.type == SDL_QUIT) quit = true;
            if (e.type == SDL_DROPFILE) { drops.push_back(e.drop.file); SDL_free(e.drop.file); }
        }
        if (!drops.empty()) {
            // on the Antidote tab with a picture open: a protected copy to test; otherwise the picture
            if (tab == 1 && !original.empty()) loadTestPicture(drops.back()); else loadPicture(drops.back());
        }
        // the command line's steps, one after another once the worker is free
        if (modelsReady && idle() && progress < 0 && !original.empty()) {
            bool empty; { std::lock_guard<std::mutex> g(mtx); empty = onMain.empty(); }
            if (empty) {
                if (autoProtect) { autoProtect = false; runProtect(); }
                else if (autoTest && !protectedPic.empty()) { autoTest = false; runTest(); }
                else if (autoPurify) { autoPurify = false; runPurify(); }
                else if (autoInspect) { autoInspect = false; runInspect(); }
            }
        }
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        frame();
        ImGui::Render();
        int w, h;
        SDL_GL_GetDrawableSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.1f, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        ++frames;
        if (!capture.empty()) {
            bool quiet; { std::lock_guard<std::mutex> g(mtx); quiet = jobs.empty() && onMain.empty(); }
            const bool pending = autoProtect || autoPurify || autoInspect || (autoTest && !protectedPic.empty());
            quietFrames = (modelsReady && quiet && !busy && progress < 0 && !pending && (file.empty() || !original.empty())) ? quietFrames + 1 : 0;
            if (frames >= 8 && quietFrames >= 8) {
                std::vector<unsigned char> px((size_t)w * h * 4);
                glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
                std::vector<unsigned char> flip(px.size());
                for (int y = 0; y < h; ++y) memcpy(&flip[(size_t)y * w * 4], &px[(size_t)(h - 1 - y) * w * 4], (size_t)w * 4);
                SDL_Surface* s = SDL_CreateRGBSurfaceWithFormatFrom(flip.data(), w, h, 32, w * 4, SDL_PIXELFORMAT_ABGR8888);
                SDL_SaveBMP(s, capture.c_str());
                SDL_FreeSurface(s);
                break;
            }
        }
        SDL_GL_SwapWindow(window);
    }
    stopJob = true;
    workerQuit = true;
    if (worker.joinable()) worker.join();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
