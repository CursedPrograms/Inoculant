// Inoculant - the splash: the logo's shield drawn large (grey for the picture, Atelier blue for its cloak, a grain
// of noise on the blue), a face's outline in it, the name, the version, the credits and what's loading. Solid
// colours only, drawn with ImGui's draw list so it's sharp at any size.
#include "splash.h"
#include "lang.h"

#include <imgui.h>

#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

#ifndef INOCULANT_VERSION
#define INOCULANT_VERSION "0.1.0"
#endif

namespace Splash {

static ImFont *titleFont = nullptr, *smallFont = nullptr;

void setFonts(ImFont* title, ImFont* small) { titleFont = title; smallFont = small; }

static const ImU32 BG = IM_COL32(0x1a, 0x1a, 0x1a, 255), LINE = IM_COL32(0x3a, 0x3a, 0x3a, 255),
                   TEXT = IM_COL32(0xe6, 0xe6, 0xe6, 255), DIM = IM_COL32(0x9a, 0x9a, 0x9a, 255),
                   BLUE = IM_COL32(0x00, 0x7b, 0xb7, 255), DEEP = IM_COL32(0x00, 0x51, 0x84, 255),
                   LIGHT = IM_COL32(0x34, 0xa9, 0xe3, 255), DARK = IM_COL32(0x3a, 0x3a, 0x3a, 255),
                   FACE = IM_COL32(0xe6, 0xe6, 0xe6, 255);

static void text(ImDrawList* dl, ImFont* f, float size, ImVec2 p, ImU32 col, const char* s) {
    if (!f) f = ImGui::GetFont();
    dl->AddText(f, size, p, col, s);
}
static float textWidth(ImFont* f, float size, const char* s) {
    if (!f) f = ImGui::GetFont();
    return f->CalcTextSizeA(size, FLT_MAX, 0, s).x;
}

// the shield's outline: a flat top, straight sides, curving to a point
static std::vector<ImVec2> shieldOutline(ImVec2 c, float r) {
    std::vector<ImVec2> p;
    const float top = c.y - r * 1.05f, side = c.y + r * 0.15f;
    auto bez = [&](ImVec2 a, ImVec2 b, ImVec2 cc, ImVec2 d) {
        for (int i = 1; i <= 24; ++i) {
            const float t = i / 24.0f, u = 1 - t;
            p.push_back(ImVec2(u * u * u * a.x + 3 * u * u * t * b.x + 3 * u * t * t * cc.x + t * t * t * d.x,
                               u * u * u * a.y + 3 * u * u * t * b.y + 3 * u * t * t * cc.y + t * t * t * d.y));
        }
    };
    p.push_back(ImVec2(c.x - r, top)); p.push_back(ImVec2(c.x + r, top)); p.push_back(ImVec2(c.x + r, side));
    bez(ImVec2(c.x + r, side), ImVec2(c.x + r, c.y + r * 0.7f), ImVec2(c.x + r * 0.45f, c.y + r * 1.0f), ImVec2(c.x, c.y + r * 1.2f));
    bez(ImVec2(c.x, c.y + r * 1.2f), ImVec2(c.x - r * 0.45f, c.y + r * 1.0f), ImVec2(c.x - r, c.y + r * 0.7f), ImVec2(c.x - r, side));
    return p;
}
static bool inside(const std::vector<ImVec2>& poly, float x, float y) {
    bool in = false;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
        if ((poly[i].y > y) != (poly[j].y > y) && x < (poly[j].x - poly[i].x) * (y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x) in = !in;
    return in;
}

static void shield(ImDrawList* dl, ImVec2 c, float r) {
    const std::vector<ImVec2> outline = shieldOutline(c, r);
    dl->AddConvexPolyFilled(outline.data(), (int)outline.size(), DARK);
    // the right half in blue: the same outline, clipped to the right
    dl->PushClipRect(ImVec2(c.x, c.y - r * 2), ImVec2(c.x + r * 2, c.y + r * 2), true);
    dl->AddConvexPolyFilled(outline.data(), (int)outline.size(), BLUE);
    dl->PopClipRect();
    // the cloak's grain: a fixed scatter of small squares, lighter and deeper blue, inside the blue half
    unsigned s = 12345u;
    for (int i = 0; i < 260; ++i) {
        s = s * 1103515245u + 12345u; const float u = ((s >> 8) & 0xffff) / 65535.0f;
        s = s * 1103515245u + 12345u; const float v = ((s >> 8) & 0xffff) / 65535.0f;
        const float x = c.x + u * r, y = c.y - r * 1.0f + v * r * 2.1f, q = r * 0.035f;
        if (!inside(outline, x, y) || !inside(outline, x + q, y + q)) continue;
        dl->AddRectFilled(ImVec2(x, y), ImVec2(x + q, y + q), (i & 1) ? LIGHT : DEEP);
    }
    // a face, drawn in one line across both halves: the head, the eyes, a mouth
    const float fr = r * 0.48f;
    const ImVec2 fc(c.x, c.y - r * 0.12f);
    dl->AddCircle(fc, fr, FACE, 64, r * 0.045f);
    dl->AddCircleFilled(ImVec2(fc.x - fr * 0.38f, fc.y - fr * 0.15f), fr * 0.11f, FACE, 24);
    dl->AddCircleFilled(ImVec2(fc.x + fr * 0.38f, fc.y - fr * 0.15f), fr * 0.11f, FACE, 24);
    dl->AddLine(ImVec2(fc.x - fr * 0.35f, fc.y + fr * 0.42f), ImVec2(fc.x + fr * 0.35f, fc.y + fr * 0.42f), FACE, r * 0.045f);
}

void draw(ImDrawList* dl, float x, float y, const char* status, float progress) {
    const ImVec2 o(x, y), end(x + W, y + H);
    dl->PushClipRect(o, end, true);
    dl->AddRectFilled(o, end, BG);

    // the art: the shield large, on the right
    shield(dl, ImVec2(x + 500, y + 170), 120);

    // the name: "Inocu" in the text colour, "lant" in blue
    const float ts = 50, tx = x + 40, ty = y + 92;
    text(dl, titleFont, ts, ImVec2(tx, ty), TEXT, "Inocu");
    text(dl, titleFont, ts, ImVec2(tx + textWidth(titleFont, ts, "Inocu"), ty), BLUE, "lant");
    text(dl, nullptr, 16, ImVec2(tx + 2, ty + 62), DIM, (std::string(T("Version")) + " " INOCULANT_VERSION).c_str());
    text(dl, nullptr, 16, ImVec2(tx + 2, ty + 84), DIM, T("Poisons and antidotes for your own pictures"));

    // the credits, small
    const float cs = 13, cy = y + 306;
    text(dl, smallFont, cs, ImVec2(tx, cy), DIM, "Methods: PGD, MI-FGSM, EOT, feature squeezing (ART). Model: ArcFace (InsightFace).");
    text(dl, smallFont, cs, ImVec2(tx, cy + 17), DIM, "Built with Dear ImGui, SDL2, stb and ONNX Runtime with DirectML.");
    text(dl, smallFont, cs, ImVec2(tx, cy + 34), DIM, "Farica\xe2\x98\x86Kimora \xc2\xa9 Cursed Entertainment");

    // what's loading, and how far along
    if (status) text(dl, nullptr, 16, ImVec2(tx, y + 368), TEXT, status);
    if (progress >= 0) {
        const float p = progress > 1 ? 1 : progress;
        dl->AddRectFilled(ImVec2(x, end.y - 3), end, LINE);
        dl->AddRectFilled(ImVec2(x, end.y - 3), ImVec2(x + W * p, end.y), BLUE);
    }
    dl->AddRect(o, end, LINE);
    dl->PopClipRect();
}

}  // namespace Splash
