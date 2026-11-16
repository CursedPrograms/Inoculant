// Inoculant - the poison: an identity cloak. Small, bounded changes to the pixels of each face, found by projected
// gradient descent (PGD, Madry et al. 2018) with momentum (MI-FGSM, Dong et al. 2018) against a face recogniser
// (ArcFace), so that the recogniser - and the face swappers and identity-driven editors that take its embedding as
// "who this is" (inswapper, SimSwap, HyperSwap, InstantID-style adapters) - no longer read the face as you.
// Robustness comes from Expectation over Transformation (EOT, Athalye et al. 2018): every step looks at the face
// through a few random small shifts, turns, scalings and blurs, so the cloak isn't tied to one exact alignment.
//
// Two goals: Hide (push the face's embedding away from your own) and Decoy (pull it toward another face's).
#pragma once
#include "faces.h"
#include "image.h"

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace Cloak {

enum Goal { G_HIDE, G_DECOY, G_COUNT };

struct Settings {
    int goal = G_HIDE;
    float strength = 4.0f;     // the most any pixel may change, in levels of 255 (L-infinity budget)
    int steps = 40;            // PGD iterations
    float stepSize = 0.5f;     // levels per iteration
    float momentum = 0.9f;     // MI-FGSM: 0 plain PGD; higher carries to other models better
    int samples = 4;           // EOT: views of the face per step (the first is always the plain one)
    float robustness = 0.5f;   // EOT: how far the views are jittered (shift, turn, scale, blur)
    float smoothness = 0.0f;   // the noise's grain: 0 per-pixel, 1 soft blotches (survives JPEG and blur better)
    float texture = 0.6f;      // perceptual masking: 0 the same everywhere, 1 mostly in hair, brows, texture
    float colour = 0.5f;       // 0 light and dark only (R, G, B move together), 1 each colour on its own
    float stopAt = -1.0f;      // Hide: stop early once every face's similarity is below this (-1: run every step)
    float coverage = 1.0f;     // how much of the area round the face may change: 0 the inner face .. 1 the whole crop
    int seed = 1;
};

struct FaceReport {
    float box[4] = {0, 0, 0, 0};
    float before = 1.0f;       // similarity of the clean face to itself (1)
    float after = 1.0f;        // similarity of the protected face to the clean one (what a recogniser sees)
    float decoy = 0.0f;        // Decoy: similarity of the protected face to the decoy
    bool found = true;         // the face was found again in the protected picture
};

struct Result {
    Img out;
    std::vector<FaceReport> faces;
    double psnr = 0, maxDiff = 0, meanDiff = 0;
    int steps = 0;
    double seconds = 0;
};

// what the worker says while it runs: a fraction and a line; return false to stop (what was done is kept)
using Progress = std::function<bool(float, const std::string&)>;

const char* goalName(int g);
DetectSettings detectSettings();    // how Inoculant finds faces (RetinaFace, margin for close-ups)
bool ready(std::string& err, Ort::Say say = nullptr);   // the recogniser and its gradient graph (made once)
std::string gradModelPath();

float cosine(const std::vector<float>& a, const std::vector<float>& b);
// the faces in a picture with their embeddings
bool faces(const Img& im, std::vector<Face>& out, std::string& err);
// the protected picture; decoy: the decoy face's unit embedding (Decoy only)
bool protect(const Img& clean, const Settings& st, const std::vector<float>& decoy, Result& res, std::string& err, Progress progress);
// how alike two pictures' faces are to the recogniser: for each face in a, the best match in b (-1: none found)
std::vector<float> compare(const std::vector<Face>& a, const std::vector<Face>& b);
// the closest face in b to face f by position (the same face after a change), or -1
int sameFace(const Face& f, const std::vector<Face>& b);

}  // namespace Cloak
