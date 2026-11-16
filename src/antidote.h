// Inoculant - the antidotes: what someone might do to a protected picture to wash the protection out, so you can
// see how much of it survives before you share it; and an inspector that guesses whether a picture carries a cloak
// at all. The purifiers are the classic input-transformation defences (as in the Adversarial Robustness Toolbox):
// JPEG compression (Dziugaite et al. 2016; Das et al. 2017), Gaussian blur, a median filter (spatial smoothing),
// bit-depth reduction and shrink-and-enlarge (feature squeezing, Xu et al. 2018), and added noise (Gaussian
// augmentation). The inspector is feature squeezing used as a detector: a clean face's embedding hardly moves when
// the picture is squeezed; a cloaked one jumps, because the cloak lives in exactly the detail squeezing removes.
// NightshadeAntidote's idea (look for the poison's fingerprint in the high frequencies) gives the second clue: how
// much fine noise sits on smooth skin.
#pragma once
#include "faces.h"
#include "image.h"

#include <string>
#include <vector>

namespace Antidote {

enum Filter { F_JPEG, F_BLUR, F_MEDIAN, F_BITS, F_RESIZE, F_NOISE, F_COUNT };
const char* filterName(int f);
const char* filterHint(int f);

struct Settings {
    bool on[F_COUNT] = {true, false, false, false, false, false};
    float jpeg = 75;       // quality 10..100
    float blur = 1.0f;     // sigma, pixels
    float median = 1;      // radius, pixels
    float bits = 5;        // bits per channel 1..8
    float resize = 50;     // shrink to this % and back
    float noise = 4;       // sigma, levels of 255
    int seed = 7;
};

Img apply(const Img& im, int filter, const Settings& st);
Img purify(const Img& im, const Settings& st);   // the ones switched on, in this order

// how much of the protection each purifier leaves
struct Row {
    std::string name;
    float purifiedSim = -1;   // purified protected face vs the clean face (higher: more washed out)
    float baselineSim = -1;   // purified clean face vs the clean face (what the purifier does on its own)
    float kept = 0;           // share of the protection left, 0..1+
};
struct Report {
    float protectedSim = -1;  // the protected face vs the clean one, before any purifier
    std::vector<Row> rows;    // each purifier switched on, then all of them chained ("Purify")
};
// clean: the original; prot: the protected picture. The first face found is compared.
bool test(const Img& clean, const Img& prot, const Settings& st, Report& rep, std::string& err);

// the inspector: no original needed
struct Inspection {
    int faces = 0;
    float squeeze = 0;        // the largest embedding change under squeezing (1 - similarity)
    float residual = 0;       // fine noise on smooth areas of the face, levels of 255 (RMS)
    float strength = 0;       // a guess at how strong a cloak is: 0 none .. 1 strong
    float score = 0;          // 0 clean .. 1 surely cloaked
    std::string verdict;
};
bool inspect(const Img& im, Inspection& out, std::string& err);

}  // namespace Antidote
