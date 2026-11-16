// Inoculant - the gradient graph. ONNX Runtime only runs models forwards, but a cloak needs to know which way each
// pixel should move to change what a face recogniser sees: the gradient of its answer with respect to the picture.
// So Inoculant reads the recogniser's .onnx file itself and writes a second one that also runs it backwards
// (reverse-mode differentiation, written out op by op), once, on this PC:
//
//   inputs   x  [N,3,H,W]   the face crops, as the model takes them
//            t  [1,D]       a unit-length target embedding
//   outputs  the model's own embedding e [N,D]
//            cos  [N,1]     <e/|e|, t>: how alike the crop and the target are to the model
//            grad [N,3,H,W] d cos / d x
//
// The ops handled are the ones ResNet-style recognisers use (ArcFace's IResNet): Conv, BatchNormalization, PRelu,
// Relu, Add, Flatten, Reshape, Gemm, MatMul, Identity. A model with anything else is refused with its op's name.
// tools/make_grad.py is the same builder in Python, for checking against.
#pragma once
#include <string>

namespace GradGraph {
// reads src (a forward model), writes dst (the gradient graph); err says why not
bool build(const std::string& src, const std::string& dst, std::string& err);
}
