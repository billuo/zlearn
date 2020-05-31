#pragma once

#include <cstddef>

#define PROJECT_NAME zlearn
#define NAMESPACE_NAME PROJECT_NAME
#define NAMESPACE_BEGIN namespace PROJECT_NAME {
#define NAMESPACE_END }

using size_t = std::size_t;

// type of parameter values in a model.
// 32-bit floats should usually be enough.
using real_t = float;
