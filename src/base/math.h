#pragma once

#include "config.h"

#include <cmath>

NAMESPACE_BEGIN

inline real_t sigmoid(real_t x) {
		return real_t(1.0) / (real_t(1.0) + std::exp(-x));
}

NAMESPACE_END
