#pragma once

#include "config.h"
#include <random>

NAMESPACE_BEGIN

inline auto& random_generator() {
		static std::random_device rd;
		static std::mt19937 g(rd());
		return g;
}

NAMESPACE_END
