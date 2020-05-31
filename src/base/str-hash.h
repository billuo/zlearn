#pragma once

#include "str.h"

#include <functional>

namespace std {
template <>
struct hash<zlearn::String> {
		size_t operator()(const zlearn::String& s) const noexcept {
				return s.hash();
		}
};
} // namespace std
