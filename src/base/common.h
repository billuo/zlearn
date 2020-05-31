#pragma once

#include "macros.h"
#include "platform.h"

#include <limits>
#include <memory>
#include <utility>

NAMESPACE_BEGIN

template <typename T>
using Limits = std::numeric_limits<T>;

struct source_location {
		constexpr source_location(const char* file,
								  const char* func,
								  int line) noexcept
		: file(file), func(func), line(line) {}
		const char* file;
		const char* func;
		int line;
};
#define SOURCE_LOCATION(name)                                                  \
		::NAMESPACE_NAME::source_location name(__FILE__, FUNCTION_NAME,        \
											   __LINE__)

NAMESPACE_END
