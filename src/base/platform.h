#pragma once

#include "config.h"

#include <cstdint>

NAMESPACE_BEGIN

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using c8 = char;
using c16 = char16_t;
using c32 = char32_t;

/// attribute to enable printf style formatting checks
#if defined(__GNUC__)
#define ATTR_FORMAT_PRINTF(m, n) __attribute__((format(printf, m, n)))
#define ATTR_FORMAT_SCANF(m, n) __attribute__((format(scanf, m, n)))
#else
#define ATTR_FORMAT_PRINTF(m, n)
#define ATTR_FORMAT_SCANF(m, n)
#define
#endif

/// @brief branch prediction hints
#if defined(__GNUC__)
#define LIKELY(x) __builtin_expect(!!(x), true)
#define UNLIKELY(x) __builtin_expect(!!(x), false)
#else
#define LIKELY(x) x
#define UNLIKELY(x) x
#endif

#if defined(__PRETTY_FUNCTION__)
#define FUNCTION_NAME __PRETTY_FUNCTION__
#else
#define FUNCTION_NAME __FUNCTION__
#endif

NAMESPACE_END
