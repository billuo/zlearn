#pragma once

#include "str.h"

#include <exception>

NAMESPACE_BEGIN

class Exception : public std::exception {
	public:
		explicit Exception(source_location loc, const char* fmt, ...)
			ATTR_FORMAT_PRINTF(3, 4)
		: msg(String::printf("%s:%d %s: ", loc.file, loc.line, loc.func)) {
				va_list ap;
				va_start(ap, fmt);
				msg.vappendf(fmt, ap);
				va_end(ap);
		}
		explicit Exception(const char* fmt, ...) ATTR_FORMAT_PRINTF(2, 3)
		: msg() {
				va_list ap;
				va_start(ap, fmt);
				msg.vappendf(fmt, ap);
				va_end(ap);
		}
		const char* what() const noexcept override { return msg.c_str(); }

	protected:
		String msg;
};

#if defined(NDEBUG)
#define THROW(fmt, ...) throw ::NAMESPACE_NAME::Exception(fmt, ##__VA_ARGS__);
#else
#define THROW(fmt, ...)                                                        \
		do {                                                                   \
				SOURCE_LOCATION(loc);                                          \
				throw ::NAMESPACE_NAME::Exception(loc, fmt, ##__VA_ARGS__);    \
		} while (0)
#endif

#define ASSERT_(expr)                                                          \
		do {                                                                   \
				if (!(expr)) {                                                 \
						SOURCE_LOCATION(loc);                                  \
						throw ::NAMESPACE_NAME::Exception(                     \
							loc, "assertion failed: '%s'", #expr);             \
				}                                                              \
		} while (0)
#define RELEASE_ASSERT(expr) ASSERT_(expr)
#if defined(NDEBUG)
#define ASSERT(expr) (void) sizeof(expr)
#else
#define ASSERT(expr) ASSERT_(expr)
#endif

#if defined(NDEBUG)
#define UNREACHABLE(...) __builtin_unreachable()
#else
#define UNREACHABLE(...) THROW(__VA_ARGS__)
#endif

#define UNIMPLEMENTED(fmt, ...) THROW("unimplemented; " fmt, ##__VA_ARGS__)

NAMESPACE_END
