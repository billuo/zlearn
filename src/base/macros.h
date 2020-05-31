#pragma once

#define EXPAND_MACRO(x) x
#define CONCAT_(a, b) EXPAND_MACRO(a##b)
#define CONCAT(a, b) CONCAT_(a, b)
#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
#define UNUSED(x) (void) (x)
#define ONCE(stat)                                                             \
		do {                                                                   \
				static bool _b = false;                                        \
				if (!_b) {                                                     \
						_b = true;                                             \
						stat;                                                  \
				}                                                              \
		} while (0)
