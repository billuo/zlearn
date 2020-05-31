#pragma once

#include "common.h"

#include <spdlog/logger.h>

#if defined(LOGGER_OSTREAM)
#include <spdlog/fmt/ostr.h>
#endif

NAMESPACE_BEGIN

namespace logger {

void initialize();

extern std::shared_ptr<spdlog::logger> console;

template <typename... Args>
void trace(Args&&... args) {
		console->trace(std::forward<Args>(args)...);
}
template <typename... Args>
void debug(Args&&... args) {
		console->debug(std::forward<Args>(args)...);
}
template <typename... Args>
void info(Args&&... args) {
		console->info(std::forward<Args>(args)...);
}
template <typename... Args>
void warn(Args&&... args) {
		console->warn(std::forward<Args>(args)...);
}
template <typename... Args>
void error(Args&&... args) {
		console->error(std::forward<Args>(args)...);
}
template <typename... Args>
void critical(Args&&... args) {
		console->critical(std::forward<Args>(args)...);
}

} // namespace logger

NAMESPACE_END
