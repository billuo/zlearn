#include "logger.h"
#include "str.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

static void flush_all_loggers() {
		spdlog::apply_all(
			[](const std::shared_ptr<spdlog::logger>& l) { l->flush(); });
}

namespace NAMESPACE_NAME::logger {

std::shared_ptr<spdlog::logger> console;

void initialize() {
		using namespace spdlog;
		std::atexit(flush_all_loggers);
		std::at_quick_exit(flush_all_loggers);
		String filename = "zlearn.log";
		try {
				auto console_sink =
					std::make_shared<sinks::ansicolor_stdout_sink_mt>();
				auto file_sink = std::make_shared<sinks::basic_file_sink_mt>(
					filename.c_str());
				console =
					std::make_shared<spdlog::logger>("console", console_sink);
				set_default_logger(console);
				console->set_pattern("(tid %=6t)%^[%=10l]%$ %v");
#if defined(NDEBUG)
				console->set_level(level::info);
#else
				console->set_level(level::debug);
#endif
				flush_every(std::chrono::seconds(5));
		} catch (const std::exception& e) {
				fprintf(stderr, "failed to initialize logger: %s\n", e.what());
				std::exit(EXIT_FAILURE);
		}
}

NAMESPACE_END
