#include "application.h"
#include "base/logger.h"

int main(int argc, char* argv[]) {
		using namespace zlearn;
		logger::initialize();
		try {
				Application application;
				application.parse_options(argc, argv);
				return application.run();
		} catch (const std::exception& e) {
				logger::critical("{}", e.what());
				return -1;
		}
}
