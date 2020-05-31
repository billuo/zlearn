#pragma once

#include "base/common.h"

NAMESPACE_BEGIN

struct Application_impl;
class Application {
	public:
		Application();
		~Application();
		void parse_options(int argc, char** argv);

		int run();

	private:
		std::unique_ptr<Application_impl> m_impl;
};

NAMESPACE_END
