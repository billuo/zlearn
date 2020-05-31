#include <catch2/catch.hpp>

#include "base/logger.h"
#include "data/sampler.h"

using namespace NAMESPACE_NAME;

TEST_CASE("composite sampler") {
		logger::initialize();
		try {
				std::vector<std::shared_ptr<Sampler>> samplers;
				auto data = std::make_shared<DataSet>();
				data->add_entry();
				data->add_entry();
				data->add_entry();
				samplers.push_back(Sampler::create(data));
				samplers.push_back(Sampler::create(data));
				auto composite_sampler = Sampler::create(samplers);

				Entries e;
				REQUIRE(composite_sampler->get_samples(-1, e) == 6);
				composite_sampler->restart();
				REQUIRE(composite_sampler->get_samples(7, e) == 6);
				composite_sampler->restart();

				for (size_t i = 0; i <= 6; ++i) {
						REQUIRE(composite_sampler->get_samples(i, e) == i);
						composite_sampler->restart();
				}

		} catch (const std::exception& e) {
				logger::error("{}", e.what());
				throw;
		}
}
