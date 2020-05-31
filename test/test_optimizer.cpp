#include <catch2/catch.hpp>

#include "base/logger.h"
#include "optimizer/optimizer.h"

using namespace NAMESPACE_NAME;

TEST_CASE("extras") {
        logger::initialize();
        try {
				auto sgd = std::make_shared<SGD>(0, 0);
				REQUIRE(sgd->extras().empty());
				auto adagrad = std::make_shared<AdaGrad>(0, 0);
				REQUIRE(!adagrad->extras().empty());
				REQUIRE(adagrad->extras()[0] == 1);
        } catch (const std::exception& e) {
                logger::error("{}", e.what());
                throw;
        }
}
