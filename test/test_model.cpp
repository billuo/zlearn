#include <catch2/catch.hpp>

#define LOGGER_OSTREAM

#include "base/logger.h"
#include "model/ffm.h"
#include "model/fm.h"
#include "model/hofm.h"
#include "model/lm.h"
#include "optimizer/optimizer.h"

using namespace NAMESPACE_NAME;

TEST_CASE("LM") {
		logger::initialize();
		try {
				auto data = DataSet::dummy(3, 10, 3);
				auto sampler = Sampler::create(data);
				LM model;
				auto adagrad = std::make_shared<AdaGrad>(0, 0);
				model.initialize(*sampler, adagrad->extras());
				logger::debug("b=\n{}", model.b);
				logger::debug("w=\n{}", model.w);
				model.serialize("lm.bin");
				LM new_model;
				new_model.deserialize("lm.bin");
				logger::debug("b=\n{}", new_model.b);
				logger::debug("w=\n{}", new_model.w);
		} catch (const std::exception& e) {
				logger::error("{}", e.what());
				throw;
		}
}
TEST_CASE("FM") {
		logger::initialize();
		try {
				auto data = DataSet::dummy(3, 10, 3);
				auto sampler = Sampler::create(data);
				FM model(4);
				auto adagrad = std::make_shared<AdaGrad>(0, 0);
				model.initialize(*sampler, adagrad->extras());
				logger::debug("b=\n{}", model.b);
				logger::debug("w=\n{}", model.w);
				logger::debug("v=\n{}", model.v);
				model.serialize("fm.bin");
				FM new_model(0);
				new_model.deserialize("fm.bin");
				logger::debug("b=\n{}", new_model.b);
				logger::debug("w=\n{}", new_model.w);
				logger::debug("v=\n{}", new_model.v);
		} catch (const std::exception& e) {
				logger::error("{}", e.what());
				throw;
		}
}
TEST_CASE("FFM") {
		logger::initialize();
		try {
				auto data = DataSet::dummy(3, 10, 3);
				auto sampler = Sampler::create(data);
				FFM model(4);
				auto adagrad = std::make_shared<AdaGrad>(0, 0);
				model.initialize(*sampler, adagrad->extras());
				logger::debug("b=\n{}", model.b);
				logger::debug("w=\n{}", model.w);
				logger::debug("v=\n{}", model.v);
				model.serialize("ffm.bin");
				FFM new_model(0);
				new_model.deserialize("ffm.bin");
				logger::debug("b=\n{}", new_model.b);
				logger::debug("w=\n{}", new_model.w);
				logger::debug("v=\n{}", new_model.v);
		} catch (const std::exception& e) {
				logger::error("{}", e.what());
				throw;
		}
}
TEST_CASE("HOFM") {
		logger::initialize();
		try {
				auto data = DataSet::dummy(3, 10, 3);
				auto sampler = Sampler::create(data);
				HOFM model(3, 4);
				auto adagrad = std::make_shared<AdaGrad>(0, 0);
				model.initialize(*sampler, adagrad->extras());
				logger::debug("b=\n{}", model.b);
				logger::debug("w=\n{}", model.w);
				logger::debug("v0=\n{}", model.v[0]);
				logger::debug("v1=\n{}", model.v[1]);
				model.serialize("hofm.bin");
				HOFM new_model(0, 0);
				new_model.deserialize("hofm.bin");
				logger::debug("b=\n{}", new_model.b);
				logger::debug("w=\n{}", new_model.w);
				logger::debug("v0=\n{}", new_model.v[0]);
				logger::debug("v1=\n{}", new_model.v[1]);
		} catch (const std::exception& e) {
				logger::error("{}", e.what());
				throw;
		}
}
