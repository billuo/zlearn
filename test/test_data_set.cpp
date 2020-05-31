#include <catch2/catch.hpp>

#include "base/logger.h"
#include "data/data_set.h"

using namespace NAMESPACE_NAME;

TEST_CASE("serialization & deserialization") {
		logger::initialize();
		try {
				DataSet m;
				m.has_label(true);
				std::shared_ptr<Entry> row;

				row = m.add_entry();
				row->label = 0;
				row->features.emplace_back(1, 1, 1);
				row->features.emplace_back(2, 4, 0.5);
				row->set_normalize(true);

				row = m.add_entry();
				row->label = 1;
				row->features.emplace_back(1, 2, 1);
				row->features.emplace_back(2, 5, 0.3);
				row->set_normalize(true);

				row = m.add_entry();
				row->label = 0;
				row->features.emplace_back(1, 3, 1);
				row->features.emplace_back(2, 6, 0.1);
				row->set_normalize(true);

				m.serialize("data.out");

				m = DataSet();
				m.deserialize("data.out");
				REQUIRE(m[0]->label == 0);
				REQUIRE(m[0]->get_inv_norm2() == real_t(0.89442719));
				REQUIRE(m(0, 0) == Feature(1, 1, 1));
				REQUIRE(m(0, 1) == Feature(2, 4, 0.5));
				REQUIRE(m[1]->label == 1);
				REQUIRE(m(1, 0) == Feature(1, 2, 1));
				REQUIRE(m(1, 1) == Feature(2, 5, 0.3));
				REQUIRE(m[2]->label == 0);
				REQUIRE(m(2, 0) == Feature(1, 3, 1));
				REQUIRE(m(2, 1) == Feature(2, 6, 0.1));
		} catch (const std::exception& e) {
				logger::error("{}", e.what());
				throw;
		}
}

TEST_CASE("parse from text") {
		logger::initialize();
		try {
				auto ffm_with_label =
					DataSet::from_file("ffm_with_label.txt", false, " ");
				auto ffm_no_label =
					DataSet::from_file("ffm_no_label.txt", false, " ");
				REQUIRE(ffm_with_label->has_label());
				REQUIRE(!ffm_no_label->has_label());
		} catch (const std::exception& e) {
				logger::error("{}", e.what());
				throw;
		}
		try {
				// separators not specified
				auto ffm_with_label =
					DataSet::from_file("ffm_with_label.txt", false);
				REQUIRE(ffm_with_label->has_label());
		} catch (const std::exception& e) {
				logger::error("{}", e.what());
				throw;
		}
}
