#pragma once

#include "base/common.h"
#include "base/math.h"

#include <optional>
#include <vector>

NAMESPACE_BEGIN

struct Feature {
		Feature() = default;
		Feature(const Feature&) = default;
		Feature(Feature&&) = default;
		Feature& operator=(const Feature&) = default;
		Feature& operator=(Feature&&) = default;
		Feature(size_t id, real_t value) : id(id), value(value) {
				normalized_value = value;
		}
		Feature(size_t field_id, size_t id, real_t value)
		: field_id(field_id), id(id), value(value) {
				normalized_value = value;
		}

		size_t field_id = 0;
		size_t id = 0;
		real_t value = 0;
		real_t normalized_value; // could be set by containing Entry

		bool operator==(const Feature& rhs) const;
		bool operator!=(const Feature& rhs) const { return !(rhs == *this); }
};
using FeatureVector = std::vector<Feature>;
inline bool Feature::operator==(const Feature& rhs) const {
		return field_id == rhs.field_id && id == rhs.id && value == rhs.value;
}

struct Entry {
		// {0, 1} for binary classification; ratings for regression, etc.
		real_t label = 0;
		FeatureVector features;
		// cached inverse of norm of feature vector.
		// only has value when normalizing

		void set_normalize(bool b);
		real_t get_inv_norm2() const { return inv_norm2.value_or(1.0); }
		void sort_features();
		bool operator==(const Entry& rhs) const;
		bool operator!=(const Entry& rhs) const { return !(rhs == *this); }

	private:
		std::optional<real_t> inv_norm2;
};
using Entries = std::vector<std::shared_ptr<Entry>>;

inline bool Entry::operator==(const Entry& rhs) const {
		return label == rhs.label && features == rhs.features
			&& inv_norm2 == rhs.inv_norm2;
}

NAMESPACE_END
