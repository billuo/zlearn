#include "lm.h"
#include "base/logger.h"
#include "data/sampler.h"

NAMESPACE_BEGIN

void LM::initialize(Sampler& sampler, std::vector<real_t> extras) {
		m_extras = extras;
		size_t n_extras = extras.size();

		b.resize(1 + n_extras);
		b[0] = 0;
		for (size_t i = 1; i <= n_extras; ++i) {
				b[i] = extras[i - 1];
		}

		Entries entries;
		size_t max_feature_id = 0;
		size_t max_field_id = 0;
		while (sampler.get_samples(-1, entries)) {
				max_feature_id =
					std::max(max_feature_id, DataSet::max_feature_id(entries));
				max_field_id =
					std::max(max_field_id, DataSet::max_field_id(entries));
				entries.clear();
		}
		check_feature_id(max_feature_id);
		check_field_id(max_field_id);

		logger::info("LM initialized; {} features", n);
}

real_t LM::predict(Entry& entry) {
		real_t inv_norm = std::sqrt(entry.get_inv_norm2());

		real_t y_l = 0; // linear part
		for (auto& f : entry.features) {
				if (f.id >= n) continue;
				y_l += w.row(f.id).coeff(0) * f.value;
		}
		y_l = y_l * inv_norm + b.coeff(0);
		return y_l;
}

void LM::check_feature_id(size_t id) {
		if (id < n) return;
		size_t n_extras = m_extras.size();
		w.conservativeResize(id + 1, 1 + n_extras);
		auto W = w.bottomRows(id + 1 - n);
		W.col(0).fill(0);
		for (size_t i = 1; i <= n_extras; ++i) {
				W.col(i).fill(m_extras[i - 1]);
		}
		n = id + 1;
}
void LM::check_field_id(size_t) {}

NAMESPACE_END
