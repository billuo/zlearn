#include "fm.h"
#include "base/logger.h"
#include "base/random.h"
#include "data/sampler.h"

NAMESPACE_BEGIN

void FM::initialize(Sampler& sampler, std::vector<real_t> extra) {
		m_extras = extra;
		size_t n_extras = m_extras.size();

		b.resize(1 + n_extras);
		b[0] = 0;
		for (size_t i = 1; i <= n_extras; ++i) {
				b[i] = extra[i - 1];
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

		logger::info("FM initialized; {} features, k={}", n, k);
}

real_t FM::predict(Entry& entry) {
		real_t inv_norm2 = entry.get_inv_norm2();
		real_t inv_norm = std::sqrt(inv_norm2);

		real_t y_l = 0; // linear part
		Vector<Dynamic> s = allocate_s();
		for (const auto& f : entry.features) {
				if (f.id >= n) continue;
				y_l += w.row(f.id).coeff(0) * f.value;
				s += v.row(f.id).head(k) * f.value;
		}

		Vector<Dynamic> y_v; // latent part
		y_v.setZero(k);
		for (const auto& f : entry.features) {
				if (f.id >= n) continue;
				auto wx = v.row(f.id).head(k) * f.value;
				y_v += wx.cwiseProduct(s - wx);
		}
		return 0.5 * y_v.sum() * inv_norm2 + y_l * inv_norm + b.coeff(0);
}

void FM::check_feature_id(size_t id) {
		if (id < n) return;
		size_t n_extras = m_extras.size();

		w.conservativeResize(id + 1, 1 + n_extras);
		auto W = w.bottomRows(id + 1 - n);
		W.col(0).fill(0);
		for (size_t i = 1; i <= n_extras; ++i) {
				W.col(i).fill(m_extras[i - 1]);
		}

		v.conservativeResize(id + 1, k * (1 + n_extras));
		auto V = v.bottomRows(id + 1 - n);
		auto G = random_generator();
		std::uniform_real_distribution<real_t> dist(0.0, 0.66 / std::sqrt(k));
		for (index_t r = 0; r < V.rows(); ++r) {
				for (size_t c = 0; c < k; ++c) {
						V(r, c) = dist(G);
				}
		}
		for (size_t i = 1; i <= n_extras; ++i) {
				V.middleCols(k * i, k).fill(m_extras[i - 1]);
		}

		n = id + 1;
}
void FM::check_field_id(size_t) {}

NAMESPACE_END
