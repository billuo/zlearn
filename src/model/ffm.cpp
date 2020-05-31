#include "ffm.h"
#include "base/logger.h"
#include "base/random.h"
#include "data/sampler.h"

NAMESPACE_BEGIN

void FFM::initialize(Sampler& sampler, std::vector<real_t> extra) {
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

		logger::info("FFM initialized; "
					 "{} features, {} fields, k={}",
					 n, n_f, k);
}

real_t FFM::predict(Entry& entry) {
		const auto& features = entry.features;
		real_t inv_norm2 = entry.get_inv_norm2();
		real_t inv_norm = std::sqrt(inv_norm2);

		real_t y_l = 0;      // linear part
		Vector<Dynamic> y_v; // latent part
		y_v.setZero(k);
		for (auto it1 = features.begin(); it1 != features.end(); ++it1) {
				auto& f1 = *it1;
				if (f1.id >= n) continue;
				y_l += w.row(f1.id).coeff(0) * f1.value;
				for (auto it2 = it1 + 1; it2 != features.end(); ++it2) {
						auto& f2 = *it2;
						if (f2.id >= n) continue;
						auto v_i_fj = get_v(f1.id, f2.field_id);
						auto v_j_fi = get_v(f2.id, f1.field_id);
						y_v +=
							f1.value * f2.value * v_i_fj.cwiseProduct(v_j_fi);
				}
		}
		return inv_norm2 * y_v.sum() + inv_norm * y_l + b.coeff(0);
}

void FFM::check_feature_id(size_t id) {
		if (id < n) return;
		size_t n_extras = m_extras.size();

		w.conservativeResize(id + 1, 1 + n_extras);
		auto W = w.bottomRows(id + 1 - n);
		W.col(0).fill(0);
		for (size_t i = 1; i <= n_extras; ++i) {
				W.col(i).fill(m_extras[i - 1]);
		}

		v.conservativeResize(id + 1, k * (1 + n_extras) * n_f);
		auto V = v.bottomRows(id + 1 - n);
		auto G = random_generator();
		std::uniform_real_distribution<real_t> dist(0.0, 0.66 / std::sqrt(k));
		for (index_t r = 0; r < V.rows(); ++r) {
				for (size_t f = 0; f < n_f; ++f) {
						for (index_t c = 0; c < k; ++c) {
								V(f * k * (1 + n_extras), c) = dist(G);
						}
				}
		}
		for (size_t f = 0; f < n_f; ++f) {
				for (size_t i = 1; i <= n_extras; ++i) {
						V.middleCols(f * k * (1 + n_extras) + i * k, k)
							.fill(m_extras[i - 1]);
				}
		}
		n = id + 1;
}
void FFM::check_field_id(size_t id) {
		if (id < n_f) return;
		size_t n_extras = m_extras.size();

		v.conservativeResize(n, k * (1 + n_extras) * (id + 1));
		auto G = random_generator();
		std::uniform_real_distribution<real_t> dist(0.0, 0.66 / std::sqrt(k));
		for (index_t r = 0; r < v.rows(); ++r) {
				for (size_t f = n_f; f <= id; ++f) {
						for (index_t c = 0; c < k; ++c) {
								v(r, f * k * (1 + n_extras) + c) = dist(G);
						}
				}
		}
		for (size_t f = n_f; f <= id; ++f) {
				for (size_t i = 1; i <= n_extras; ++i) {
						v.middleCols(f * k * (1 + n_extras) + i * k, k)
							.fill(m_extras[i - 1]);
				}
		}

		n_f = id + 1;
}

NAMESPACE_END
