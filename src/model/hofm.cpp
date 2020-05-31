#define LOGGER_OSTREAM
#include "hofm.h"
#include "base/logger.h"
#include "base/random.h"

NAMESPACE_BEGIN

Matrix<Dynamic, Dynamic> HOFM::get_dp_eval(Entry& entry, size_t m) const {
		auto& features = entry.features;
		auto end = std::partition(features.begin(), features.end(),
								  [this](auto& f) { return f.id < n; });
		size_t d = end - features.begin();

		Matrix<Dynamic, Dynamic> dp;
		dp.setZero(m + 1, (d + 1) * k);

		dp.row(0).setOnes();
		for (size_t t = 1; t <= m; ++t) {
				for (size_t j = t; j <= d; ++j) {
						auto& f = features[j - 1];
						auto a_j_t = dp.row(t).segment(j * k, k);
						auto a_jm1_t = dp.row(t).segment((j - 1) * k, k);
						auto a_jm1_tm1 = dp.row(t - 1).segment((j - 1) * k, k);
						auto p = get_p(m, f.id);
						a_j_t = a_jm1_t
							+ a_jm1_tm1.cwiseProduct(p) * f.normalized_value;
				}
		}
		return dp;
}

Matrix<Dynamic, Dynamic> HOFM::get_dp_grad(Entry& entry, size_t m) const {
		auto& features = entry.features;
		auto end = std::partition(features.begin(), features.end(),
								  [this](auto& f) { return f.id < n; });
		size_t d = end - features.begin();

		Matrix<Dynamic, Dynamic> dp;
		// there're k redundant columns for more convenient indexing
		dp.setZero(m + 1, (d + 1) * k);

		dp.row(m).tail(k).setOnes();
		for (int t = m - 1; t >= 0; --t) {
				for (int j = d - 1; j >= t; --j) {
						auto& f = features[j]; // NOTE: using x_{j+1}
						auto a_j_t = dp.row(t).segment(j * k, k);
						auto a_jp1_t = dp.row(t).segment((j + 1) * k, k);
						auto a_jp1_tp1 = dp.row(t + 1).segment((j + 1) * k, k);
						a_j_t = a_jp1_t
							+ a_jp1_tp1.cwiseProduct(get_p(m, f.id))
								* f.normalized_value;
				}
		}
		return dp;
}

void HOFM::initialize(Sampler& sampler, std::vector<real_t> extra) {
		v.resize(order - 1);

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

		logger::info("HOFM of {} orders initialized; "
					 "{} features, k={}",
					 order, n, k);
}

real_t HOFM::predict(Entry& entry) {
		real_t inv_norm2 = entry.get_inv_norm2();
		real_t inv_norm = std::sqrt(inv_norm2);
		const auto& features = entry.features;

		real_t y_l = 0; // linear part
		for (auto& f : features) {
				if (f.id >= n) continue;
				y_l += w.row(f.id).coeff(0) * f.value;
		}

		Vector<Dynamic> y_v; // latent part
		y_v.setZero(k);
		for (size_t m = 2; m <= order; ++m) {
				auto dp = get_dp_eval(entry, m);
				y_v += dp.row(m).tail(k);
		}
		return y_v.sum() + inv_norm * y_l + b.coeff(0);
}

void HOFM::check_feature_id(size_t id) {
		if (id < n) return;
		size_t n_extras = m_extras.size();

		w.conservativeResize(id + 1, 1 + n_extras);
		auto W = w.bottomRows(id + 1 - n);
		W.col(0).fill(0);
		for (size_t i = 1; i <= n_extras; ++i) {
				W.col(i).fill(m_extras[i - 1]);
		}

		for (auto& p : v) {
				p.conservativeResize(id + 1, k * (1 + n_extras));
				auto P = p.bottomRows(id + 1 - n);
				auto G = random_generator();
				std::uniform_real_distribution<real_t> dist(
					0.0, 0.66 / std::sqrt(k));
				for (index_t r = 0; r < P.rows(); ++r) {
						for (size_t c = 0; c < k; ++c) {
								P(r, c) = dist(G);
						}
				}
				for (size_t i = 1; i <= n_extras; ++i) {
						P.middleCols(k * i, k).fill(m_extras[i - 1]);
				}
				ASSERT(!p.hasNaN());
		}

		n = id + 1;
}
void HOFM::check_field_id(size_t) {}

NAMESPACE_END
