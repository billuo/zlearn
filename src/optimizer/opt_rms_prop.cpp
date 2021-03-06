#include "optimizer.h"

#include "model/ffm.h"
#include "model/fm.h"
#include "model/hofm.h"
#include "model/lm.h"

// not an inline function in fear of breaking lazy evaluation
#define MIX(x, y, alpha) (alpha) * (x) + (1 - (alpha)) * (y)

NAMESPACE_BEGIN

static void optimize_bias(
	Vector<Dynamic>& b, real_t pg, real_t lr, real_t alpha, real_t eps) {
		auto& B = b.coeffRef(0);
		auto& G = b.coeffRef(1);
		G = MIX(G, pg * pg, alpha);
		B -= lr * pg / std::sqrt(G + eps);
}

void RMSProp::optimize(LM& lm, Entry& entry, real_t pg) const {
		optimize_bias(lm.b, pg, m_learning_rate, m_alpha, Epsilon);

		for (const auto& f : entry.features) {
				lm.check_feature_id(f.id);
				real_t& w = lm.w.row(f.id).coeffRef(0);
				real_t& G = lm.w.row(f.id).coeffRef(1);
				real_t g = m_lambda_r * w + pg * f.normalized_value;
				G = MIX(G, g * g, m_alpha);
				w -= m_learning_rate * g / std::sqrt(G + Epsilon);
		}
}

void RMSProp::optimize(FM& fm, Entry& entry, real_t pg) const {
		auto s = fm.allocate_s();

		optimize_bias(fm.b, pg, m_learning_rate, m_alpha, Epsilon);

		for (const auto& f : entry.features) {
				fm.check_feature_id(f.id);
				real_t& w = fm.w.row(f.id).coeffRef(0);
				real_t& G = fm.w.row(f.id).coeffRef(1);
				real_t g = m_lambda_r * w + pg * f.normalized_value;
				G = MIX(G, g * g, m_alpha);
				w -= m_learning_rate * g / std::sqrt(G + Epsilon);
				s += fm.v.row(f.id).head(fm.k) * f.normalized_value;
		}
		for (const auto& f : entry.features) {
				fm.check_feature_id(f.id);
				auto v = fm.v.row(f.id).head(fm.k);
				auto G = fm.v.row(f.id).tail(fm.k);
				auto g = m_lambda_r * v + pg * (s - v * f.normalized_value);
				G = MIX(G, g.cwiseProduct(g), m_alpha);
				auto d = g.array().cwiseProduct(
					(G.array() + Epsilon).cwiseSqrt().cwiseInverse());
				v.array() -= m_learning_rate * d;
		}
}

void RMSProp::optimize(FFM& ffm, Entry& entry, real_t pg) const {
		const auto& features = entry.features;
		real_t inv_norm2 = entry.get_inv_norm2();

		optimize_bias(ffm.b, pg, m_learning_rate, m_alpha, Epsilon);

		for (auto it1 = features.begin(); it1 != features.end(); ++it1) {
				auto& f1 = *it1;
				ffm.check_feature_id(f1.id);
				ffm.check_field_id(f1.field_id);
				real_t& w = ffm.w.row(f1.id).coeffRef(0);
				real_t& G = ffm.w.row(f1.id).coeffRef(1);
				real_t g = m_lambda_r * w + pg * f1.normalized_value;
				G = MIX(G, g * g, m_alpha);
				w -= m_learning_rate * g / std::sqrt(G + Epsilon);
				for (auto it2 = it1 + 1; it2 != features.end(); ++it2) {
						auto& f2 = *it2;
						ffm.check_feature_id(f2.id);
						ffm.check_field_id(f2.field_id);
						auto v_i_fj = ffm.get_v(f1.id, f2.field_id);
						auto v_j_fi = ffm.get_v(f2.id, f1.field_id);
						auto G_i = ffm.get_v(f1.id, f2.field_id, 1);
						auto G_j = ffm.get_v(f2.id, f1.field_id, 1);
						auto pgx = pg * f1.value * f2.value * inv_norm2;
						auto g_i = m_lambda_r * v_i_fj + pgx * v_j_fi;
						auto g_j = m_lambda_r * v_j_fi + pgx * v_i_fj;
						G_i = MIX(G_i, g_i.cwiseProduct(g_i), m_alpha);
						G_j = MIX(G_j, g_j.cwiseProduct(g_j), m_alpha);
						auto d_i = g_i.array().cwiseProduct(
							(G_i.array() + Epsilon).cwiseSqrt().cwiseInverse());
						auto d_j = g_j.array().cwiseProduct(
							(G_j.array() + Epsilon).cwiseSqrt().cwiseInverse());
						v_i_fj.array() -= m_learning_rate * d_i;
						v_j_fi.array() -= m_learning_rate * d_j;
				}
		}
}

void RMSProp::optimize(HOFM& hofm, Entry& entry, real_t pg) const {
		const auto& features = entry.features;
		size_t d = features.size();
		size_t k = hofm.k;

		optimize_bias(hofm.b, pg, m_learning_rate, m_alpha, Epsilon);

		for (const auto& f : entry.features) {
				hofm.check_feature_id(f.id);
				real_t& w = hofm.w.row(f.id).coeffRef(0);
				real_t& G = hofm.w.row(f.id).coeffRef(1);
				real_t g = m_lambda_r * w + pg * f.normalized_value;
				G = MIX(G, g * g, m_alpha);
				w -= m_learning_rate * g / std::sqrt(G + Epsilon);
		}

		for (size_t m = 2; m <= hofm.order; ++m) {
				auto a = hofm.get_dp_eval(entry, m);
				auto a_adj = hofm.get_dp_grad(entry, m);
				for (size_t j = 1; j <= d; ++j) {
						auto& f = features[j - 1];
						auto p = hofm.get_p(m, f.id);
						auto G = hofm.get_p(m, f.id, 1);
						auto a_adj_j = a_adj.block(1, j * k, m, k);
						auto a_jm1 = a.block(0, (j - 1) * k, m, k);
						auto g = m_lambda_r * p
							+ pg * a_adj_j.cwiseProduct(a_jm1).colwise().sum()
								* f.value;
						G = MIX(G, g.cwiseProduct(g), m_alpha);
						p.array() -= m_learning_rate
							* g.array().cwiseProduct((G.array() + Epsilon)
														 .cwiseSqrt()
														 .cwiseInverse());
				}
		}
}

std::vector<real_t> RMSProp::extras() const { return {1}; }

NAMESPACE_END
