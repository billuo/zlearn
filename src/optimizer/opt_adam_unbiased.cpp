#include "optimizer.h"

#include "model/ffm.h"
#include "model/fm.h"
#include "model/hofm.h"
#include "model/lm.h"

// not an inline function in fear of breaking lazy evaluation
#define MIX(x, y, alpha) (alpha) * (x) + (1 - (alpha)) * (y)

NAMESPACE_BEGIN

static void optimize_bias(Vector<Dynamic>& b,
						  real_t pg,
						  real_t lr,
						  real_t beta_1,
						  real_t beta_2,
						  real_t beta_1_pow,
						  real_t beta_2_pow,
						  real_t eps) {
		auto& B = b.coeffRef(0);
		auto& M = b.coeffRef(1);
		auto& V = b.coeffRef(2);
		M = MIX(M, pg, beta_1);
		V = MIX(V, pg * pg, beta_2);
		real_t Mt = M / (1 - beta_1_pow);
		real_t Vt = V / (1 - beta_2_pow);
		B -= lr * Mt / std::sqrt(Vt + eps);
}

void AdamUnbiased::set_epoch(int epoch) {
		Optimizer::set_epoch(epoch);
		m_beta_1_pow = std::pow(m_beta_1, epoch);
		m_beta_2_pow = std::pow(m_beta_2, epoch);
}

void AdamUnbiased::optimize(LM& lm, Entry& entry, real_t pg) const {
		optimize_bias(lm.b, pg, m_learning_rate, m_beta_1, m_beta_2,
					  m_beta_1_pow, m_beta_2_pow, Epsilon);

		for (const auto& f : entry.features) {
				lm.check_feature_id(f.id);
				real_t& w = lm.w.row(f.id).coeffRef(0);
				real_t& M = lm.w.row(f.id).coeffRef(1);
				real_t& V = lm.w.row(f.id).coeffRef(2);
				real_t g = m_lambda_r * w + pg * f.normalized_value;
				M = MIX(M, g, m_beta_1);
				V = MIX(V, g * g, m_beta_2);
				real_t Mt = M / (1 - m_beta_1_pow);
				real_t Vt = V / (1 - m_beta_2_pow);
				w -= m_learning_rate * Mt / std::sqrt(Vt + Epsilon);
		}
}

void AdamUnbiased::optimize(FM& fm, Entry& entry, real_t pg) const {
		auto s = fm.allocate_s();

		optimize_bias(fm.b, pg, m_learning_rate, m_beta_1, m_beta_2,
					  m_beta_1_pow, m_beta_2_pow, Epsilon);

		for (const auto& f : entry.features) {
				fm.check_feature_id(f.id);
				real_t& w = fm.w.row(f.id).coeffRef(0);
				real_t& M = fm.w.row(f.id).coeffRef(1);
				real_t& V = fm.w.row(f.id).coeffRef(2);
				real_t g = m_lambda_r * w + pg * f.normalized_value;
				M = MIX(M, g, m_beta_1);
				V = MIX(V, g * g, m_beta_2);
				real_t Mt = M / (1 - m_beta_1_pow);
				real_t Vt = V / (1 - m_beta_2_pow);
				w -= m_learning_rate * Mt / std::sqrt(Vt + Epsilon);
				s += fm.v.row(f.id).head(fm.k) * f.normalized_value;
		}
		for (const auto& f : entry.features) {
				fm.check_feature_id(f.id);
				auto v = fm.v.row(f.id).head(fm.k);
				auto M = fm.v.row(f.id).segment(fm.k, fm.k);
				auto V = fm.v.row(f.id).tail(fm.k);
				auto g = m_lambda_r * v + pg * (s - v * f.normalized_value);
				M = MIX(M, g, m_beta_1);
				V = MIX(V, g.cwiseProduct(g), m_beta_2);
				auto Mt = M / (1 - m_beta_1_pow);
				auto Vt = V / (1 - m_beta_2_pow);
				v.array() -= m_learning_rate
					* Mt.array().cwiseProduct(
						(Vt.array() + Epsilon).cwiseSqrt().cwiseInverse());
		}
}

void AdamUnbiased::optimize(FFM& ffm, Entry& entry, real_t pg) const {
		const auto& features = entry.features;
		real_t inv_norm2 = entry.get_inv_norm2();

		optimize_bias(ffm.b, pg, m_learning_rate, m_beta_1, m_beta_2,
					  m_beta_1_pow, m_beta_2_pow, Epsilon);

		for (auto it1 = features.begin(); it1 != features.end(); ++it1) {
				auto& f1 = *it1;
				ffm.check_feature_id(f1.id);
				ffm.check_field_id(f1.field_id);
				real_t& w = ffm.w.row(f1.id).coeffRef(0);
				real_t& M = ffm.w.row(f1.id).coeffRef(1);
				real_t& V = ffm.w.row(f1.id).coeffRef(2);
				real_t g = m_lambda_r * w + pg * f1.normalized_value;
				M = MIX(M, g, m_beta_1);
				V = MIX(V, g * g, m_beta_2);
				real_t Mt = M / (1 - m_beta_1_pow);
				real_t Vt = V / (1 - m_beta_2_pow);
				w -= m_learning_rate * Mt / std::sqrt(Vt + Epsilon);
				for (auto it2 = it1 + 1; it2 != features.end(); ++it2) {
						auto& f2 = *it2;
						ffm.check_feature_id(f2.id);
						ffm.check_field_id(f2.field_id);
						auto v_i_fj = ffm.get_v(f1.id, f2.field_id);
						auto v_j_fi = ffm.get_v(f2.id, f1.field_id);
						auto M_i = ffm.get_v(f1.id, f2.field_id, 1);
						auto M_j = ffm.get_v(f2.id, f1.field_id, 1);
						auto V_i = ffm.get_v(f1.id, f2.field_id, 2);
						auto V_j = ffm.get_v(f2.id, f1.field_id, 2);
						auto pgx = pg * f1.value * f2.value * inv_norm2;
						auto g_i = m_lambda_r * v_i_fj + pgx * v_j_fi;
						auto g_j = m_lambda_r * v_j_fi + pgx * v_i_fj;
						M_i = MIX(M_i, g_i, m_beta_1);
						M_j = MIX(M_j, g_j, m_beta_1);
						V_i = MIX(V_i, g_i.cwiseProduct(g_i), m_beta_2);
						V_j = MIX(V_j, g_j.cwiseProduct(g_j), m_beta_2);
						auto Mt_i = M_i / (1 - m_beta_1_pow);
						auto Mt_j = M_j / (1 - m_beta_1_pow);
						auto Vt_i = V_i / (1 - m_beta_2_pow);
						auto Vt_j = V_j / (1 - m_beta_2_pow);
						v_i_fj.array() -= m_learning_rate
							* Mt_i.array().cwiseProduct((Vt_i.array() + Epsilon)
															.cwiseSqrt()
															.cwiseInverse());
						v_j_fi.array() -= m_learning_rate
							* Mt_j.array().cwiseProduct((Vt_j.array() + Epsilon)
															.cwiseSqrt()
															.cwiseInverse());
				}
		}
}

void AdamUnbiased::optimize(HOFM& hofm, Entry& entry, real_t pg) const {
		const auto& features = entry.features;
		size_t d = features.size();
		size_t k = hofm.k;

		optimize_bias(hofm.b, pg, m_learning_rate, m_beta_1, m_beta_2,
					  m_beta_1_pow, m_beta_2_pow, Epsilon);

		for (const auto& f : entry.features) {
				hofm.check_feature_id(f.id);
				real_t& w = hofm.w.row(f.id).coeffRef(0);
				real_t& M = hofm.w.row(f.id).coeffRef(1);
				real_t& V = hofm.w.row(f.id).coeffRef(2);
				real_t g = m_lambda_r * w + pg * f.normalized_value;
				M = MIX(M, g, m_beta_1);
				V = MIX(V, g * g, m_beta_2);
				real_t Mt = M / (1 - m_beta_1_pow);
				real_t Vt = V / (1 - m_beta_2_pow);
				w -= m_learning_rate * Mt / std::sqrt(Vt + Epsilon);
		}

		for (size_t m = 2; m <= hofm.order; ++m) {
				auto a = hofm.get_dp_eval(entry, m);
				auto a_adj = hofm.get_dp_grad(entry, m);
				for (size_t j = 1; j <= d; ++j) {
						auto& f = features[j - 1];
						auto p = hofm.get_p(m, f.id);
						auto M = hofm.get_p(m, f.id, 1);
						auto V = hofm.get_p(m, f.id, 2);
						auto a_adj_j = a_adj.block(1, j * k, m, k);
						auto a_jm1 = a.block(0, (j - 1) * k, m, k);
						auto g = m_lambda_r * p
							+ pg * a_adj_j.cwiseProduct(a_jm1).colwise().sum()
								* f.value;
						M = MIX(M, g, m_beta_1);
						V = MIX(V, g.cwiseProduct(g), m_beta_2);
						auto Mt = M / (1 - m_beta_1_pow);
						auto Vt = V / (1 - m_beta_2_pow);
						p.array() -= m_learning_rate
							* Mt.array().cwiseProduct((Vt.array() + Epsilon)
														  .cwiseSqrt()
														  .cwiseInverse());
				}
		}
}

NAMESPACE_END
