#include "metric.h"
#include "base/enum_db.h"
#include "base/exception.hpp"
#include "base/math.h"
#include "data/sampler.h"
#include "model/model.h"

NAMESPACE_BEGIN

struct Metric_impl {
		real_t sum = 0;   // meaning depends to metric type
		size_t count = 0; // indicates how many samples sum consist

		// for AUC
		static constexpr size_t bucket_size = 1e6;
		std::array<size_t, bucket_size> positives = {};
		std::array<size_t, bucket_size> negatives = {};
};

ENUM_DB_DEFINITION(Metric::Type) = {
	{Metric::Accuracy, "acc"},  {Metric::Precision, "prec"},
	{Metric::Recall, "recall"}, {Metric::MAE, "mae"},
	{Metric::MAPE, "mape"},     {Metric::RMSD, "rmsd"},
	{Metric::AUC, "auc"},
};

Metric::Metric(Metric::Type type) : m_type(type), m_impl(new Metric_impl) {}
Metric::~Metric() = default;
Metric::Metric(Metric&&) = default;
Metric& Metric::operator=(Metric&&) = default;

void Metric::reset() {
		m_impl->sum = m_impl->count = 0;
		if (m_type == AUC) {
				m_impl->positives.fill(0);
				m_impl->negatives.fill(0);
		}
}

void Metric::evaluate(Model& model, Sampler& sampler) {
		std::vector<real_t> pred;
		Entries entries;
		while (size_t n_sample = sampler.get_samples(-1, entries)) {
				pred.reserve(pred.size() + n_sample);
				for (auto& e : entries) {
						pred.push_back(model.predict(*e));
				}
				accumulate(pred.data(), entries.data(), pred.size());
				entries.clear();
		}
}

void Metric::accumulate(real_t* predicted,
						std::shared_ptr<Entry>* truth,
						size_t n) {
		size_t n_count = 0;
		for (size_t i = 0; i < n; ++i) {
				auto P = predicted[i];
				auto T = truth[i]->label;
				switch (m_type) {
				case Type::Accuracy:
						ASSERT(P == 0 || P == 1);
						ASSERT(T == 0 || T == 1);
						m_impl->sum += P == T;
						++n_count;
						break;
				case Type::Precision:
						ASSERT(P == 0 || P == 1);
						ASSERT(T == 0 || T == 1);
						if (P == 1) {
								m_impl->sum += T == 1;
								++n_count;
						}
						break;
				case Type::Recall:
						ASSERT(P == 0 || P == 1);
						ASSERT(T == 0 || T == 1);
						if (T == 1) {
								m_impl->sum += P == 1;
								++n_count;
						}
						break;
				case Type::MAE:
						m_impl->sum += std::abs(P - T);
						++n_count;
						break;
				case Type::MAPE:
						m_impl->sum += std::abs(P - T) / T;
						++n_count;
						break;
				case Type::RMSD:
						m_impl->sum += (P - T) * (P - T);
						++n_count;
						break;
				case Type::AUC: {
						int idx = sigmoid(P) * m_impl->bucket_size;
						idx %= m_impl->bucket_size;
						++(T > 0 ? m_impl->positives : m_impl->negatives)[idx];
				} break;
				default: UNREACHABLE("bad metric type");
				}
		}
		m_impl->count += n_count;
}

real_t Metric::value() const {
		switch (m_type) {
		case Type::Accuracy:
		case Type::Precision:
		case Type::Recall:
		case Type::MAE:
		case Type::MAPE: return m_impl->sum / m_impl->count;
		case Type::RMSD: return std::sqrt(m_impl->sum / m_impl->count);
		case Type::AUC: {
				auto& positives = m_impl->positives;
				auto& negatives = m_impl->negatives;
				u64 positive_sum = 0;
				u64 negative_sum = 0;
				double auc = 0.0;
				for (size_t i = 0; i < m_impl->bucket_size; ++i) {
						auto t = positive_sum;
						positive_sum += positives[i];
						negative_sum += negatives[i];
						auc += (t + positive_sum) * (negatives[i]);
				}
				auc /= 2;
				return 1.0 - auc / (positive_sum * negative_sum);
		}
		}
		UNREACHABLE("bad metric type");
}

String Metric::name() const { return EnumDB<Type>::to_string(m_type); }

bool Metric::is_score(Type type) {
		switch (type) {
		case Type::Accuracy:
		case Type::Precision:
		case Type::Recall:
		case Type::AUC: return true;
		case Type::MAE:
		case Type::MAPE:
		case Type::RMSD: return false;
		}
		UNREACHABLE("bad metric type");
}

bool Metric::better(Metric::Type type, real_t x, real_t y) {
		if (is_score(type)) return x > y;
		if (is_loss(type)) return x < y;
		THROW("can not compare");
}
bool Metric::operator<(const Metric& rhs) const {
		if (m_type != rhs.m_type)
				throw Exception("comparing different metric: %s & %s",
								name().c_str(), rhs.name().c_str());
		return better(m_type, rhs.value(), value()); // greater is better!
}

NAMESPACE_END
