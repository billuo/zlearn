#pragma once

#include "base/common.h"
#include "base/str.h"

NAMESPACE_BEGIN

class Model;
class Sampler;
struct Entry;

struct Metric_impl;
class Metric {
	public:
		enum Type { Accuracy, Precision, Recall, MAE, MAPE, RMSD, AUC };

		Metric(Type type);
		Metric(Metric&&);
		Metric& operator=(Metric&&);
		~Metric();

		void reset();
		void evaluate(Model&, Sampler&);
		void accumulate(real_t* P, std::shared_ptr<Entry>* T, size_t n);

		real_t value() const;
		String name() const;
		Type type() const { return m_type; }

		static bool is_score(Type type);
		static bool is_loss(Type type) { return !is_score(type); }
		// a metric is a score if the higher it is, the better.
		bool is_score() const { return is_score(m_type); }
		// a metric is a loss if the lower it is, the better.
		bool is_loss() const { return is_loss(m_type); }

		static bool better(Type type, real_t x, real_t y);
		// a good metric value is 'greater' than a bad metric value.
		bool operator<(const Metric& rhs) const;
		bool operator>(const Metric& rhs) const { return rhs < *this; }
		bool operator<=(const Metric& rhs) const { return !(rhs < *this); }
		bool operator>=(const Metric& rhs) const { return !(*this < rhs); }

	private:
		Type m_type;
		std::unique_ptr<Metric_impl> m_impl;
};

NAMESPACE_END
