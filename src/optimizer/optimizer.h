#pragma once

#include "base/config.h"

#include <vector>

NAMESPACE_BEGIN

struct Entry;
class LM;
class FM;
class FFM;
class HOFM;

class Optimizer {
	public:
		Optimizer(real_t learning_rate, real_t lambda_r)
		: m_learning_rate(learning_rate), m_lambda_r(lambda_r) {}

		virtual void set_epoch(int epoch) { m_epoch = epoch; }

		// NOTE: double dispatch
		virtual void optimize(LM&, Entry&, real_t pg) const = 0;
		virtual void optimize(FM&, Entry&, real_t pg) const = 0;
		virtual void optimize(FFM&, Entry&, real_t pg) const = 0;
		virtual void optimize(HOFM&, Entry&, real_t pg) const = 0;

		// # extra number used as cache for each parameter to optimize
		virtual std::vector<real_t> extras() const = 0;

	protected:
		real_t m_learning_rate;
		real_t m_lambda_r;
		int m_epoch = 0; // current epoch counting from 1

		static constexpr real_t Epsilon = 9e-8; // for avoiding division by zero
};

class SGD : public Optimizer {
	public:
		SGD(real_t learning_rate, real_t lambda_r)
		: Optimizer(learning_rate, lambda_r) {}

		void optimize(LM&, Entry&, real_t pg) const override;
		void optimize(FM&, Entry&, real_t pg) const override;
		void optimize(FFM&, Entry&, real_t pg) const override;
		void optimize(HOFM&, Entry&, real_t pg) const override;

		std::vector<real_t> extras() const override;
};

class AdaGrad : public Optimizer {
	public:
		AdaGrad(real_t learning_rate, real_t lambda_r)
		: Optimizer(learning_rate, lambda_r) {}

		void optimize(LM&, Entry&, real_t pg) const override;
		void optimize(FM&, Entry&, real_t pg) const override;
		void optimize(FFM&, Entry&, real_t pg) const override;
		void optimize(HOFM&, Entry&, real_t pg) const override;

		std::vector<real_t> extras() const override;
};

class RMSProp : public Optimizer {
	public:
		RMSProp(real_t learning_rate, real_t lambda_r, real_t alpha)
		: Optimizer(learning_rate, lambda_r), m_alpha(alpha) {}

		void optimize(LM&, Entry&, real_t pg) const override;
		void optimize(FM&, Entry&, real_t pg) const override;
		void optimize(FFM&, Entry&, real_t pg) const override;
		void optimize(HOFM&, Entry&, real_t pg) const override;

		std::vector<real_t> extras() const override;

	protected:
		real_t m_alpha;
};

class Momentum : public Optimizer {
	public:
		Momentum(real_t learning_rate, real_t lambda_r, real_t gamma)
		: Optimizer(learning_rate, lambda_r), m_gamma(gamma) {}

		void optimize(LM&, Entry&, real_t pg) const override;
		void optimize(FM&, Entry&, real_t pg) const override;
		void optimize(FFM&, Entry&, real_t pg) const override;
		void optimize(HOFM&, Entry&, real_t pg) const override;

		std::vector<real_t> extras() const override;

	protected:
		real_t m_gamma;
};

class Adam : public Optimizer {
	public:
		Adam(real_t learning_rate,
			 real_t lambda_r,
			 real_t beta_1,
			 real_t beta_2)
		: Optimizer(learning_rate, lambda_r)
		, m_beta_1(beta_1)
		, m_beta_2(beta_2) {}

		void optimize(LM&, Entry&, real_t pg) const override;
		void optimize(FM&, Entry&, real_t pg) const override;
		void optimize(FFM&, Entry&, real_t pg) const override;
		void optimize(HOFM&, Entry&, real_t pg) const override;

		std::vector<real_t> extras() const override;

	protected:
		real_t m_beta_1;
		real_t m_beta_2;
};

class AdamUnbiased : public Adam {
	public:
		AdamUnbiased(real_t learning_rate,
					 real_t lambda_r,
					 real_t beta_1,
					 real_t beta_2)
		: Adam(learning_rate, lambda_r, beta_1, beta_2) {}

		void set_epoch(int epoch) override;

		void optimize(LM&, Entry&, real_t pg) const override;
		void optimize(FM&, Entry&, real_t pg) const override;
		void optimize(FFM&, Entry&, real_t pg) const override;
		void optimize(HOFM&, Entry&, real_t pg) const override;

	protected:
		real_t m_beta_1_pow;
		real_t m_beta_2_pow;
};

class AMSGrad : public Adam {
	public:
		AMSGrad(real_t learning_rate,
				real_t lambda_r,
				real_t beta_1,
				real_t beta_2)
		: Adam(learning_rate, lambda_r, beta_1, beta_2) {}

		void optimize(LM&, Entry&, real_t pg) const override;
		void optimize(FM&, Entry&, real_t pg) const override;
		void optimize(FFM&, Entry&, real_t pg) const override;
		void optimize(HOFM&, Entry&, real_t pg) const override;

		std::vector<real_t> extras() const override;
};

NAMESPACE_END
