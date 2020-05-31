#pragma once

#include "base/algebra.h"
#include "model.h"
#include "optimizer/optimizer.h"

NAMESPACE_BEGIN

class LM : public Model {
	public:
		size_t n = 0; // # features
		// given feature vector, x, of size n:
		// y= w*x+b
		Vector<Dynamic> b;
		Matrix<Dynamic, Dynamic> w;

		size_t n_parameters() const override { return w.size() + b.size(); }

		void initialize(Sampler& sampler, std::vector<real_t> extras) override;
		void check_feature_id(size_t id) override;
		void check_field_id(size_t id) override;
		real_t predict(Entry&) override;
		void serialize_txt(const String& filename) const override;
		void serialize(const String& filename) const override;
		void deserialize(const String& filename) override;

		void take_snapshot() override {
				b_copy = b;
				w_copy = w;
		}
		void restore_snapshot() override {
				b = b_copy;
				w = w_copy;
		}
		size_t max_feature_id() override { return n - 1; }

	private:
		Vector<Dynamic> b_copy;
		Matrix<Dynamic, Dynamic> w_copy;

		void optimize(Optimizer& optimizer, Entry& entry, real_t pg) override {
				optimizer.optimize(*this, entry, pg);
		}
};

NAMESPACE_END
