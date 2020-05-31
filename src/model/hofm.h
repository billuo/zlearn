#pragma once

#include "base/algebra.h"
#include "model.h"
#include "optimizer/optimizer.h"

NAMESPACE_BEGIN

class HOFM : public Model {
	public:
		HOFM(size_t order, size_t k) : order(order), k(k) {}

		size_t order; // # order
		size_t k;     // # latent factors
		size_t n = 0; // # features

		Vector<Dynamic> b;
		Matrix<Dynamic, Dynamic> w;
		std::vector<Matrix<Dynamic, Dynamic>> v;
		// v consists of m matrices p;
		// layout of p_i for i=1,...,n:
		// size=(n, k*(1+n_extra))
		//  <- k -> <-  k  ->       <-  k  ->
		// |-------|---------|-----|---------|
		// |  p_i  | extra_1 | ... | extra_m |
		// |-------|---------|-----|---------|

		auto get_p(size_t m, index_t feature_id, size_t nth_extra = 0) {
				return v[m - 2].row(feature_id).segment(nth_extra * k, k);
		}
		auto get_p(size_t m, index_t feature_id, size_t nth_extra = 0) const {
				return v[m - 2].row(feature_id).segment(nth_extra * k, k);
		}

		Matrix<Dynamic, Dynamic> get_dp_eval(Entry& entry, size_t m) const;
		Matrix<Dynamic, Dynamic> get_dp_grad(Entry& entry, size_t m) const;

		size_t n_parameters() const override {
				size_t ret = b.size() + w.size();
				for (auto& p : v) {
						ret += p.size();
				}
				return ret;
		}

		void initialize(Sampler& sampler, std::vector<real_t> extra) override;
		void check_feature_id(size_t id) override;
		void check_field_id(size_t id) override;
		real_t predict(Entry& entry) override;
		void serialize_txt(const String& filename) const override;
		void serialize(const String& filename) const override;
		void deserialize(const String& filename) override;

		void take_snapshot() override {
				b_copy = b;
				w_copy = w;
				v_copy = v;
		}
		void restore_snapshot() override {
				b = b_copy;
				w = w_copy;
				v = v_copy;
		}
		size_t max_feature_id() override { return n - 1; }

	private:
		Vector<Dynamic> b_copy;
		Matrix<Dynamic, Dynamic> w_copy;
		std::vector<Matrix<Dynamic, Dynamic>> v_copy;

		void optimize(Optimizer& optimizer, Entry& entry, real_t pg) override {
				optimizer.optimize(*this, entry, pg);
		}
};

NAMESPACE_END
