#pragma once

#include "base/algebra.h"
#include "model.h"
#include "optimizer/optimizer.h"

NAMESPACE_BEGIN

class FFM : public Model {
	public:
		FFM(size_t k) : k(k) {}

		size_t k;       // # latent factors
		size_t n = 0;   // # features
		size_t n_f = 0; // # fields

		// given feature vector, x, of size n:
		// y= ∑(v_i_fj * v_j_fi) x_i x_j + L, i=1,...,n; j=i+1,...,n
		// where v_i is of size k; L = w*x+b

		Vector<Dynamic> b;
		Matrix<Dynamic, Dynamic> w;
		Matrix<Dynamic, Dynamic> v;
		// layout of v_i for i=1,...,n:
		// size=(n, k*(1+n_extra)*n_f)
		//  <-  k -> <-  k  ->     <-  k  ->       <-  k -> <-  k  ->     <-  k  ->
		// |--------|---------|---|---------|-----|--------|---------|---|---------|
		// | v_i_f0 | ex_1_f0 |...| ex_n_f0 | ... | v_i_fn | ex_1_fn |...| ex_n_fn |
		// |--------|---------|---|---------|-----|--------|---------|---|---------|

		auto get_v(index_t feature_id, index_t field_id, size_t nth_extra = 0) {
				return v.row(feature_id)
					.segment(field_id * k * (1 + m_extras.size())
								 + k * nth_extra,
							 k);
		}
		auto get_v(index_t feature_id,
				   index_t field_id,
				   size_t nth_extra = 0) const {
				return v.row(feature_id)
					.segment(field_id * k * (1 + m_extras.size())
								 + k * nth_extra,
							 k);
		}

		size_t n_parameters() const override {
				return b.size() + w.size() + v.size();
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
		virtual size_t max_field_id() override { return n_f - 1; }

	private:
		Vector<Dynamic> b_copy;
		Matrix<Dynamic, Dynamic> w_copy;
		Matrix<Dynamic, Dynamic> v_copy;

		void optimize(Optimizer& optimizer, Entry& entry, real_t pg) override {
				optimizer.optimize(*this, entry, pg);
		}
};

NAMESPACE_END
