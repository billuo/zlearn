#pragma once

#include "data_set.h"

NAMESPACE_BEGIN

// data sampler (from disk or memory).
// NOTE: data entry already sampled won't be returned again unless sampler is restarted.
class Sampler {
	public:
		// create a sampler that samples from already built data set.
		static std::shared_ptr<Sampler> create(std::shared_ptr<DataSet> data);
		// create a sampler that samples from a series of other samplers.
		static std::shared_ptr<Sampler>
		create(std::vector<std::shared_ptr<Sampler>> sampler);

		// get (at most) n samples into given entries.
		// return actual # samples sampled.
		// NOTE: what's already in result will NOT be cleared.
		virtual size_t get_samples(size_t n, Entries& result) = 0;

		// shuffle the order of future data to sample; no guaranteed to have any effect
		virtual void shuffle() = 0;

		// make reader to sample from the very beginning of data
		virtual void restart() = 0;

		void set_normalize(bool b) { m_normalize = b; }
		bool normalize() const { return m_normalize; }

	protected:
		bool m_normalize = false; // compute norm feature vector
};

NAMESPACE_END
