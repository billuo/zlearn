#include "sampler.h"

#include "base/random.h"
#include <algorithm>

NAMESPACE_BEGIN

class InMemorySampler : public Sampler {
	public:
		InMemorySampler(std::shared_ptr<DataSet> data) {
				reset(std::move(data));
		}

		void reset(std::shared_ptr<DataSet> data) {
				m_data = std::move(data);
				for (size_t i = 0; i < m_data->size(); ++i) {
						m_order.push_back(i);
				}
		}

		void restart() override { m_offset = 0; }
		void shuffle() override {
				std::shuffle(m_order.begin(), m_order.end(),
							 random_generator());
		}

	protected:
		std::shared_ptr<DataSet> m_data;
		std::vector<size_t> m_order;
		size_t m_offset = 0;

		size_t get_samples(size_t n, Entries& samples) override {
				ASSERT(m_data != nullptr);
				auto begin = m_data->entries().begin() + m_offset;
				auto end = m_data->entries().end();
				if (static_cast<size_t>(end - begin) > n) end = begin + n;
				m_offset += end - begin;
				for (auto it = begin; it != end; ++it) {
						(*it)->set_normalize(normalize());
				}
				std::copy(begin, end, std::back_inserter(samples));
				return end - begin;
		}
};

class CompositeSampler : public Sampler {
	public:
		explicit CompositeSampler(
			std::vector<std::shared_ptr<Sampler>> samplers = {})
		: m_samplers(std::move(samplers)) {}

		void shuffle() override {
				for (size_t i = m_index; i < m_samplers.size(); ++i) {
						m_samplers[i]->shuffle();
				}
		}
		void restart() override {
				for (size_t i = 0; i <= m_index && i < m_samplers.size(); ++i) {
						m_samplers[i]->restart();
				}
				m_index = 0;
		}

	protected:
		std::vector<std::shared_ptr<Sampler>> m_samplers;
		size_t m_index = 0; // current sampler

		size_t get_samples(size_t n, Entries& samples) override {
				size_t n_newly_sampled = 0;
				while (m_index != m_samplers.size() && n_newly_sampled < n) {
						auto& sampler = m_samplers[m_index];
						ASSERT(sampler);
						if (size_t n_sampled = sampler->get_samples(
								n - n_newly_sampled, samples)) {
								n_newly_sampled += n_sampled;
						} else {
								++m_index;
						}
				}
				auto sz = samples.size();
				for (size_t i = sz - 1; i >= sz - n_newly_sampled; --i) {
						samples[i]->set_normalize(normalize());
				}
				return n_newly_sampled;
		}
};

std::shared_ptr<Sampler> Sampler::create(std::shared_ptr<DataSet> data) {
		return std::make_shared<InMemorySampler>(std::move(data));
}
std::shared_ptr<Sampler>
Sampler::create(std::vector<std::shared_ptr<Sampler>> sampler) {
		return std::make_shared<CompositeSampler>(std::move(sampler));
}

NAMESPACE_END
