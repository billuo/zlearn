#pragma once

#include "data/entry.h"
#include "data/sampler.h"

NAMESPACE_BEGIN

class Sampler;
class ThreadPool;
class Optimizer;

enum class Loss {
		Squared,
		CrossEntropy,
};
class Model {
	public:
		static std::unique_ptr<Model> create_LM();
		static std::unique_ptr<Model> create_FM(size_t k);
		static std::unique_ptr<Model> create_FFM(size_t k);
		static std::unique_ptr<Model> create_HOFM(size_t m, size_t k);
		static std::unique_ptr<Model> from_file(const String& filename);

		Model() = default;
		virtual ~Model();

		// prepare the model for to be trained with some data.
		// concretely, # features/fields need to be determined before training.
		virtual void initialize(Sampler& sampler,
								std::vector<real_t> extras) = 0;
		virtual void check_feature_id(size_t id) = 0;
		virtual void check_field_id(size_t id) = 0;

		virtual real_t predict(Entry&) = 0;

		virtual size_t n_parameters() const = 0;
		// serialize model to a text file
		virtual void serialize_txt(const String& filename) const = 0;
		// serialize model to a binary file
		virtual void serialize(const String& filename) const = 0;
		// deserialize model from a binary file
		virtual void deserialize(const String& filename) = 0;

		// to keep record of the best model during training to support early stopping.
		virtual void take_snapshot() = 0;
		virtual void restore_snapshot() = 0;

		virtual size_t max_feature_id() = 0;
		virtual size_t max_field_id() { return -1; }

		real_t evaluate(Loss loss, Sampler& sampler, ThreadPool& pool) {
				return run_model(loss, sampler, pool, nullptr);
		}
		real_t update(Loss loss,
					  Optimizer& optimizer,
					  Sampler& sampler,
					  ThreadPool& pool) {
				return run_model(loss, sampler, pool, &optimizer);
		}

	protected:
		std::vector<real_t> m_extras;

	private:
		real_t run_model(Loss, Sampler&, ThreadPool&, Optimizer* optimizer);
		void thread_run_model(Loss loss,
							  std::shared_ptr<Entry>* start,
							  std::shared_ptr<Entry>* end,
							  real_t* result,
							  Optimizer* optimizer);
		virtual void optimize(Optimizer&, Entry&, real_t pg) = 0;
};

NAMESPACE_END
