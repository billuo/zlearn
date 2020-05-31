#include "model.h"
#include "base/enum_db.h"
#include "base/logger.h"
#include "base/thread_pool.h"
#include "data/sampler.h"
#include "ffm.h"
#include "fm.h"
#include "hofm.h"
#include "lm.h"
#include "optimizer/optimizer.h"

#include <valarray>

NAMESPACE_BEGIN

ENUM_DB_DEFINITION(Loss) = {
	{Loss::Squared, "MSE"},
	{Loss::CrossEntropy, "log loss"},
};

Model::~Model() = default;

std::unique_ptr<Model> Model::create_LM() { return std::make_unique<LM>(); }
std::unique_ptr<Model> Model::create_FM(size_t k) {
		return std::make_unique<FM>(k);
}
std::unique_ptr<Model> Model::create_FFM(size_t k) {
		return std::make_unique<FFM>(k);
}
std::unique_ptr<Model> Model::create_HOFM(size_t m, size_t k) {
		return std::make_unique<HOFM>(m, k);
}

static std::atomic_bool thrown = false;
real_t Model::run_model(Loss loss,
						Sampler& sampler,
						ThreadPool& thread_pool,
						Optimizer* optimizer) {
		size_t n_threads = thread_pool.size();
		size_t total_samples = 0;
		std::valarray<real_t> thread_sum(n_threads);
		Entries entries;
		while (size_t n_sample = sampler.get_samples(-1, entries)) {
				total_samples += n_sample;
				auto splits = ThreadPool::split_task(n_sample, n_threads);
				auto p = entries.data();
				for (size_t i = 0; i < n_threads; ++i) {
						auto start = p + splits[i];
						auto end = p + splits[i + 1];
						auto task =
							std::bind(&Model::thread_run_model, this, loss,
									  start, end, &thread_sum[i], optimizer);
						thread_pool.enqueue(std::move(task));
				}
				thread_pool.sync(n_threads);
				if (thrown) throw Exception("exception thrown");
				entries.clear();
		}
		return thread_sum.sum() / total_samples;
}

void Model::thread_run_model(Loss loss,
							 std::shared_ptr<Entry>* start,
							 std::shared_ptr<Entry>* end,
							 real_t* result,
							 Optimizer* optimizer) {
		try {
				real_t loss_sum = 0;
				switch (loss) {
				case Loss::Squared: {
						for (auto p = start; p < end; ++p) {
								auto& entry = **p;
								real_t predicted = predict(entry);
								real_t error = entry.label - predicted;
								loss_sum += error * error;
								if (optimizer) {
										real_t pg = -error;
										optimize(*optimizer, entry, pg);
								}
						}
						loss_sum /= 2;
				} break;
				case Loss::CrossEntropy: {
						for (auto p = start; p < end; ++p) {
								auto& entry = **p;
								real_t predicted = predict(entry);
								real_t y = entry.label > 0 ? 1.0 : -1.0;
								auto t = std::exp(-y * predicted);
								loss_sum += std::log1p(t);
								if (optimizer) {
										real_t pg = -y * t / (1.0 + t);
										optimize(*optimizer, entry, pg);
								}
						}
				} break;
				default: UNREACHABLE("bad loss");
				}
				*result += loss_sum;
		} catch (const std::exception& e) {
				logger::error("{}", e.what());
				thrown = true;
				*result = NAN;
		}
}

NAMESPACE_END
