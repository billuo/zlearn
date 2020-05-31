#include "application_impl.h"
#include "base/logger.h"
#include "base/timer.h"
#include "data/sampler.h"

NAMESPACE_BEGIN

// TODO parallelize
void Application_impl::predict(Sampler& sampler) {
		Timer predict_timer;
		predict_timer.tic();

		Entries entries;
		predicted.clear();
		while (sampler.get_samples(-1, entries)) {
				for (auto& e : entries) {
						predicted.push_back(model->predict(*e));
				}
		}

		predict_timer.toc();
		logger::info("finish predicting... {:.2} seconds elapsed",
					 predict_timer.seconds());
}

NAMESPACE_END
