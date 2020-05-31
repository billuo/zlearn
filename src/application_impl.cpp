#include "application_impl.h"

NAMESPACE_BEGIN

void Application_impl::update_train_stats() {
		RELEASE_ASSERT(!train_info.empty());
		if (train_info.size() == 1) {
				best_epoch = 1;
				model->take_snapshot();
				return;
		}

		auto& this_info = train_info[train_info.size() - 1];
		auto& last_info = train_info[train_info.size() - 2];
		auto& best_info = train_info[best_epoch - 1];
		bool made_progress = false;
		bool is_best = false;
		if (metric) {
				if (Metric::better(metric->type(), this_info.test_metric,
								   last_info.test_metric)) {
						made_progress = true;
				}
				if (Metric::better(metric->type(), this_info.test_metric,
								   best_info.test_metric)) {
						is_best = true;
				}
		} else {
				if (this_info.test_loss < last_info.test_loss) {
						made_progress = true;
				}
				if (this_info.test_loss < best_info.test_metric) {
						is_best = true;
				}
		}
		if (made_progress) {
				bad_epoch_acc = 0;
		} else {
				++bad_epoch_acc;
		}
		if (is_best) {
				best_epoch = this_info.epoch;
				model->take_snapshot();
		}
}

bool Application_impl::should_early_stop() {
		return window > 0 && bad_epoch_acc >= window;
}

NAMESPACE_END
