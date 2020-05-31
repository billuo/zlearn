#include "application_impl.h"
#include "base/defer.hpp"
#include "base/enum_db.h"
#include "base/io_util.h"
#include "base/logger.h"
#include "base/timer.h"
#include "data/sampler.h"

NAMESPACE_BEGIN

namespace {

String readable_size(size_t n_bytes) {
		constexpr auto KB = 1024LL;
		constexpr auto MB = 1024LL * 1024;
		constexpr auto GB = 1024LL * 1024 * 1024;
		constexpr auto TB = 1024LL * 1024 * 1024 * 1024;
		if (n_bytes < KB) return String::printf("%d B", int(n_bytes));
		double bytes = n_bytes;
		if (n_bytes < MB) return String::printf("%.3f KB", bytes / KB);
		if (n_bytes < GB) return String::printf("%.3f MB", bytes / MB);
		if (n_bytes < TB) return String::printf("%.3f GB", bytes / GB);
		return String::printf("%.3f TB", bytes / TB);
}

} // namespace

void Application_impl::train(Optimizer& optimizer,
							 Sampler& train,
							 std::shared_ptr<Sampler> test) {
		Timer train_timer;
		train_timer.tic();
		defer([&] {
				train_timer.toc();
				logger::info("finish training... {:.2} seconds elapsed",
							 train_timer.seconds());
		});

		logger::info("start training...");
		{
				Timer init_timer;
				init_timer.tic();
				defer([&] {
						init_timer.toc();
						logger::info("model initialization took {:.2} seconds",
									 init_timer.seconds());
				});

				train.restart();
				model->initialize(train, optimizer.extras());
				auto n_bytes = model->n_parameters() * sizeof(real_t);
				logger::info("model size: {}", readable_size(n_bytes).c_str());
		}

		if (metric && !test) {
				logger::warn("metric specified but no test data provided");
		}

		constexpr auto header_fmt = "{:<6}|{:^20}|{:^20}|{:^20}|{:>8}";
		constexpr auto progress_fmt = "{:6}|{:>20.4}|{:>20.4}|{:>20.4}|{:8.2}";
		String loss_str = EnumDB<Loss>::to_string(loss);
		String train_loss_str = String::printf("train %s", loss_str.c_str());
		String test_loss_str =
			test ? String::printf("test %s", loss_str.c_str()) : "(no test)";
		String metric_str = metric ? metric->name() : "(no metric)";
		logger::info(header_fmt, "epoch", train_loss_str.c_str(),
					 test_loss_str.c_str(), metric_str.c_str(), "seconds");
		train_info.clear();
		for (int n = 1; n <= n_epochs; ++n) {
				auto& info = train_info.emplace_back();
				info.epoch = n;

				Timer epoch_timer;
				epoch_timer.tic();
				train.restart();
				train.shuffle();
				optimizer.set_epoch(n);
				info.train_loss =
					model->update(loss, optimizer, train, *thread_pool);
				if (test) {
						test->restart();
						info.test_loss =
							model->evaluate(loss, *test, *thread_pool);
						if (metric) {
								metric->reset();
								test->restart();
								metric->evaluate(*model, *test);
								info.test_metric = metric->value();
						}
						update_train_stats();
				}
				epoch_timer.toc();
				info.seconds = epoch_timer.seconds();

				logger::info(progress_fmt, n, info.train_loss, info.test_loss,
							 info.test_metric, epoch_timer.seconds());

				if (should_early_stop()) {
						logger::info("early stopped at epoch {}", n);
						break;
				}
		}
		logger::info("restoring to best model at epoch {}...", best_epoch);
		logger::info("best test loss={}; best metric={}",
					 train_info[best_epoch - 1].test_loss,
					 train_info[best_epoch - 1].test_metric);
		model->restore_snapshot();

		if (summary) {
				*summary << "tr_loss,tt_loss,tt_metric,seconds,is_best"
						 << std::endl;
				for (auto& info : train_info) {
						*summary
							<< fmt::format("{},{},{},{},{}", info.train_loss,
										   info.test_loss, info.test_metric,
										   info.seconds,
										   info.epoch == best_epoch ? "1" : "")
							<< std::endl;
				}
		}
}

NAMESPACE_END
