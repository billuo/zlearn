#pragma once

#include "application.h"
#include "base/thread_pool.h"
#include "model/metric.h"
#include "model/model.h"
#include "optimizer/optimizer.h"

NAMESPACE_BEGIN

struct TrainInfo {
        int epoch = 0;
        real_t train_loss = NAN;
        real_t test_loss = NAN;
        real_t test_metric = NAN;
        float seconds = 0;
};


struct Application_impl {
		// commonly shared
		std::shared_ptr<ThreadPool> thread_pool;
		std::unique_ptr<Model> model;
		std::shared_ptr<std::fstream> summary;

		// for training
		int n_epochs = 10;
		Loss loss;
		std::optional<Metric> metric;
		int window = 3;

        std::vector<TrainInfo> train_info;
		void train(Optimizer&,
				   Sampler& train,
				   std::shared_ptr<Sampler> test = nullptr);

		// for predict
		std::vector<real_t> predicted;

		void predict(Sampler&);

	private:
		int best_epoch;
		void update_train_stats();

		int bad_epoch_acc = 0; // should stop when it exceeds window
		bool should_early_stop();
};

NAMESPACE_END
