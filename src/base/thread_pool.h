#pragma once

#include "common.h"

#include <functional>
#include <future>

NAMESPACE_BEGIN

struct ThreadPool_impl;
class ThreadPool {
	public:
		ThreadPool(int n_threads);
		~ThreadPool();
		size_t size();

		void stop();

		template <typename F, typename... Args>
		using future_result_t = std::future<std::invoke_result_t<F, Args...>>;
		template <typename F, typename... Args>
		future_result_t<F, Args...> enqueue(F&& f, Args&&... args) {
				using R = std::invoke_result_t<F, Args...>;
				auto task = std::make_shared<std::packaged_task<R()>>(
					std::bind(std::forward<F>(f), std::forward<Args>(args)...));
				std::future<R> res = task->get_future();
				do_enqueue([task]() { (*task)(); });
				return res;
		}
		void sync(int n_wait);

		static std::vector<size_t> split_task(size_t total, size_t n);

	private:
		std::unique_ptr<ThreadPool_impl> m_impl;
		void do_enqueue(std::function<void()> task);
};

NAMESPACE_END
