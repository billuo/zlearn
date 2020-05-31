#include "thread_pool.h"
#include "exception.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

NAMESPACE_BEGIN

struct ThreadPool_impl {
		void worker() {
				while (true) {
						std::function<void()> task;
						{
								std::unique_lock<std::mutex> lock(task_mutex);
								condition.wait(lock, [this]() {
										return stop || !tasks.empty();
								});
								// allow workers in a stopped pool
								// to still finish remaining tasks if any
								if (stop && tasks.empty()) { return; }
								task = std::move(tasks.front());
								tasks.pop();
						}
						task();
						sync.fetch_add(1);
						sync_condition.notify_one();
				}
		}

		ThreadPool_impl(int n_threads) {
				if (n_threads <= 0)
						throw Exception("n_threads must be positive");
				for (int i = 0; i < n_threads; ++i) {
						workers.emplace_back([this]() { worker(); });
				}
		}
		~ThreadPool_impl() {
				{
						std::unique_lock<std::mutex> lock(task_mutex);
						stop = true;
				}
				condition.notify_all();
				for (auto& worker : workers) {
						worker.join();
				}
		}

		void do_enqueue(std::function<void()> function) {
				if (stop) throw Exception("enqueue stopped ThreadPool");
				{
						std::unique_lock<std::mutex> lock(task_mutex);
						tasks.emplace(std::move(function));
				}
				condition.notify_one();
		}

		std::atomic_bool stop = false;

		// keep track of threads so we can join them
		std::vector<std::thread> workers;

		// the task queue
		std::queue<std::function<void()>> tasks;
		std::mutex task_mutex;

		// synchronization
		std::atomic_int sync = 0;
		std::mutex sync_mutex;
		std::condition_variable condition;
		std::condition_variable sync_condition;
};

ThreadPool::ThreadPool(int n_threads)
: m_impl(std::make_unique<ThreadPool_impl>(n_threads)) {}
ThreadPool::~ThreadPool() = default;

void ThreadPool::do_enqueue(std::function<void()> task) {
		m_impl->do_enqueue(std::move(task));
}
size_t ThreadPool::size() { return m_impl->workers.size(); }
void ThreadPool::sync(int n_wait) {
		std::unique_lock lock(m_impl->sync_mutex);
		m_impl->sync_condition.wait(lock,
									[&]() { return m_impl->sync == n_wait; });
		m_impl->sync = 0;
}

std::vector<size_t> ThreadPool::split_task(size_t total, size_t n_share) {
		std::vector<size_t> ret;
		size_t chunk = total / n_share;
		for (size_t i = 0; i < n_share; ++i) {
				ret.push_back(chunk * i);
		}
		ret.push_back(total);
		ASSERT(ret.size() == n_share + 1);
		return ret;
}
void ThreadPool::stop() { m_impl.reset(); }

NAMESPACE_END
