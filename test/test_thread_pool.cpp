#include <catch2/catch.hpp>

#include "base/thread_pool.h"
using namespace NAMESPACE_NAME;

TEST_CASE("async print") {
  ThreadPool pool(4);
  auto hello_worker = [](int id) { printf("hello worker #%d\n", id); };
  for (int i = 0; i < 3; ++i) {
    auto n_tasks = 8;
    for (int n = 1; n <= n_tasks; ++n) {
      pool.enqueue(hello_worker, n);
    }
    pool.sync(n_tasks);
    printf("hello master\n");
  }
  printf("done\n");
}

std::array<int, 4> A;
TEST_CASE("async sum") {
  ThreadPool pool(4);
  auto compute = [](int* x) {for (int i = 0; i < 5; ++i) {++*x;} };
  for (int i = 0; i < 10; ++i) {
    for (auto& a : A) {
      pool.enqueue(compute, &a);
    }
    pool.sync(A.size());
  }
  auto sum = std::accumulate(A.begin(), A.end(), 0);
  REQUIRE(sum == A.size() * 5 * 10);
}
