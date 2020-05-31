#pragma once

#include "config.h"
#include "macros.h"

#include <functional>
#include <tuple>

NAMESPACE_BEGIN

template <typename F, typename... Args>
class Deferred {
		static_assert(std::is_invocable_v<F, Args...>);
		using R = std::invoke_result_t<F, Args...>;

		std::function<R(Args...)> func;
		std::tuple<Args...> args;

	public:
		explicit Deferred(F&& f, Args&&... args)
		: func(std::forward<F>(f))
		, args(std::forward_as_tuple(std::forward<Args>(args)...)) {}
		~Deferred() { std::apply(func, args); }
};
template <typename F, typename... Args>
Deferred(F&& f, Args&&... args) -> Deferred<F, Args...>;

NAMESPACE_END

#define defer(f, ...)                                                          \
		auto CONCAT(_deferred_, __COUNTER__) =                                 \
			::NAMESPACE_NAME::Deferred(f, ##__VA_ARGS__)
