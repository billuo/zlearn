#pragma once

#include "exception.hpp"

#include <chrono>
#include <optional>

NAMESPACE_BEGIN

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

struct Timer {
		std::optional<TimePoint> start;
		std::optional<TimePoint> stop;
		void reset() {
				start.reset();
				stop.reset();
		}
		TimePoint tic() {
				start = Clock::now();
				return *start;
		}
		TimePoint toc() {
				stop = Clock::now();
				return *stop;
		}
		template <typename Duration = std::chrono::seconds>
		auto duration() const {
				if (!start) throw Exception("did not tic");
				if (!stop) throw Exception("did not toc");
				return std::chrono::duration_cast<Duration>(*stop - *start);
		}
		float seconds() const {
				using D = std::chrono::duration<float>;
				return duration<D>().count();
		}
		float milliseconds() const {
				using D = std::chrono::duration<float, std::milli>;
				return duration<D>().count();
		}
};

NAMESPACE_END
