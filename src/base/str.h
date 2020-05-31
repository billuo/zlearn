#pragma once

#include "common.h"

#include <cctype>
#include <cstdarg>
#include <cstdio>

NAMESPACE_BEGIN

class String {
	public:
		String();
		String(const char* str);
		String(const char* str, std::size_t len);
		String(const String&);
		String(String&& other);
		String& operator=(const String&);
		String& operator=(String&& other) {
				String t(std::move(other));
				swap(t);
				return *this;
		}
		~String();

		void swap(String& other) noexcept { std::swap(sds, other.sds); }

		bool empty() const { return size() == 0; }
		std::size_t size() const;
		std::size_t capacity() const;
		char* c_str() { return sds; }
		const char* c_str() const { return sds; }
		std::size_t hash() const;

		char* begin() { return sds; }
		const char* begin() const { return sds; }
		char* end() { return sds + size(); }
		const char* end() const { return sds + size(); }

		void clear() { resize(0); }
		void resize(std::size_t new_size);
		void reserve_for(std::size_t size);
		String substr(int i, int j);
		void push_back(char);
		void pop_back();

		bool operator==(const String& rhs) const {
				return compare(*this, rhs) == 0;
		}
		bool operator!=(const String& rhs) const {
				return compare(*this, rhs) != 0;
		}
		bool operator<(const String& rhs) const {
				return compare(*this, rhs) < 0;
		}
		bool operator>(const String& rhs) const {
				return compare(*this, rhs) > 0;
		}
		bool operator<=(const String& rhs) const {
				return compare(*this, rhs) <= 0;
		}
		bool operator>=(const String& rhs) const {
				return compare(*this, rhs) >= 0;
		}
		static int compare(const String& s1, const String& s2);

		String& operator+=(char rhs) {
				push_back(rhs);
				return *this;
		}
		String operator+(char rhs) const& { return String(*this) += rhs; }
		String operator+(char rhs) && { return std::move(*this += rhs); }
		String& operator+=(const char* rhs);
		String operator+(const char* rhs) const& {
				return String(*this) += rhs;
		}
		String operator+(const char* rhs) && { return std::move(*this += rhs); }
		friend String operator+(const char* lhs, const String& rhs) {
				return String(lhs) + rhs;
		}
		String& operator+=(const String& rhs);
		String operator+(const String& rhs) const& {
				return String(*this) += rhs;
		}
		String operator+(const String& rhs) && {
				return std::move(*this += rhs);
		}

		static String printf(const char* fmt, ...) ATTR_FORMAT_PRINTF(1, 2);
		void vappendf(const char* fmt, va_list ap);
		void appendf(const char* fmt, ...) ATTR_FORMAT_PRINTF(2, 3);

		template <typename Os>
		friend Os& operator<<(Os& os, const String& str) {
				os.write(str.c_str(), str.size());
				return os;
		}

		float to_float() const;
		double to_double() const;
		real_t to_real() const {
				static_assert(std::is_same_v<real_t, float>);
				return to_float();
		}

	private:
		char* sds;
};

template <typename IStream>
IStream& operator>>(IStream& is, String& str) {
		str.clear();
		auto c = is.get();
		while (c != EOF && !std::isspace(c)) {
				str.push_back(c);
				c = is.get();
		}
		return is;
}

NAMESPACE_END
