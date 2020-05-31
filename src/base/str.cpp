#include "str.h"
#include "exception.hpp"
#include "str-inl.h"

#include <cstdlib>

NAMESPACE_BEGIN

String::String() : sds(sdsempty()) {}
String::String(const char* str) : sds(sdsnew(str)) {}
String::String(const char* str, std::size_t len) : sds(sdsnewlen(str, len)) {}
String::String(const String& other) : sds(sdsdup(other.sds)) {}
String::String(String&& other) : sds(other.sds) {
		other.sds = sdsempty();
		if (!other.sds) throw Exception("sdsempty failed");
}
String& String::operator=(const String& rhs) {
		if (this != &rhs) {
				String t(rhs);
				swap(t);
		}
		return *this;
}
String::~String() {
		sdsfree(sds);
		sds = nullptr;
}

std::size_t String::size() const { return sdslen(sds); }
std::size_t String::capacity() const { return sdsalloc(sds); }
std::size_t String::hash() const {
		std::size_t h = 5381;
		for (auto p = end(); p > begin(); p -= 5) {
				h = h * 33 ^ *p;
		}
		return h;
}

void String::resize(std::size_t new_size) {
		if (new_size < size()) {
				sds[new_size] = '\0';
				sdssetlen(sds, new_size);
		} else if (new_size > size()) {
				sds = sdsgrowzero(sds, new_size);
				if (!sds) throw Exception("sdsgrowzero failed");
		}
}
void String::reserve_for(std::size_t size) {
		sds = sdsMakeRoomFor(sds, size);
		if (!sds) throw Exception("sdsMakeRoomFor failed");
}
/// @brief select a range of this String
/// @param i index of the first char to include. count from 0.
/// @param j index of the last char to include. count from 0.
/// @return a new copy of the sub-string.
/// @details both i and j can be negative, which means to count backwards.
/// @example String("abc").substr(1, -1) == String("bc");
String String::substr(int i, int j) {
		String str(*this);
		sdsrange(str.sds, i, j);
		return str;
}

void String::push_back(char rhs) {
		sds = sdscatlen(sds, &rhs, 1);
		if (!sds) throw Exception("sdscatlen failed");
}
void String::pop_back() {
		if (empty()) throw Exception("pop_back(): empty String");
		resize(size() - 1);
}

int String::compare(const String& s1, const String& s2) {
		return sdscmp(s1.sds, s2.sds);
}

String& String::operator+=(const char* rhs) {
		sds = sdscat(sds, rhs);
		if (!sds) throw Exception("sdscat failed");
		return *this;
}
String& String::operator+=(const String& rhs) {
		sds = sdscatsds(sds, rhs.sds);
		if (!sds) throw Exception("sdscatsds failed");
		return *this;
}

void String::vappendf(const char* fmt, va_list ap) {
		sds = sdscatvprintf(sds, fmt, ap);
		if (!sds) throw Exception("sdscatvprintf failed");
}
void String::appendf(const char* fmt, ...) {
		va_list ap;
		va_start(ap, fmt);
		vappendf(fmt, ap);
		va_end(ap);
}
String String::printf(const char* fmt, ...) {
		String s;
		va_list ap;
		va_start(ap, fmt);
		s.vappendf(fmt, ap);
		va_end(ap);
		return s;
}
float String::to_float() const {
		char* end;
		float ret = std::strtof(sds, &end);
		if (end == sds) throw Exception("failed to convert %s to float", sds);
		return ret;
}
double String::to_double() const {
		char* end;
		double ret = std::strtod(sds, &end);
		if (end == sds) throw Exception("failed to convert %s to double", sds);
		return ret;
}

NAMESPACE_END

#include "str-inl.cpp" // in hope of better optimization
