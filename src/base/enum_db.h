#pragma once

#include "common.h"
#include "exception.hpp"

#include <vector>

NAMESPACE_BEGIN

template <typename E, typename = std::enable_if<std::is_enum_v<E>>>
class EnumDB {
		static std::vector<std::pair<E, String>> db;
		using U = std::underlying_type_t<E>;

	public:
		template <typename OutputIt>
		static void get_names(OutputIt out) {
				for (const auto& pair : db) {
						*out++ = pair.second.c_str();
				}
		}
		static E to_enum(const String& name) { return to_enum(name.c_str()); }
		static E to_enum(const char* name) {
				for (const auto& pair : db) {
						if (pair.second == name) return pair.first;
				}
				THROW("can not convert '%s' to enum", name);
		}
		static String to_string(E e) {
				for (const auto& pair : db) {
						if (pair.first == e) return pair.second;
				}
				THROW("can not convert '%d' to string", static_cast<U>(e));
		}
};
#define DEFINE_STATIC_MEMBER(cls, m) decltype(cls::m) cls::m
#define ENUM_DB_DEFINITION(E)                                                  \
		template <>                                                            \
		DEFINE_STATIC_MEMBER(EnumDB<E>, db)

template <typename E, typename = std::enable_if<std::is_enum_v<E>>>
inline auto to_string(E e) {
		return EnumDB<E>::to_string(e);
}
template <typename E, typename = std::enable_if<std::is_enum_v<E>>>
inline auto to_enum(const String& name) {
		return EnumDB<E>::to_enum(name);
}
template <typename E, typename = std::enable_if<std::is_enum_v<E>>>
inline auto to_enum(const char* name) {
		return EnumDB<E>::to_enum(name);
}

NAMESPACE_END
