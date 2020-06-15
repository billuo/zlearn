#pragma once

#include "base/exception.hpp"
#include "entry.h"

NAMESPACE_BEGIN

class DataSet : public std::enable_shared_from_this<DataSet> {
	public:
		static std::shared_ptr<DataSet>
		from_file(String p, bool remove_zeros, String separators = "");

		static std::shared_ptr<DataSet>
		dummy(int n_entries, int n_features, int n_fields = 1);

		DataSet() = default;
		~DataSet() = default;
		DataSet(const DataSet&) = default;
		DataSet(DataSet&&) = default;
		DataSet& operator=(const DataSet&) = default;
		DataSet& operator=(DataSet&&) = default;

		DataSet(Entries entries, bool has_label)
		: m_entries(std::move(entries)), m_has_label(has_label) {}

		void reserve(size_t capacity) { m_entries.reserve(capacity); }
		void has_label(bool b) { m_has_label = b; }
		bool has_label() const { return m_has_label; }
		size_t size() const { return m_entries.size(); }
		const auto& entries() const { return m_entries; }

		auto& add_entry() {
				return m_entries.emplace_back(std::make_shared<Entry>());
		}
		void sort_entries() {
				for (auto& e : m_entries) {
						e->sort_features();
				}
		}

		std::shared_ptr<DataSet>
		train_test_split(int train_share, int test_share, bool shuffle);

		void serialize(String filename);
		void deserialize(String filename);
		void serialize_txt(String filename);

		static size_t max_feature_id(Entries& entries) {
				size_t max_id = 0;
				for (const auto& entry : entries) {
						for (auto& feature : entry->features) {
								max_id = std::max(max_id, feature.id);
						}
				}
				return max_id;
		}
		static size_t max_field_id(Entries& entries) {
				size_t max_id = 0;
				for (const auto& entry : entries) {
						for (auto& feature : entry->features) {
								max_id = std::max(max_id, feature.field_id);
						}
				}
				return max_id;
		}

		std::shared_ptr<Entry> operator[](size_t index) {
				return m_entries[index];
		}
		std::shared_ptr<const Entry> operator[](size_t index) const {
				return m_entries[index];
		}
		Feature& operator()(size_t index, size_t field) {
				return m_entries[index]->features[field];
		}
		const Feature& operator()(size_t index, size_t field) const {
				return m_entries[index]->features[field];
		}

		bool operator==(const DataSet& rhs) const;
		bool operator!=(const DataSet& rhs) const { return !(rhs == *this); }

	private:
		Entries m_entries; // NOTE: none of the entry should be nullptr
		bool m_has_label = false;
		// TODO to speed up data set comparison
		u64 m_hash_1 = 0;
		u64 m_hash_2 = 0;
};

inline bool DataSet::operator==(const DataSet& rhs) const {
		if (m_hash_1 != rhs.m_hash_1 || m_hash_2 != rhs.m_hash_2) return false;
		if (m_has_label != rhs.m_has_label || size() != rhs.size())
				return false;
		for (size_t i = 0; i < size(); ++i) {
				auto e1 = m_entries[i];
				auto e2 = rhs.m_entries[i];
				if (e1 != e2 && *e1 != *e2) return false;
		}

		return true;
}

NAMESPACE_END
