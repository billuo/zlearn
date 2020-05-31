#include "data_set.h"
#include "base/io_util.h"
#include "base/logger.h"
#include "base/math.h"
#include "base/random.h"
#include "base/serial.h"

#include <algorithm>
#include <cstring>

NAMESPACE_BEGIN

std::shared_ptr<DataSet> DataSet::train_test_split(int train_share,
												   int test_share) {
		std::shuffle(m_entries.begin(), m_entries.end(), random_generator());
		auto split_at = m_entries.begin()
			+ m_entries.size() * test_share / (test_share + train_share);
		Entries test_entries(std::make_move_iterator(split_at),
							 std::make_move_iterator(m_entries.end()));
		m_entries.erase(split_at, m_entries.end());
		return std::make_shared<DataSet>(test_entries, m_has_label);
}

void DataSet::serialize(String filename) {
		auto pfile = must_open_file(filename.c_str(),
									std::ios_base::binary | std::ios_base::out);
		auto& file = *pfile;
		file.exceptions(std::ios::badbit | std::ios::failbit);

		serialize_to(file, m_hash_1);
		serialize_to(file, m_hash_2);
		serialize_to(file, m_has_label);
		serialize_to(file, m_entries.size());
		for (const auto& entry : m_entries) {
				if (m_has_label) { serialize_to(file, entry->label); }
				serialize_to(file, entry->features.size());
				for (auto& feature : entry->features) {
						serialize_to(file, feature.field_id);
						serialize_to(file, feature.id);
						serialize_to(file, feature.value);
				}
		}
}

void DataSet::deserialize(String filename) {
		auto pfile = must_open_file(filename.c_str(),
									std::ios_base::binary | std::ios_base::in);
		auto& file = *pfile;
		file.exceptions(std::ios::badbit | std::ios::failbit);

		deserialize_from(file, m_hash_1);
		deserialize_from(file, m_hash_2);
		deserialize_from(file, m_has_label);
		size_t n_entries;
		deserialize_from(file, n_entries);
		reserve(n_entries);
		for (size_t i = 0; i < n_entries; ++i) {
				auto&& entry = add_entry();
				if (m_has_label) { deserialize_from(file, entry->label); }
				size_t n_features;
				deserialize_from(file, n_features);
				entry->features.resize(n_features);
				for (auto& feature : entry->features) {
						deserialize_from(file, feature.field_id);
						deserialize_from(file, feature.id);
						deserialize_from(file, feature.value);
				}
		}
}

void DataSet::serialize_txt(String filename) {
		auto pfile = must_open_file(filename.c_str(), std::ios_base::out);
		auto& file = *pfile;
		file.exceptions(std::ios::badbit | std::ios::failbit);

		for (auto& e : m_entries) {
				if (has_label()) file << e->label << ' ';
				for (auto& f : e->features) {
						file << f.field_id << ':' << f.id << ':' << f.value;
						if (&f != &e->features.back()) file << ' ';
				}
				file << std::endl;
		}
}

enum class FileFormat {
		CSV,
		SVM,
		FFM,
};

static std::shared_ptr<DataSet>
parse_csv(std::fstream& fs, bool remove_zeros, String separators) {
		auto ret = std::make_shared<DataSet>();
		ret->has_label(true);
		std::string line;
		while (std::getline(fs, line)) {
				auto&& entry = ret->add_entry();
				auto token = std::strtok(line.data(), separators.c_str());
				entry->label = std::atof(token);
				size_t feature_id = 0;
				size_t field_id = 0;
				token = std::strtok(nullptr, separators.c_str());
				while (token) {
						real_t value = std::atof(token);
						if (!(remove_zeros && value == 0)) {
								entry->features.emplace_back(
									field_id++, feature_id++, value);
						}
						token = std::strtok(nullptr, separators.c_str());
				}
		}
		return ret;
}
static std::shared_ptr<DataSet> parse_svm(std::fstream& fs,
										  bool remove_zeros,
										  String separators,
										  bool has_label) {
		auto ret = std::make_shared<DataSet>();
		ret->has_label(has_label);
		std::string line;
		while (std::getline(fs, line)) {
				auto&& entry = ret->add_entry();
				auto token = std::strtok(line.data(), separators.c_str());
				if (has_label) {
						entry->label = std::atof(token);
						token = std::strtok(nullptr, separators.c_str());
				}
				while (token) {
						std::string_view unit(token, strlen(token));
						auto colon1 = unit.find_first_of(':');
						ASSERT(unit.find_first_of(':', colon1 + 1)
							   == std::string_view::npos);
						token[colon1] = '\0';
						size_t feature_id = std::atoll(token);
						real_t value = std::atof(token + colon1 + 1);
						if (!(remove_zeros && value == 0)) {
								entry->features.emplace_back(feature_id, value);
						}
						token = std::strtok(nullptr, separators.c_str());
				}
		}
		return ret;
}
static std::shared_ptr<DataSet> parse_ffm(std::fstream& fs,
										  bool remove_zeros,
										  String separators,
										  bool has_label) {
		auto ret = std::make_shared<DataSet>();
		ret->has_label(has_label);
		std::string line;
		while (std::getline(fs, line)) {
				auto&& entry = ret->add_entry();
				auto token = std::strtok(line.data(), separators.c_str());
				if (has_label) {
						entry->label = std::atof(token);
						token = std::strtok(nullptr, separators.c_str());
				}
				while (token) {
						std::string_view unit(token, strlen(token));
						auto colon1 = unit.find_first_of(':');
						auto colon2 = unit.find_first_of(':', colon1 + 1);
						ASSERT(unit.find_first_of(':', colon2 + 1)
							   == std::string_view::npos);
						token[colon1] = token[colon2] = '\0';
						size_t field_id = std::atoll(token);
						size_t feature_id = std::atoll(token + colon1 + 1);
						real_t value = std::atof(token + colon2 + 1);
						if (!(remove_zeros && value == 0)) {
								entry->features.emplace_back(field_id,
															 feature_id, value);
						}
						token = std::strtok(nullptr, separators.c_str());
				}
		}
		return ret;
}

std::shared_ptr<DataSet>
DataSet::from_file(String p, bool remove_zeros, String separators) {
		filesystem::path path(p.c_str());
		auto openmode = std::ios::in;
		auto pfile = must_open_file(path, openmode);
		auto& file = *pfile;
		logger::info("loading data from {}", p.c_str());
		if (remove_zeros) {
				logger::warn("feature of value 0 will be ignored");
		}

		if (separators.empty()) {
				int count[256] = {};
				int c;
				while ((c = file.get()) != EOF) {
						count[c]++;
				}
				file.clear();
				file.seekg(0, std::ios::beg);
				int freq_blank = ' ';
				int n_freq_blank = 0;
				for (c = 0; c < 256; ++c) {
						if (std::isblank(c) && count[c] > n_freq_blank) {
								freq_blank = c;
								n_freq_blank = count[c];
						}
				}
				separators += freq_blank;
				if (separators.empty())
						throw Exception("failed to detect separators");
		}

		FileFormat format;
		bool has_label;
		auto line = peek_first_line(file);
		auto word1 = std::strtok(line.data(), separators.c_str());
		int n_colon_1 = std::count(word1, word1 + strlen(word1), ':');
		auto word2 = std::strtok(nullptr, separators.c_str());
		if (word2 == nullptr) {
				has_label = false;
				switch (n_colon_1) {
				case 0: format = FileFormat::CSV; break;
				case 1: format = FileFormat::SVM; break;
				case 2: format = FileFormat::FFM; break;
				default: THROW("can not detect format: '%s'", word1);
				}
		} else {
				int n_colon_2 = std::count(word2, word2 + strlen(word2), ':');
				switch (n_colon_2) {
				case 0:
						format = FileFormat::CSV;
						has_label = true;
						break;
				case 1:
						format = FileFormat::SVM;
						has_label =
							std::count(word1, word1 + strlen(word1), ':') == 0;
						break;
				case 2:
						format = FileFormat::FFM;
						has_label =
							std::count(word1, word1 + strlen(word1), ':') == 0;
						break;
				default: THROW("can not detect format: '%s'", word2);
				}
		}
		switch (format) {
		case FileFormat::CSV:
				logger::info("detected CSV format");
				return parse_csv(file, remove_zeros, separators);
		case FileFormat::SVM:
				logger::info("detected SVM format");
				return parse_svm(file, remove_zeros, separators, has_label);
		case FileFormat::FFM:
				logger::info("detected FFM format");
				return parse_ffm(file, remove_zeros, separators, has_label);
		default: UNREACHABLE("file format");
		}
}

std::shared_ptr<DataSet>
DataSet::dummy(int n_entries, int n_features, int n_fields) {
		RELEASE_ASSERT(n_fields > 0);
		RELEASE_ASSERT(n_features > 0);
		auto ret = std::make_shared<DataSet>();
		int id = 0;
		for (int i = 0; i < n_entries; ++i) {
				auto& e = ret->add_entry();
				e->label = i % 2;
				for (int f = 0; f < n_fields; ++f) {
						e->features.emplace_back(f, id, 0);
						id = (id + 1) % n_features;
				}
		}
		return ret;
}

NAMESPACE_END
