#include "csv2ffm.h"
#include "base/defer.hpp"
#include "base/enum_db.h"
#include "base/io_util.h"
#include "base/logger.h"
#include "base/str-hash.h"
#include "data/data_set.h"

#include <tsl/robin_map.h>
using IndexMap = tsl::robin_map<std::string, size_t>;

struct GroupParser {
	public:
		std::vector<size_t> result;
		void parse(const std::string s) {
				p = s.data();
				end = s.data() + s.size();
				curr_group = 0;
				while (p < end) {
						parse_group();
						++curr_group;
				}
		}

	private:
		const char* p;
		const char* end;
		size_t curr_group;

		void parse_group() {
				while (p < end) {
						parse_component();
						if (p >= end) break;
						auto c = *p++;
						switch (c) {
						case ',': continue;
						case ';': return;
						default:
								THROW("unexpected charcter %d='%c'", int(c), c);
						}
				}
		}
		void parse_component() {
				char* q;
				int a = std::strtol(p, &q, 10);
				p = q;
				if (*p == '-') {
						int b = std::strtol(p + 1, &q, 10);
						p = q;
						check_size(b);
						for (int i = a; i <= b; ++i) {
								result[i - 1] = curr_group;
						}
				} else {
						check_size(a);
						result[a - 1] = curr_group;
				}
		}
		void check_size(size_t sz) {
				if (result.size() < sz) { result.resize(sz); }
		}
};

NAMESPACE_BEGIN

static real_t to_real(const std::string& s) {
		static_assert(std::is_same_v<real_t, float>);
		return std::stof(s);
}

using Encoding = CSV2FFM::Encoding;
ENUM_DB_DEFINITION(Encoding) = {
	{Encoding::Numeral, "numeral"},
	{Encoding::Categorical, "categorical"},
};

void CSV2FFM::convert(const std::string& csv_file,
					  const std::string& ffm_file,
					  const std::string& encode,
					  const std::string& group) {
		m_input = must_open_file(csv_file.c_str(), std::ios::in);
		defer([this] { m_input = nullptr; });

		auto espec = get_encoding(encode);
		auto gspec = get_grouping(group);
		get_info();
		if (espec.size() != n_features) {
				if (encode.empty()) {
						espec.resize(n_features, Numeral);
				} else {
						THROW("encoding string specifies %lu features.",
							  espec.size());
				}
		}
		if (gspec.size() != n_features) {
				if (group.empty()) {
						gspec.resize(n_features);
						for (size_t i = 0; i < n_features; ++i) {
								gspec[i] = i + 1;
						}
				} else {
						THROW("grouping string specifies %lu features.",
							  gspec.size());
				}
		}

		std::vector<Entry> entries;
		entries.resize(n_rows);
		if (per_column) {
				// sanity check
				for (int i = 0; i < 10; ++i) {
						std::vector<std::string> row(n_cols);
						parse_record(row.data(), n_cols);
						for (size_t c = has_label; c < n_cols; ++c) {
								if (row[c].empty()) continue;
								try {
										switch (espec[c - has_label]) {
										case Categorical:
												// nothing to check
												break;
										case Numeral:
												// should be convertible to real number
												(void) to_real(row[c]);
												break;
										}
								} catch (const std::exception& e) {
										logger::error("sanity check failed:"
													  "feature#{}: {}",
													  c - has_label + 1,
													  e.what());
										std::exit(EXIT_FAILURE);
								}
						}
						if (not next_row()) break;
				}

				restart_input();
				std::vector<std::string> column;
				size_t idx = 0;
				for (size_t c = 0; c < n_cols; ++c) {
						size_t f = c - has_label;
						if (has_label && c == 0) {
								logger::info("processing label");
						} else {
								logger::info(
									"processing feature#{} [name={}, "
									"{},field={}]",
									f + 1,
									has_header ? headers[c].c_str() : "?",
									EnumDB<Encoding>::to_string(espec[f])
										.c_str(),
									gspec[f]);
						}
						do {
								parse_record(&column.emplace_back(), 1, c);
						} while (next_row());
						if (column.size() != n_rows) {
								THROW("column#%lu does not have enough values",
									  c);
						}
						if (has_label && c == 0) {
								for (size_t i = 0; i < n_rows; ++i) {
										entries[i].label = to_real(column[i]);
								}
						} else {
								switch (espec[f]) {
								case Categorical: {
										IndexMap idx_map;
										for (auto& v : column) {
												if (idx_map.count(v) == 0) {
														idx_map[v] = idx++;
												}
										}
										for (size_t i = 0; i < n_rows; ++i) {
												if (column[i].empty()) continue;
												entries[i]
													.features.emplace_back(
														gspec[f],
														idx_map[column[i]], 1);
										}
										break;
								}
								case Numeral:
										for (size_t i = 0; i < n_rows; ++i) {
												if (column[i].empty()) continue;
												entries[i]
													.features.emplace_back(
														gspec[f], idx,
														to_real(column[i]));
										}
										++idx;
										break;
								}
						}
						column.clear();
						restart_input();
				}
		} else {
				std::vector<std::string> rows;
				rows.reserve(n_rows * n_cols);
				std::vector<IndexMap> idx_maps;
				idx_maps.resize(n_features);
				do {
						rows.resize(rows.size() + n_cols);
						auto row = rows.data() + rows.size() - n_cols;
						parse_record(row, n_cols);
						for (size_t f = 0; f < n_features; ++f) {
								auto& feat = row[f + has_label];
								switch (espec[f]) {
								case Categorical: {
										auto& idx_map = idx_maps[f];
										if (idx_map.count(feat) == 0) {
												idx_map[feat] = idx_map.size();
										}
								} break;
								case Numeral: break;
								}
						}
				} while (next_row());
				logger::info("all parsed");
				std::vector<size_t> idx_base(n_features, 1);
				for (size_t f = 1; f < n_features; ++f) {
						switch (espec[f - 1]) {
						case Categorical:
								idx_base[f] =
									idx_base[f - 1] + idx_maps[f - 1].size();
								break;
						case Numeral: idx_base[f] = idx_base[f - 1] + 1; break;
						}
				}
				logger::info("converting...");
				for (size_t i = 0; i < n_rows; ++i) {
						auto row = rows.data() + i * n_cols;
						if (has_label) entries[i].label = to_real(row[0]);
						auto& features = entries[i].features;
						for (size_t f = 0; f < n_features; ++f) {
								auto& feat = row[f + has_label];
								if (feat.empty()) continue;
								switch (espec[f]) {
								case Categorical:
										features.emplace_back(
											gspec[f],
											idx_base[f] + idx_maps[f][feat], 1);
										break;
								case Numeral:
										features.emplace_back(gspec[f],
															  idx_base[f],
															  to_real(feat));
										break;
								}
						}
				}
		}

		// output
		auto file = must_open_file(ffm_file.c_str(), std::ios_base::out);
		file->exceptions(std::ios::badbit | std::ios::failbit);
		for (auto& e : entries) {
				if (has_label) *file << e.label << ' ';
				for (auto& f : e.features) {
						*file << f.field_id << ':' << f.id << ':' << f.value;
						if (&f != &e.features.back()) *file << ' ';
				}
				*file << std::endl;
		}
}

void CSV2FFM::get_info() {
		if (has_header) {
				parse_header(&headers);
				n_cols = headers.size();
				accept(CR);
				expect(LF);
		}
		m_record_begin = m_input->tellg();
		logger::debug("record begins at character {}", m_record_begin);
		if (!has_header) {
				headers.clear();
				n_cols = parse_record();
				restart_input();
		}
		n_rows = 0;
		do {
				parse_record();
				++n_rows;
		} while (next_row());
		restart_input();
		n_features = n_cols - has_label;
		logger::info("header {}; detected {} columns, {} rows, {} features",
					 has_header ? "present" : "absent", n_cols, n_rows,
					 n_features);
}
std::vector<size_t> CSV2FFM::get_grouping(const std::string& s) {
		GroupParser parser;
		parser.parse(s);
		return parser.result;
}
std::vector<CSV2FFM::Encoding> CSV2FFM::get_encoding(const std::string& s) {
		std::vector<Encoding> ret;
		const char* p = s.data();
		const char* end = s.data() + s.size();
		char* q;
		while (p < end) {
				auto c = *p++;
				int repeat = std::strtol(p, &q, 10);
				if (!repeat) {
						repeat = 1;
				} else {
						p = q;
				}
				Encoding e;
				switch (c) {
				case 'c': e = Categorical; break;
				case 'n': e = Numeral; break;
				default: THROW("unknown encoding: '%c'", *p);
				}
				for (int i = 0; i < repeat; ++i) {
						ret.push_back(e);
				}
		}
		return ret;
}

NAMESPACE_END
