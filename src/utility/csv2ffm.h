#pragma once

#include "base/exception.hpp"

#include <iostream>
#include <vector>

NAMESPACE_BEGIN

class CSV2FFM {
		using Symbol = std::ios::int_type;
		using Headers = std::vector<std::string>;

		static constexpr Symbol DQUOTE = '"';
		static constexpr Symbol CR = '\x0d';
		static constexpr Symbol LF = '\x0a';
		class UnexpectedSymbol : public Exception {
			public:
				UnexpectedSymbol(Symbol expected, Symbol got)
				: Exception("expected %c, got %c", char(expected), char(got)) {}
		};

	public:
		enum Encoding {
				Categorical,
				Numeral,
		};

		Symbol separator;
		bool has_label;
		bool has_header;
		bool per_column = false;

		explicit CSV2FFM(Symbol separator, bool has_label, bool has_header)
		: separator(separator), has_label(has_label), has_header(has_header) {}

		void convert(const std::string& csv_file,
					 const std::string& ffm_file,
					 const std::string& encode,
					 const std::string& group);

		Headers headers;
		size_t n_cols;
		size_t n_features;
		size_t n_rows;

	private:
		std::shared_ptr<std::iostream> m_input = nullptr;
		std::ios::pos_type m_record_begin;

		void restart_input() {
				m_input->clear();
				m_input->seekg(m_record_begin, std::ios::beg);
		}
		bool accept(Symbol symbol) {
				if (m_input->peek() != symbol) return false;
				m_input->get();
				return true;
		}
		void expect(Symbol symbol) {
				if (!accept(symbol))
						throw UnexpectedSymbol(symbol, m_input->peek());
		}

		// end current row and move on.
		// returns true if there is really a next row.
		bool next_row() {
				accept(CR);
				if (accept(LF)) {
						if (accept(EOF)) return false;
				} else if (accept(EOF)) {
						return false;
				}
				return true;
		}

		void parse_header(Headers* headers) {
				if (headers) {
						headers->clear();
						parse_field(&headers->emplace_back());
						while (accept(separator)) {
								parse_field(&headers->emplace_back());
						}
				} else {
						parse_field(nullptr);
						while (accept(separator)) {
								parse_field(nullptr);
						}
				}
		}

		/// @brief parse some continuous columns in a record
		/// @param output pointer to an array of String
		/// @param n size of the output array
		/// @param col the first column to store into output, counting from 0
		/// @returns actual # all columns parsed
		size_t parse_record(std::string* output = nullptr,
							const size_t n = 1,
							const size_t col = 0) {
				size_t curr_col = 0;
				do {
						if (output && curr_col >= col && curr_col < col + n) {
								parse_field(output++);
						} else {
								parse_field();
						}
						++curr_col;
				} while (accept(separator));
				if (curr_col < col + n - 1)
						THROW("not enough column;"
							  " tried to parse [%lu, %lu),"
							  " but got only %lu columns",
							  col, col + n, curr_col);
				return curr_col;
		}
		void parse_field(std::string* field = nullptr) {
				if (accept(DQUOTE)) {
						while (true) {
								if (accept(EOF)) THROW("unexpected EOF");
								if (accept(DQUOTE)) {
										if (accept(DQUOTE)) {
												if (field)
														field->push_back('"');
										} else {
												break;
										}
								} else {
										Symbol c = m_input->get();
										if (field) field->push_back(c);
								}
						}
				} else {
						while (true) {
								Symbol c = m_input->get();
								if (c == separator || c == CR || c == LF
									|| c == EOF) {
										m_input->putback(c);
										break;
								}
								if (field) field->push_back(c);
						}
				}
		}

		static std::vector<Encoding> get_encoding(const std::string& s);
		static std::vector<size_t> get_grouping(const std::string& s);
		void get_info();
};

NAMESPACE_END
