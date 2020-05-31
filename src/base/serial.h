#pragma once

#include "algebra.h"

#include <fstream>

NAMESPACE_BEGIN

template <typename T, typename = std::enable_if<std::is_arithmetic_v<T>>>
void serialize_to(std::fstream& out, T v) {
		out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T, typename = std::enable_if<std::is_arithmetic_v<T>>>
void serialize_to(std::fstream& out, const std::vector<T>& v) {
		serialize_to(out, v.size());
		out.write(reinterpret_cast<const char*>(v.data()),
				  v.size() * sizeof(T));
}

template <typename T, typename = std::enable_if<std::is_arithmetic_v<T>>>
void deserialize_from(std::fstream& in, T& v) {
		in.read(reinterpret_cast<char*>(&v), sizeof(T));
}
template <typename T, typename = std::enable_if<std::is_arithmetic_v<T>>>
void deserialize_from(std::fstream& in, std::vector<T>& v) {
		size_t size;
		deserialize_from(in, size);
		v.resize(size);
		in.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
}

template <typename T,
		  typename = std::enable_if<std::is_base_of_v<Eigen::EigenBase<T>, T>>>
void serialize_matrix(std::fstream& out, const T& m) {
		index_t rows = m.rows(), cols = m.cols();
		serialize_to(out, rows);
		serialize_to(out, cols);
		for (index_t r = 0; r < rows; ++r) {
				for (index_t c = 0; c < cols; ++c) {
						serialize_to(out, m.coeff(r, c));
				}
		}
}

template <int R, int C>
void deserialize_matrix(std::fstream& in, Matrix<R, C>& m) {
		index_t rows, cols;
		deserialize_from(in, rows);
		deserialize_from(in, cols);
		index_t size = rows * cols;
		m.resize(rows, cols);
		for (index_t i = 0; i < size; ++i) {
				deserialize_from(in, m.coeffRef(i));
		}
}

NAMESPACE_END
