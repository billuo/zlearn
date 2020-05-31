#pragma once

#define EIGEN_NO_AUTOMATIC_RESIZING 1
#define EIGEN_RUNTIME_NO_MALLOC 1

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "config.h"

NAMESPACE_BEGIN

using Eigen::Dynamic;

// typically the 1st column is always the model parameter itself;
// the other optional columns are cached values used only during optimization.
template <int Rows, int Columns>
using Matrix =
	Eigen::Matrix<real_t, Rows, Columns, Eigen::AutoAlign | Eigen::RowMajor>;
template <int Size>
using Vector =
	Eigen::Matrix<real_t, 1, Size, Eigen::AutoAlign | Eigen::RowMajor>;

using index_t = Eigen::Index;

inline void disable_malloc() { Eigen::internal::set_is_malloc_allowed(false); }
inline void enable_malloc() { Eigen::internal::set_is_malloc_allowed(true); }
inline bool is_malloc_enabled() { return Eigen::internal::is_malloc_allowed(); }

template <typename T,
		  typename = std::enable_if<std::is_base_of_v<Eigen::EigenBase<T>, T>>>
inline std::string to_string(const T& m) {
		std::stringstream ss;
		index_t rows = m.rows(), cols = m.cols();
		for (index_t r = 0; r < rows; ++r) {
				for (index_t c = 0; c < cols; ++c) {
						ss << m.coeff(r, c);
						if (c != cols - 1) ss << ' ';
				}
				if (r != rows - 1) ss << std::endl;
		}
		return ss.str().c_str();
}

NAMESPACE_END
