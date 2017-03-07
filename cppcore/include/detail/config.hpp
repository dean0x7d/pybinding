#pragma once
#include <cstddef>

#ifdef CPB_USE_MKL
#define EIGEN_USE_MKL_ALL
#define CPB_USE_FEAST
#endif

#define EIGEN_DONT_PARALLELIZE // disable Eigen's internal multi-threading (doesn't do much anyway)
#define EIGEN_MAX_ALIGN_BYTES 32 // always use AVX alignment
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::ptrdiff_t
#define EIGEN_DEFAULT_TO_ROW_MAJOR

namespace cpb {
    using idx_t = std::ptrdiff_t; // type for general indexing and interfaces
    using storage_idx_t = int; // type used when storing indices in containers
}
