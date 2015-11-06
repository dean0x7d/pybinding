#pragma once

#ifdef TBM_USE_MKL
#define EIGEN_USE_MKL_ALL
#define TBM_USE_FEAST
#endif

// disable Eigen's internal multi-threading (doesn't do much anyway)
#define EIGEN_DONT_PARALLELIZE

