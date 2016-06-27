#pragma once

#ifdef CPB_USE_MKL
#define EIGEN_USE_MKL_ALL
#define CPB_USE_FEAST
#endif

// disable Eigen's internal multi-threading (doesn't do much anyway)
#define EIGEN_DONT_PARALLELIZE
