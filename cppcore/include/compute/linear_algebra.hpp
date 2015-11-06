#pragma once

#ifdef TBM_USE_MKL
# include "mkl/linear_algebra.hpp"
#else
# include "eigen3/linear_algebra.hpp"
#endif
