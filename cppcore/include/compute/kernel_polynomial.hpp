#pragma once

#ifdef TBM_USE_MKL
# include "mkl/kernel_polynomial.hpp"
#else
# include "eigen3/kernel_polynomial.hpp"
#endif
