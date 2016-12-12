#pragma once

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wall"
# pragma clang diagnostic ignored "-Wunused-parameter"
#endif

#include <thrust/host_vector.h>
#include <thrust/complex.h>

#ifdef __clang__
# pragma clang diagnostic pop
#endif

namespace thr = thrust;
