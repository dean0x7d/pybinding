#pragma once
#include "numeric/traits.hpp"
#include "cuda/thrust.hpp"
#include <cuComplex.h>

namespace cpb { namespace num {

namespace detail {
    template<class T>
    struct complex_traits<thr::complex<T>> {
        using real_t = T;
        using complex_t = thr::complex<T>;
        static constexpr bool is_complex = true;
    };

    template<class T> struct thrust_scalar { using type = T; };
    template<class T> struct thrust_scalar<std::complex<T>> { using type = thr::complex<T>; };

    template<class T> struct host_scalar { using type = T; };
    template<class T> struct host_scalar<thr::complex<T>> { using type = std::complex<T>; };

    template<class T> struct cuda_scalar { using type = T; };
    template<> struct cuda_scalar<std::complex<float>> { using type = cuFloatComplex; };
    template<> struct cuda_scalar<std::complex<double>> { using type = cuDoubleComplex; };
    template<class T> struct cuda_scalar<thr::complex<T>> : cuda_scalar<std::complex<T>> {};
} // namespace detail

/**
 Return the thrust scalar type corresponding to the given host scalar

 For example:
   float               -> float
   std::complex<float> -> thrust::complex<float>
 */
template<class scalar_t>
using get_thrust_t = typename detail::thrust_scalar<scalar_t>::type;

/**
 Return the host scalar type corresponding to the given thrust scalar

 For example:
   float                  -> float
   thrust::complex<float> -> std::complex<float>
 */
template<class thrust_scalar_t>
using get_host_t = typename detail::host_scalar<thrust_scalar_t>::type;

/**
 Return the Cuda scalar type corresponding to the given host or thrust scalar

 For example:
   float                  -> float
   std::complex<float>    -> cuFloatComplex
   thrust::complex<float> -> cuFloatComplex
 */
template<class scalar_t>
using get_cuda_t = typename detail::cuda_scalar<scalar_t>::type;

}} // namespace cpb::num
