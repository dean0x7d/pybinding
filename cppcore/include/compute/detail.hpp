#pragma once
#include "detail/macros.hpp"

#include <complex>

namespace cpb { namespace compute { namespace detail {

/**
 These functions are needed because std::complex<T> operator* does additional
 checking which significantly slows down critical loops. This `mul` overload
 does a raw multiplication where the user must make sure there are no numerical
 complications.
 */
template<class real_t> CPB_ALWAYS_INLINE
real_t mul(real_t a, real_t b) { return a * b; }

template<class real_t> CPB_ALWAYS_INLINE
std::complex<real_t> mul(std::complex<real_t> a, std::complex<real_t> b) {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}

/**
 Return only the real part of a dot product multiplication. Compared to
 `mul(conj(a), b)`, this saves a few operations for complex scalars.
 */
template<class real_t> CPB_ALWAYS_INLINE
real_t real_dot(real_t a, real_t b) { return a * b; }

template<class real_t> CPB_ALWAYS_INLINE
real_t real_dot(std::complex<real_t> a, std::complex<real_t> b) {
    return a.real() * b.real() + a.imag() * b.imag();
}

template<class real_t> CPB_ALWAYS_INLINE
real_t square(real_t a) { return a * a; }

template<class real_t> CPB_ALWAYS_INLINE
real_t square(std::complex<real_t> a) {
    return a.real() * a.real() + a.imag() * a.imag();
}

}}} // namespace cpb::compute::detail
