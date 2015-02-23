#pragma once
#include <complex>
#include <string>

namespace num {

namespace detail {
    template<class scalar_t>
    struct complex_traits {
        using real_t = scalar_t;
        using complex_t = std::complex<scalar_t>;
        static constexpr bool is_complex = false;
    };

    template<class scalar_t>
    struct complex_traits<std::complex<scalar_t>> {
        using real_t = scalar_t;
        using complex_t = std::complex<scalar_t>;
        static constexpr bool is_complex = true;
    };
} // namespace detail

template<class scalar_t>
using get_real_t = typename detail::complex_traits<scalar_t>::real_t;
template<class scalar_t>
using get_complex_t = typename detail::complex_traits<scalar_t>::complex_t;

template<class scalar_t>
inline constexpr bool is_complex() { return detail::complex_traits<scalar_t>::is_complex; }

template<class scalar_t>
inline scalar_t conjugate(scalar_t value) { return value; }
template<class scalar_t>
inline std::complex<scalar_t> conjugate(std::complex<scalar_t> value) { return std::conj(value); }

template<class scalar_t>
inline std::string scalar_name() { return ""; }
template<> inline std::string scalar_name<float>() { return "float"; }
template<> inline std::string scalar_name<double>() { return "double"; }
template<> inline std::string scalar_name<std::complex<float>>() { return "complex<float>"; }
template<> inline std::string scalar_name<std::complex<double>>() { return "complex<double>"; }

} // namespace num
