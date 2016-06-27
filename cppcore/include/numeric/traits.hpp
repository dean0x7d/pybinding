#pragma once
#include <complex>
#include <string>
#include <type_traits>
#include <limits>

namespace cpb { namespace num {

namespace detail {
    template<class T>
    struct complex_traits {
        static_assert(std::is_arithmetic<T>::value, "");

        using real_t = T;
        using complex_t = std::complex<T>;
        static constexpr bool is_complex = false;
    };

    template<class T>
    struct complex_traits<std::complex<T>> {
        using real_t = T;
        using complex_t = std::complex<T>;
        static constexpr bool is_complex = true;
    };
} // namespace detail

/**
 Return the real type corresponding to the given scalar type

 For example:
   std::complex<float> -> float
   float               -> float
 */
template<class scalar_t>
using get_real_t = typename detail::complex_traits<scalar_t>::real_t;

/**
 Return the complex type corresponding to the given scalar type

 For example:
   std::complex<float> -> std::complex<float>
   float               -> std::complex<float>
 */
template<class scalar_t>
using get_complex_t = typename detail::complex_traits<scalar_t>::complex_t;

/**
 Is the given scalar type complex?
 */
template<class scalar_t>
inline constexpr bool is_complex() { return detail::complex_traits<scalar_t>::is_complex; }

/**
 Return the complex conjugate in the same scalar type as the input

 The standard `conj` function always returns `std::complex<T>`
 which is not convenient for generic algorithms.
 */
template<class scalar_t>
inline scalar_t conjugate(scalar_t value) { return value; }
template<class scalar_t>
inline std::complex<scalar_t> conjugate(std::complex<scalar_t> value) { return std::conj(value); }

/**
 Cast a `std::complex<double>` to another scalar

 The conversion loses precision and/or the imaginary part, as intended.
 */
template<class scalar_t>
inline scalar_t complex_cast(std::complex<double> v) { return v; }
template<>
inline std::complex<float> complex_cast<std::complex<float>>(std::complex<double> v) {
    return {static_cast<float>(v.real()), static_cast<float>(v.imag())};
}
template<>
inline double complex_cast<double>(std::complex<double> v) { return v.real(); }
template<>
inline float complex_cast<float>(std::complex<double> v) { return static_cast<float>(v.real()); }

/**
 Return a human readable name of the scalar type
 */
template<class scalar_t> inline std::string scalar_name();
template<> inline std::string scalar_name<float>() { return "float"; }
template<> inline std::string scalar_name<double>() { return "double"; }
template<> inline std::string scalar_name<std::complex<float>>() { return "complex<float>"; }
template<> inline std::string scalar_name<std::complex<double>>() { return "complex<double>"; }

/**
 Floating-point equality with precision in ULP (units in the last place)
 */
template<class T, class = typename std::enable_if<std::is_floating_point<T>::value, void>::type>
bool approx_equal(T x, T y, int ulp = 1) {
    auto const diff = std::abs(x - y);
    auto const scale = std::abs(x + y);
    return diff <= std::numeric_limits<T>::epsilon() * scale * ulp
           || diff <= std::numeric_limits<T>::min(); // subnormal case
}

/**
 Return the minimum aligned size for the given scalar type and alignment in bytes

 For example:
    aligned_size<float, 16>(3) -> 4
    aligned_size<std::complex<double>, 16>(2) -> 2
 */
template<class scalar_t, int align_bytes, class T>
T aligned_size(T size) {
    static constexpr auto step = static_cast<T>(align_bytes / sizeof(scalar_t));
    while (size % step != 0) {
        ++size;
    }
    return size;
};

}} // namespace cpb::num
