#pragma once
#include <complex>

namespace cpb { namespace constant {
    // imaginary one
    constexpr std::complex<float> i1(0, 1);
    // the omnipresent pi
    constexpr float pi = 3.14159265358979323846f;
    // electron charge [C]
    constexpr float e = 1.602e-19f;
    // reduced Planck constant [eV*s]
    constexpr float hbar = 6.58211899e-16f;
    // electron rest mass [kg]
    constexpr float m0 = 9.10938188e-31f;
    // vacuum permittivity [F/m == C/V/m]
    constexpr float epsilon0 = 8.854e-12f;
    // magnetic flux quantum (h/e)
    constexpr float phi0 = 2 * pi*hbar;
    // Boltzmann constant
    constexpr float kb = 8.6173303e-5f;
}} // namespace cpb::constant
