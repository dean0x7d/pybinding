#pragma once
#include "kpm/Bounds.hpp"

namespace cpb { namespace kpm {

/// Reconstruct a real function based on the given KPM moments
///    f(E) = 2 / (a * pi * sqrt(1 - E^2)) * sum_n( moments * cos(n * acos(E)) )
template<class real_t>
ArrayXd reconstruct(ArrayX<real_t> const& moments, ArrayXd const& energy, Scale<real_t> scale) {
    static_assert(!num::is_complex<real_t>(), "");

    auto const scaled_energy = scale(energy.cast<real_t>());
    auto const ns = make_integer_range<real_t>(moments.size());
    auto const k = real_t{2 / constant::pi} / scale.a;

    return transform<ArrayX>(scaled_energy, [&](real_t E) {
        return k / sqrt(1 - E*E) * sum(moments * cos(ns * acos(E)));
    }).template cast<double>();
}

/// Reconstruct Green's function based on the given KPM moments
///     g(E) = -2*i / (a * sqrt(1 - E^2)) * sum_n( moments * exp(-i*n*acos(E)) )
template<class scalar_t, class real_t, class complex_t = num::get_complex_t<scalar_t>>
ArrayXcd reconstruct_greens(ArrayX<scalar_t> const& moments, ArrayXd const& energy,
                            Scale<real_t> scale) {
    constexpr auto i1 = complex_t{constant::i1};

    auto const scaled_energy = scale(energy.cast<real_t>());
    auto const ns = make_integer_range<real_t>(moments.size());
    auto const k = -real_t{2} * i1 / scale.a;

    return transform<ArrayX>(scaled_energy.eval(), [&](real_t E) {
        return k / sqrt(1 - E*E) * sum(moments * exp(-i1 * ns * acos(E)));
    }).template cast<std::complex<double>>();
}

}} // namespace cpb::kpm
