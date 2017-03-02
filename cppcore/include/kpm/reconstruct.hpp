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

/// Return the velocity operator for the direction given by the `alpha` position vector
template<class scalar_t>
SparseMatrixX<scalar_t> velocity(SparseMatrixX<scalar_t> const& ham, ArrayXf const& alpha) {
    auto result = ham;
    auto const data = result.valuePtr();
    auto const indices = result.innerIndexPtr();
    auto const indptr = result.outerIndexPtr();

    auto const size = result.rows();
    for (auto row = idx_t{0}; row < size; ++row) {
        for (auto n = indptr[row]; n < indptr[row + 1]; ++n) {
            const auto col = indices[n];
            data[n] *= static_cast<scalar_t>(alpha[row] - alpha[col]);
        }
    }

    return result.markAsRValue();
}

/// Reconstruct the Kubo-Bastin formula for the conductivity:
///     sigma(mu, T) = 4 / a^2 * int_-1^1 fd(E) / (1 - E^2)^2 sum(momenta * gamma(E)) dE
/// The resulting conductivity is in units of `e^2 / h * Omega` where Omega is the volume.
template<class scalar_t, class real_t, class complex_t = num::get_complex_t<scalar_t>>
ArrayXcd reconstruct_kubo_bastin(MatrixX<scalar_t> const& moments, ArrayXd const& chemical_pot,
                                 ArrayX<real_t> const& energy_samples, double temperature,
                                 Scale<real_t> scale) {
    auto const inv_kbt_sc = static_cast<real_t>(scale.a / (constant::kb * temperature));
    auto const num_moments = moments.rows();
    auto const scaled_chemical_potential = scale(chemical_pot.cast<real_t>());
    auto const scaled_energy_samples = scale(energy_samples);

    auto gamma = [&](real_t en) {
        constexpr auto i1 = static_cast<complex_t>(constant::i1);

        using Row = Eigen::Array<real_t, 1, Eigen::Dynamic>;
        auto const ns_row = Row::LinSpaced(num_moments, 0, static_cast<real_t>(num_moments - 1));
        auto const ns_2d = ns_row.replicate(num_moments, 1).eval();

        auto const sqrt_n = en - i1 * ns_2d * sqrt(real_t{1} - en * en);
        auto const exp_n = exp(i1 * acos(en) * ns_2d);
        auto const t_m = cos(acos(en) * ns_2d.transpose());

        auto const g_p = MatrixX<complex_t>(sqrt_n * exp_n * t_m);
        return (g_p + g_p.adjoint()).array().eval();
    };

    auto const coeff = (scaled_energy_samples.maxCoeff() - scaled_energy_samples.minCoeff())
                       / static_cast<real_t>(2 * scaled_energy_samples.size());

    auto integrate = [&](ArrayX<complex_t> const& func) {
        return coeff * (real_t{2} * func.sum() - func(0) - func(func.size() - 1));
    };

    auto fermi_dirac = [&](real_t mi) {
        return real_t{1} / (real_t{1} + exp((scaled_energy_samples - mi) * inv_kbt_sc));
    };

    auto sum_nm = transform<ArrayX>(scaled_energy_samples, [&](real_t en) {
        auto const k = real_t{1} / ((real_t{1} - en * en) * (real_t{1} - en * en));
        return k * sum(moments.array() * gamma(en));
    });

    auto const prefix = scalar_t{4} / (scale.a * scale.a);
    return transform<ArrayX>(scaled_chemical_potential, [&](real_t mu) {
        return prefix * integrate(fermi_dirac(mu) * sum_nm);
    }).template cast<std::complex<double>>();
}

}} // namespace cpb::kpm
