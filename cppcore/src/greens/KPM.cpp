#include "greens/KPM.hpp"
#include "greens/kpm/calc_moments.hpp"

namespace tbm {

namespace {
    template<class scalar_t>
    kpm::Bounds<scalar_t> reset_bounds(SparseMatrixX<scalar_t> const* hamiltonian,
                                       KPMConfig const& config) {
        if (config.min_energy == config.max_energy) {
            return {hamiltonian, config.lanczos_precision}; // will be automatically computed
        } else {
            return {config.min_energy, config.max_energy}; // user-defined bounds
        }
    }
}

template<class scalar_t>
KPM<scalar_t>::KPM(SparseMatrixRC<scalar_t> h, Config const& config)
    : hamiltonian(std::move(h)), config(config), bounds(reset_bounds(hamiltonian.get(), config)),
      optimized_hamiltonian(hamiltonian.get(), config.optimization_level) {
    if (config.min_energy > config.max_energy) {
        throw std::invalid_argument("KPM: Invalid energy range specified (min > max).");
    }
    if (config.lambda <= 0) {
        throw std::invalid_argument("KPM: Lambda must be positive.");
    }
}

template<class scalar_t>
bool KPM<scalar_t>::change_hamiltonian(Hamiltonian const& h) {
    if (!ham::is<scalar_t>(h)) {
        return false;
    }

    hamiltonian = ham::get_shared_ptr<scalar_t>(h);
    optimized_hamiltonian = {hamiltonian.get(), config.optimization_level};
    bounds = reset_bounds(hamiltonian.get(), config);

    return true;
}

template<class scalar_t>
ArrayXcd KPM<scalar_t>::calc(int row, int col, ArrayXd const& energy, double broadening) {
    if (row == col) {
        auto const moments = calc_moments_diag(row, broadening);
        auto const scaled_energy = bounds.scale_energy(energy.template cast<real_t>());
        auto const greens = kpm::detail::calculate_greens(scaled_energy, moments);
        return greens.template cast<std::complex<double>>();
    } else {
        return std::move(calc_vector(row, {col}, energy, broadening).front());
    }
}

template<class scalar_t>
std::vector<ArrayXcd> KPM<scalar_t>::calc_vector(int row, std::vector<int> const& cols,
                                                 ArrayXd const& energy, double broadening) {
    assert(!cols.empty());
    auto const moment_matrix = calc_moments_matrix({row, cols}, broadening);
    auto const scaled_energy = bounds.scale_energy(energy.template cast<real_t>());
    return moment_matrix.calc_greens(scaled_energy);
}

template<class scalar_t>
std::string KPM<scalar_t>::report(bool shortform) const {
    return bounds.report(shortform)
           + optimized_hamiltonian.report(stats.last_num_moments, shortform)
           + stats.report(optimized_hamiltonian.operations(stats.last_num_moments), shortform)
           + (shortform ? "|" : "Total time:");
}

template<class scalar_t>
kpm::MomentsMatrix<scalar_t> KPM<scalar_t>::calc_moments_matrix(kpm::Indices const& idx,
                                                                double broadening) {
    auto const scale = bounds.scaling_factors();
    optimized_hamiltonian.optimize_for(idx, scale);

    auto const num_moments = kpm::required_num_moments(scale, config.lambda, broadening);
    stats = {num_moments};
    stats.moments_timer.tic();
    auto moment_matrix = [&] {
        auto const& oh = optimized_hamiltonian;
        switch (config.optimization_level) {
            case 0: return kpm::calc_moments0(oh.csr(), idx, num_moments);
            case 1: return kpm::calc_moments1(oh.csr(), oh.idx(), num_moments, oh.sizes());
            case 2: return kpm::calc_moments2(oh.csr(), oh.idx(), num_moments, oh.sizes());
            default: return kpm::calc_moments2(oh.ell(), oh.idx(), num_moments, oh.sizes());
        }
    }();
    moment_matrix.apply_lorentz_kernel(config.lambda);
    stats.moments_timer.toc();

    return moment_matrix;
}

template<class scalar_t>
ArrayX<scalar_t> KPM<scalar_t>::calc_moments_diag(int i, double broadening) {
    auto const scale = bounds.scaling_factors();
    optimized_hamiltonian.optimize_for({i, i}, scale);

    auto const num_moments = kpm::required_num_moments(scale, config.lambda, broadening);
    stats = {num_moments};
    stats.moments_timer.tic();
    auto moments = [&] {
        auto const& oh = optimized_hamiltonian;
        auto const opt_i = oh.idx().row;
        switch (config.optimization_level) {
            case 0: return kpm::calc_diag_moments0(oh.csr(), i, num_moments);
            case 1: return kpm::calc_diag_moments1(oh.csr(), opt_i, num_moments, oh.sizes());
            case 2: return kpm::calc_diag_moments2(oh.csr(), opt_i, num_moments, oh.sizes());
            default: return kpm::calc_diag_moments2(oh.ell(), opt_i, num_moments, oh.sizes());
        }
    }();
    kpm::detail::apply_lorentz_kernel(moments, config.lambda);
    stats.moments_timer.toc();

    return moments;
}

TBM_INSTANTIATE_TEMPLATE_CLASS(KPM)

} // namespace tbm
