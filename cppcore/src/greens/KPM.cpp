#include "greens/KPM.hpp"
#include "greens/kpm/calc_moments.hpp"
#ifdef CPB_USE_CUDA
# include "greens/kpm/calc_moments_cuda.hpp"
#endif

namespace cpb { namespace kpm {

namespace {
    template<class scalar_t>
    Bounds<scalar_t> reset_bounds(SparseMatrixX<scalar_t> const* hamiltonian,
                                  KPMConfig const& config) {
        if (config.min_energy == config.max_energy) {
            return {hamiltonian, config.lanczos_precision}; // will be automatically computed
        } else {
            return {config.min_energy, config.max_energy}; // user-defined bounds
        }
    }
}

template<class scalar_t, class Impl>
Strategy<scalar_t, Impl>::Strategy(SparseMatrixRC<scalar_t> h, Config const& config)
    : hamiltonian(std::move(h)), config(config), bounds(reset_bounds(hamiltonian.get(), config)),
      optimized_hamiltonian(hamiltonian.get(), Impl::matrix_config(config.opt_level)) {
    if (config.min_energy > config.max_energy) {
        throw std::invalid_argument("KPM: Invalid energy range specified (min > max).");
    }
    if (config.lambda <= 0) {
        throw std::invalid_argument("KPM: Lambda must be positive.");
    }
}

template<class scalar_t, class Impl>
bool Strategy<scalar_t, Impl>::change_hamiltonian(Hamiltonian const& h) {
    if (!ham::is<scalar_t>(h)) {
        return false;
    }

    hamiltonian = ham::get_shared_ptr<scalar_t>(h);
    optimized_hamiltonian = {hamiltonian.get(), Impl::matrix_config(config.opt_level)};
    bounds = reset_bounds(hamiltonian.get(), config);

    return true;
}

template<class scalar_t, class Impl>
ArrayXcd Strategy<scalar_t, Impl>::calc(int row, int col, ArrayXd const& energy,
                                        double broadening) {
    return std::move(calc_vector(row, {col}, energy, broadening).front());
}

template<class scalar_t, class Impl>
std::vector<ArrayXcd> Strategy<scalar_t, Impl>::calc_vector(int row, std::vector<int> const& cols,
                                                            ArrayXd const& energy,
                                                            double broadening) {
    assert(!cols.empty());
    auto const scale = bounds.scaling_factors();
    auto const scaled_energy = bounds.scale_energy(energy.template cast<real_t>());
    auto const num_moments = required_num_moments(scale, config.lambda, broadening);
    optimized_hamiltonian.optimize_for({row, cols}, scale);
    stats = {num_moments};

    if (optimized_hamiltonian.idx().is_diagonal()) {
        stats.moments_timer.tic();
        auto moments = Impl::moments_diag(optimized_hamiltonian, num_moments, config.opt_level);
        stats.moments_timer.toc();

        detail::apply_lorentz_kernel(moments, config.lambda);
        auto const greens = detail::calculate_greens(scaled_energy, moments);
        return {greens.template cast<std::complex<double>>()};
    } else {
        stats.moments_timer.tic();
        auto mvector = Impl::moments_vector(optimized_hamiltonian, num_moments, config.opt_level);
        stats.moments_timer.toc();

        mvector.apply_lorentz_kernel(config.lambda);
        return mvector.calc_greens(scaled_energy);
    }
}

template<class scalar_t, class Impl>
std::string Strategy<scalar_t, Impl>::report(bool shortform) const {
    return bounds.report(shortform)
           + optimized_hamiltonian.report(stats.last_num_moments, shortform)
           + stats.report(optimized_hamiltonian.operations(stats.last_num_moments), shortform)
           + (shortform ? "|" : "Total time:");
}

struct DefaultCalcMoments {
    static MatrixConfig matrix_config(int opt_level) {
        switch (opt_level) {
            case 0: return {MatrixConfig::Reorder::OFF, MatrixConfig::Format::CSR};
            case 1: return {MatrixConfig::Reorder::ON, MatrixConfig::Format::CSR};
            case 2: return {MatrixConfig::Reorder::ON, MatrixConfig::Format::CSR};
            default: return {MatrixConfig::Reorder::ON, MatrixConfig::Format::ELL};
        }
    };

    template<class scalar_t>
    static MomentsMatrix<scalar_t> moments_vector(OptimizedHamiltonian<scalar_t> const& oh,
                                                  int num_moments, int opt_level) {
        switch (opt_level) {
            case 0: return calc_moments0(oh.csr(), oh.idx(), num_moments);
            case 1: return calc_moments1(oh.csr(), oh.idx(), num_moments, oh.sizes());
            case 2: return calc_moments2(oh.csr(), oh.idx(), num_moments, oh.sizes());
            default: return calc_moments2(oh.ell(), oh.idx(), num_moments, oh.sizes());
        }
    }

    template<class scalar_t>
    static ArrayX<scalar_t> moments_diag(OptimizedHamiltonian<scalar_t> const& oh,
                                         int num_moments, int opt_level) {
        assert(oh.idx().is_diagonal());
        auto const i = oh.idx().row;
        switch (opt_level) {
            case 0: return calc_diag_moments0(oh.csr(), i, num_moments);
            case 1: return calc_diag_moments1(oh.csr(), i, num_moments, oh.sizes());
            case 2: return calc_diag_moments2(oh.csr(), i, num_moments, oh.sizes());
            default: return calc_diag_moments2(oh.ell(), i, num_moments, oh.sizes());
        }
    }
};

CPB_INSTANTIATE_TEMPLATE_CLASS_VARGS(Strategy, DefaultCalcMoments)

#ifdef CPB_USE_CUDA
struct CudaCalcMoments {
    static MatrixConfig matrix_config(int opt_level) {
        switch (opt_level) {
            case 0: return {MatrixConfig::Reorder::OFF, MatrixConfig::Format::ELL};
            default: return {MatrixConfig::Reorder::ON, MatrixConfig::Format::ELL};
        }
    };

    template<class scalar_t>
    static MomentsMatrix<scalar_t> moments_vector(OptimizedHamiltonian<scalar_t> const& oh,
                                                  int num_moments, int opt_level) {
        switch (opt_level) {
            default: return calc_moments2(oh.ell(), oh.idx(), num_moments, oh.sizes());
        }
    }

    template<class scalar_t>
    static ArrayX<scalar_t> moments_diag(OptimizedHamiltonian<scalar_t> const& oh,
                                         int num_moments, int opt_level) {
        assert(oh.idx().is_diagonal());
        using Cuda = cuda::I<scalar_t>;
        auto const i = oh.idx().row;
        auto const ell = ellref(oh.ell());

        auto moments = [&]{
            switch (opt_level) {
                case 0: return Cuda::calc_diag_moments0(ell, i, num_moments);
                default: return Cuda::calc_diag_moments1(ell, i, num_moments, oh.sizes());
            }
        }();
        return eigen_cast<ArrayX>(moments);
    }
};

CPB_INSTANTIATE_TEMPLATE_CLASS_VARGS(Strategy, CudaCalcMoments)
#endif // CPB_USE_CUDA

}} // namespace cpb::kpm
