#include "kpm/Strategy.hpp"

#include "kpm/calc_moments.hpp"
#ifdef CPB_USE_CUDA
# include "cuda/kpm/calc_moments.hpp"
#endif

namespace cpb { namespace kpm {

namespace {
    template<class scalar_t>
    Bounds<scalar_t> reset_bounds(SparseMatrixX<scalar_t> const* hamiltonian,
                                  Config const& config) {
        if (config.min_energy == config.max_energy) {
            return {hamiltonian, config.lanczos_precision}; // will be automatically computed
        } else {
            return {config.min_energy, config.max_energy}; // user-defined bounds
        }
    }
} // anonymous namespace

template<class scalar_t, class Impl>
StrategyTemplate<scalar_t, Impl>::StrategyTemplate(SparseMatrixRC<scalar_t> h,
                                                   Config const& config)
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
bool StrategyTemplate<scalar_t, Impl>::change_hamiltonian(Hamiltonian const& h) {
    if (!ham::is<scalar_t>(h)) {
        return false;
    }

    hamiltonian = ham::get_shared_ptr<scalar_t>(h);
    optimized_hamiltonian = {hamiltonian.get(), Impl::matrix_config(config.opt_level)};
    bounds = reset_bounds(hamiltonian.get(), config);

    return true;
}

template<class scalar_t, class Impl>
ArrayXcd StrategyTemplate<scalar_t, Impl>::greens(int row, int col, ArrayXd const& energy,
                                                  double broadening) {
    return std::move(greens_vector(row, {col}, energy, broadening).front());
}

template<class scalar_t, class Impl>
std::vector<ArrayXcd>
StrategyTemplate<scalar_t, Impl>::greens_vector(int row, std::vector<int> const& cols,
                                                ArrayXd const& energy, double broadening) {
    assert(!cols.empty());
    auto const scale = bounds.scaling_factors();
    auto const scaled_energy = bounds.scale_energy(energy.template cast<real_t>());
    auto const num_moments = required_num_moments(scale, config.lambda, broadening);
    optimized_hamiltonian.optimize_for({row, cols}, scale);
    auto const& idx = optimized_hamiltonian.idx();
    stats = {num_moments};

    if (idx.is_diagonal()) {
        auto moments = ExvalDiagonalMoments<scalar_t>(num_moments, idx.row);

        stats.moments_timer.tic();
        Impl::diagonal(moments, optimized_hamiltonian, config.opt_level);
        stats.moments_timer.toc();

        detail::apply_lorentz_kernel(moments.get(), config.lambda);
        auto const greens = detail::calculate_greens(scaled_energy, moments.get());
        return {greens.template cast<std::complex<double>>()};
    } else {
        auto moments_vector = ExvalOffDiagonalMoments<scalar_t>(num_moments, idx);

        stats.moments_timer.tic();
        Impl::off_diagonal(moments_vector, optimized_hamiltonian, config.opt_level);
        stats.moments_timer.toc();

        for (auto& moments : moments_vector.get()) {
            detail::apply_lorentz_kernel(moments, config.lambda);
        }

        auto greens = std::vector<ArrayXcd>();
        greens.reserve(idx.cols.size());
        for (auto const& moments : moments_vector.get()) {
            auto const g = detail::calculate_greens(scaled_energy, moments);
            greens.push_back(g.template cast<std::complex<double>>());
        }
        return greens;
    }
}

template<class scalar_t, class Impl>
std::string StrategyTemplate<scalar_t, Impl>::report(bool shortform) const {
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

    template<class Moments, class scalar_t>
    static void diagonal(Moments& moments, OptimizedHamiltonian<scalar_t> const& oh,
                         int opt_level) {
        assert(oh.idx().is_diagonal());
        using namespace calc_moments::diagonal;

        switch (opt_level) {
            case 0: basic(moments, oh.csr()); break;
            case 1: opt_size(moments, oh.csr(), oh.sizes()); break;
            case 2: opt_size_and_interleaved(moments, oh.csr(), oh.sizes()); break;
            default: opt_size_and_interleaved(moments, oh.ell(), oh.sizes()); break;
        }
    }

    template<class Moments, class scalar_t>
    static void off_diagonal(Moments& moments, OptimizedHamiltonian<scalar_t> const& oh,
                             int opt_level) {
        using namespace calc_moments::off_diagonal;

        switch (opt_level) {
            case 0: basic(moments, oh.csr()); break;
            case 1: opt_size(moments, oh.csr(), oh.sizes()); break;
            case 2: opt_size_and_interleaved(moments, oh.csr(), oh.sizes()); break;
            default: opt_size_and_interleaved(moments, oh.ell(), oh.sizes()); break;
        }
    }
};

CPB_INSTANTIATE_TEMPLATE_CLASS_VARGS(StrategyTemplate, DefaultCalcMoments)

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

CPB_INSTANTIATE_TEMPLATE_CLASS_VARGS(StrategyTemplate, CudaCalcMoments)
#endif // CPB_USE_CUDA

}} // namespace cpb::kpm
