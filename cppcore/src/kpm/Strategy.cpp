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

template<class scalar_t, class Compute>
StrategyTemplate<scalar_t, Compute>::StrategyTemplate(SparseMatrixRC<scalar_t> h,
                                                      Config const& config)
    : hamiltonian(std::move(h)), config(config), bounds(reset_bounds(hamiltonian.get(), config)),
      optimized_hamiltonian(hamiltonian.get(), config.matrix_format, config.algorithm.reorder()) {
    if (config.min_energy > config.max_energy) {
        throw std::invalid_argument("KPM: Invalid energy range specified (min > max).");
    }
}

template<class scalar_t, class Compute>
bool StrategyTemplate<scalar_t, Compute>::change_hamiltonian(Hamiltonian const& h) {
    if (!ham::is<scalar_t>(h)) {
        return false;
    }

    hamiltonian = ham::get_shared_ptr<scalar_t>(h);
    optimized_hamiltonian = {hamiltonian.get(), config.matrix_format, config.algorithm.reorder()};
    bounds = reset_bounds(hamiltonian.get(), config);

    return true;
}

template<class scalar_t, class Compute>
ArrayXd StrategyTemplate<scalar_t, Compute>::ldos(int index, ArrayXd const& energy,
                                                  double broadening) {
    auto const scale = bounds.scaling_factors();
    auto const scaled_energy = bounds.scaled(energy.template cast<real_t>());
    auto const num_moments = config.kernel.required_num_moments(broadening / scale.a);

    optimized_hamiltonian.optimize_for({index, index}, scale);
    stats = {num_moments, optimized_hamiltonian.operations(num_moments),
             optimized_hamiltonian.memory_usage(), hamiltonian->rows() * sizeof(scalar_t)};

    auto moments = ExvalDiagonalMoments<scalar_t>(num_moments, optimized_hamiltonian.idx().row);

    stats.moments_timer.tic();
    Compute::diagonal(moments, optimized_hamiltonian, config.algorithm);
    stats.moments_timer.toc();

    config.kernel.apply(moments.get());

    auto ldos = detail::reconstruct_function<real_t>(scaled_energy, moments.get().real());
    return ldos.template cast<double>();
}

template<class scalar_t, class Compute>
ArrayXcd StrategyTemplate<scalar_t, Compute>::greens(int row, int col, ArrayXd const& energy,
                                                     double broadening) {
    return std::move(greens_vector(row, {col}, energy, broadening).front());
}

template<class scalar_t, class Compute>
std::vector<ArrayXcd>
StrategyTemplate<scalar_t, Compute>::greens_vector(int row, std::vector<int> const& cols,
                                                   ArrayXd const& energy, double broadening) {
    assert(!cols.empty());
    auto const scale = bounds.scaling_factors();
    auto const scaled_energy = bounds.scaled(energy.template cast<real_t>());
    auto const num_moments = config.kernel.required_num_moments(broadening / scale.a);

    optimized_hamiltonian.optimize_for({row, cols}, scale);
    auto const& idx = optimized_hamiltonian.idx();
    stats = {num_moments, optimized_hamiltonian.operations(num_moments),
             optimized_hamiltonian.memory_usage(), hamiltonian->rows() * sizeof(scalar_t)};

    if (idx.is_diagonal()) {
        auto moments = ExvalDiagonalMoments<scalar_t>(num_moments, idx.row);

        stats.moments_timer.tic();
        Compute::diagonal(moments, optimized_hamiltonian, config.algorithm);
        stats.moments_timer.toc();

        config.kernel.apply(moments.get());

        auto const greens = detail::reconstruct_greens(scaled_energy, moments.get());
        return {greens.template cast<std::complex<double>>()};
    } else {
        auto moments_vector = ExvalOffDiagonalMoments<scalar_t>(num_moments, idx);

        stats.moments_timer.tic();
        Compute::off_diagonal(moments_vector, optimized_hamiltonian, config.algorithm);
        stats.moments_timer.toc();

        for (auto& moments : moments_vector.get()) {
            config.kernel.apply(moments);
        }

        auto greens = std::vector<ArrayXcd>();
        greens.reserve(idx.cols.size());
        for (auto const& moments : moments_vector.get()) {
            auto const g = detail::reconstruct_greens(scaled_energy, moments);
            greens.push_back(g.template cast<std::complex<double>>());
        }
        return greens;
    }
}

template<class scalar_t, class Compute>
std::string StrategyTemplate<scalar_t, Compute>::report(bool shortform) const {
    return bounds.report(shortform)
           + optimized_hamiltonian.report(stats.num_moments, shortform)
           + stats.report(shortform)
           + (shortform ? "|" : "Total time:");
}

struct DefaultCompute {
    template<class Moments>
    struct Diagonal {
        Moments& moments;
        OptimizedSizes const& sizes;
        AlgorithmConfig const& config;

        template<class Matrix>
        void operator()(Matrix const& h2) {
            using namespace calc_moments::diagonal;
            if (config.optimal_size && config.interleaved) {
                opt_size_and_interleaved(moments, h2, sizes);
            } else if (config.interleaved) {
                interleaved(moments, h2, sizes);
            } else if (config.optimal_size) {
                opt_size(moments, h2, sizes);
            } else {
                basic(moments, h2);
            }
        }
    };

    template<class Moments, class OptimizedHamiltonian>
    static void diagonal(Moments& m, OptimizedHamiltonian const& oh, AlgorithmConfig const& c) {
        oh.matrix().match(Diagonal<Moments>{m, oh.sizes(), c});
    }

    template<class Moments>
    struct OffDiagonal {
        Moments& moments;
        OptimizedSizes const& sizes;
        AlgorithmConfig const& config;

        template<class Matrix>
        void operator()(Matrix const& h2) {
            using namespace calc_moments::off_diagonal;
            if (config.optimal_size && config.interleaved) {
                opt_size_and_interleaved(moments, h2, sizes);
            } else if (config.interleaved) {
                interleaved(moments, h2, sizes);
            } else if (config.optimal_size) {
                opt_size(moments, h2, sizes);
            } else {
                basic(moments, h2);
            }
        }
    };

    template<class Moments, class OptimizedHamiltonian>
    static void off_diagonal(Moments& m, OptimizedHamiltonian const& oh, AlgorithmConfig const& c) {
        oh.matrix().match(OffDiagonal<Moments>{m, oh.sizes(), c});
    }
};

CPB_INSTANTIATE_TEMPLATE_CLASS_VARGS(StrategyTemplate, DefaultCompute)

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
