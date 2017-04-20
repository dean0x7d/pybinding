#include "kpm/DefaultStrategy.hpp"

#include "compute/kernel_polynomial.hpp"
#include "kpm/calc_moments.hpp"

namespace cpb { namespace kpm {

namespace {

template<class scalar_t>
struct ComputeDiagonal {
    DiagonalMoments<scalar_t>& moments;
    Starter const& starter;
    SliceMap const& map;
    AlgorithmConfig const& config;

    template<class T> using matches = std::is_same<scalar_t, typename T::Scalar>;

    template<class Matrix, std14::enable_if_t<!matches<Matrix>::value, int> = 0>
    void operator()(Matrix const&) {}

    template<class Matrix, std14::enable_if_t<matches<Matrix>::value, int> = 0>
    void operator()(Matrix const& h2) {
        using namespace calc_moments;
        simd::scope_disable_denormals guard;

        auto r0 = make_r0<scalar_t>(starter);
        auto r1 = make_r1(h2, r0);
        moments.collect_initial(r0, r1);

        if (config.optimal_size && config.interleaved) {
            diagonal::opt_size_and_interleaved(moments, std::move(r0), std::move(r1), h2, map);
        } else if (config.interleaved) {
            diagonal::interleaved(moments, std::move(r0), std::move(r1), h2, map);
        } else if (config.optimal_size) {
            diagonal::opt_size(moments, std::move(r0), std::move(r1), h2, map);
        } else {
            diagonal::basic(moments, std::move(r0), std::move(r1), h2);
        }
    }
};

template<class scalar_t>
struct ComputeOffDiagonal {
    OffDiagonalMoments<scalar_t>& moments;
    Starter const& starter;
    SliceMap const& map;
    AlgorithmConfig const& config;

    template<class T> using matches = std::is_same<scalar_t, typename T::Scalar>;

    template<class Matrix, std14::enable_if_t<!matches<Matrix>::value, int> = 0>
    void operator()(Matrix const&) {}

    template<class Matrix, std14::enable_if_t<matches<Matrix>::value, int> = 0>
    void operator()(Matrix const& h2) {
        using namespace calc_moments;
        simd::scope_disable_denormals guard;

        auto r0 = make_r0<scalar_t>(starter);
        auto r1 = make_r1(h2, r0);
        moments.collect_initial(r0, r1);

        if (config.optimal_size && config.interleaved) {
            off_diagonal::opt_size_and_interleaved(moments, std::move(r0), std::move(r1), h2, map);
        } else if (config.interleaved) {
            off_diagonal::interleaved(moments, std::move(r0), std::move(r1), h2, map);
        } else if (config.optimal_size) {
            off_diagonal::opt_size(moments, std::move(r0), std::move(r1), h2, map);
        } else {
            off_diagonal::basic(moments, std::move(r0), std::move(r1), h2);
        }
    }
};

} // anonymous namespace

template<class scalar_t>
void DefaultStrategy<scalar_t>::compute(
    DiagonalMoments<scalar_t>& m, Starter const& starter,
    AlgorithmConfig const& ac, OptimizedHamiltonian const& oh
) const {
    oh.matrix().match(ComputeDiagonal<scalar_t>{m, starter, oh.map(), ac});
}

template<class scalar_t>
void DefaultStrategy<scalar_t>::compute(
    OffDiagonalMoments<scalar_t>& m, Starter const& starter,
    AlgorithmConfig const& ac, OptimizedHamiltonian const& oh
) const {
    oh.matrix().match(ComputeOffDiagonal<scalar_t>{m, starter, oh.map(), ac});
}

CPB_INSTANTIATE_TEMPLATE_CLASS(DefaultStrategy)

}} // namespace cpb::kpm
