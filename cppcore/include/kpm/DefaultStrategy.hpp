#pragma once
#include "kpm/Strategy.hpp"

namespace cpb { namespace kpm {

/**
 Default CPU implementation for computing KPM moments, see `Strategy`
 */
template<class scalar_t>
class DefaultStrategy final : public StrategyTemplate<scalar_t> {
public:
    using StrategyTemplate<scalar_t>::StrategyTemplate;

    void compute(DiagonalMoments<scalar_t>& m, VectorX<scalar_t>&& r0,
                 AlgorithmConfig const& ac, OptimizedHamiltonian<scalar_t> const& oh) const final;
    void compute(OffDiagonalMoments<scalar_t>& m, VectorX<scalar_t>&& r0,
                 AlgorithmConfig const& ac, OptimizedHamiltonian<scalar_t> const& oh) const final;
};

CPB_EXTERN_TEMPLATE_CLASS(DefaultStrategy)

}} // namespace cpb::kpm
