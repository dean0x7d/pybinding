#pragma once
#include "kpm/Core.hpp"

namespace cpb { namespace kpm {

/**
 Default CPU implementation for computing KPM moments, see `Core`
 */
class DefaultCompute : public Compute::Interface {
public:
    DefaultCompute(idx_t num_threads = -1);

    void moments(MomentsRef m, Starter const& s, AlgorithmConfig const& ac,
                 OptimizedHamiltonian const& oh) const override;

private:
    idx_t num_threads;
};

}} // namespace cpb::kpm
