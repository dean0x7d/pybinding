#pragma once
#include "kpm/Core.hpp"

namespace cpb { namespace kpm {

/**
 Default CPU implementation for computing KPM moments, see `Core`
 */
class DefaultCompute : public Compute::Interface {
public:
    using ProgressCallback = std::function<void (idx_t delta, idx_t total)>;

    DefaultCompute(idx_t num_threads = -1, ProgressCallback progress_callback = {});

    void moments(MomentsRef m, Starter const& s, AlgorithmConfig const& ac,
                 OptimizedHamiltonian const& oh) const override;

    idx_t get_num_threads() const { return num_threads; }

    void progress_start(idx_t total) const;
    void progress_update(idx_t delta, idx_t total) const;
    void progress_finish(idx_t total) const;

private:
    idx_t num_threads;
    ProgressCallback progress_callback;
};

}} // namespace cpb::kpm
