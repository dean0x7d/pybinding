#pragma once
#include "leads/Spec.hpp"
#include "system/System.hpp"

#include <vector>

namespace cpb { namespace leads {

/**
 Lead structural information: sites, hoppings and connection to main system
 */
struct Structure {
    std::vector<int> indices; ///< map from lead Hamiltonian indices to main system indices
    System system; ///< description of lead sites and boundaries

    Structure(Foundation const& foundation, HamiltonianIndices const& indices,
              Spec const& spec);

    /// Return the lead index corresponding to the main system Hamiltonian index
    int lead_index(int system_index) const {
        auto const it = std::find(indices.begin(), indices.end(), system_index);
        if (it == indices.end()) {
            return -1;
        } else {
            return static_cast<int>(it - indices.begin());
        }
    }
};

}} // namespace cpb::leads
