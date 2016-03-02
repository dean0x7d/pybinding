#pragma once
#include "system/Lattice.hpp"

#include "support/dense.hpp"

#include <vector>
#include <algorithm>
#include <memory>

namespace tbm {

/**
 Modify the state (valid or invalid) of lattice sites, e.g. to create vacancies
 */
class SiteStateModifier {
public:
    using Function = std::function<void(ArrayX<bool>& /*state*/, CartesianArray const& /*pos*/,
                                        SubIdRef /*sublattice*/)>;
    Function apply; ///< to be user-implemented

    SiteStateModifier(Function const& apply) : apply(apply) {}
};

/**
 Modify the position of lattice sites, e.g. to apply geometric deformations
 */
class PositionModifier {
public:
    using Function = std::function<void(CartesianArray& /*position*/, SubIdRef /*sublattice*/)>;
    Function apply; ///< to be user-implemented

    PositionModifier(Function const& apply) : apply(apply) {}
};

struct SystemModifiers {
    std::vector<SiteStateModifier> state;
    std::vector<PositionModifier> position;

    void clear();
    bool empty() const;
};

} // namespace tbm
