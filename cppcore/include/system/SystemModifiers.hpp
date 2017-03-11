#pragma once
#include "Lattice.hpp"

#include "numeric/dense.hpp"

#include <vector>
#include <algorithm>
#include <memory>

namespace cpb {

/**
 Modify the state (valid or invalid) of lattice sites, e.g. to create vacancies
 */
class SiteStateModifier {
public:
    using Function = std::function<void(Eigen::Ref<ArrayX<bool>> state, CartesianArrayConstRef pos,
                                        string_view sublattice)>;
    Function apply; ///< to be user-implemented
    int min_neighbors; ///< afterwards, remove sites with less than this number of neighbors

    SiteStateModifier(Function const& apply, int min_neighbors = 0)
        : apply(apply), min_neighbors(min_neighbors) {}
};

/**
 Modify the position of lattice sites, e.g. to apply geometric deformations
 */
class PositionModifier {
public:
    using Function = std::function<void(CartesianArrayRef position, string_view sublattice)>;
    Function apply; ///< to be user-implemented

    PositionModifier(Function const& apply) : apply(apply) {}
};

struct SystemModifiers {
    std::vector<SiteStateModifier> state;
    std::vector<PositionModifier> position;

    void clear();
    bool empty() const;
};

} // namespace cpb
