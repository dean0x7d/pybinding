#include "system/SystemModifiers.hpp"

#include "system/Foundation.hpp"

namespace cpb {

void apply(SiteStateModifier const& m, Foundation& f) {
    for (auto const& pair : f.get_lattice().get_sublattices()) {
        auto slice = f[pair.second.unique_id];
        m.apply(slice.get_states(), slice.get_positions(), pair.first);
    }

    if (m.min_neighbors > 0) {
        remove_dangling(f, m.min_neighbors);
    }
}

void apply(PositionModifier const& m, Foundation& f) {
    for (auto const& pair : f.get_lattice().get_sublattices()) {
        auto slice = f[pair.second.unique_id];
        m.apply(slice.get_positions(), pair.first);
    }
}

} // namespace cpb
