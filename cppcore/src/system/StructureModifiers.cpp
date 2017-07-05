#include "system/StructureModifiers.hpp"

#include "system/Foundation.hpp"
#include "system/System.hpp"

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

void apply(HoppingGenerator const& g, System& s) {
    auto const& lattice = s.lattice;
    auto const sublattices = s.compressed_sublattices.decompressed();
    auto const family_id = lattice.hopping_family(g.name).family_id;
    auto pairs = g.make(s.positions, {sublattices, lattice.sub_name_map()});
    s.hopping_blocks.append(family_id, std::move(pairs.from), std::move(pairs.to));
}

} // namespace cpb
