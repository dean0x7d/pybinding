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

void apply(SiteStateModifier const& m, System& s) {
    if (s.is_valid.size() == 0) {
        s.is_valid = ArrayX<bool>::Constant(s.num_sites(), true);
    }

    for (auto const& sub : s.compressed_sublattices) {
        m.apply(s.is_valid.segment(sub.sys_start(), sub.num_sites()),
                s.positions.segment(sub.sys_start(), sub.num_sites()),
                s.site_registry.name(sub.id()));
    }

    if (m.min_neighbors > 0) {
        throw std::runtime_error("Eliminating dangling bonds after a generator "
                                 "has not been implemented yet");
    }
}

void apply(PositionModifier const& m, Foundation& f) {
    for (auto const& pair : f.get_lattice().get_sublattices()) {
        auto slice = f[pair.second.unique_id];
        m.apply(slice.get_positions(), pair.first);
    }
}

void apply(PositionModifier const& m, System& s) {
    for (auto const& sub : s.compressed_sublattices) {
        m.apply(s.positions.segment(sub.sys_start(), sub.num_sites()),
                s.site_registry.name(sub.id()));
    }
}

void apply(HoppingGenerator const& g, System& s) {
    detail::remove_invalid(s);

    auto const sublattices = s.compressed_sublattices.decompressed();
    auto const family_id = s.hopping_registry.id(g.name);
    auto pairs = g.make(s.positions, {sublattices, s.site_registry.name_map()});
    s.hopping_blocks.append(family_id, std::move(pairs.from), std::move(pairs.to));
}

} // namespace cpb
