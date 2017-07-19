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

void apply(SiteGenerator const& g, System& s) {
    detail::remove_invalid(s);

    auto const new_positions = g.make(s);
    auto const norb = g.energy.rows();
    auto const nsites = new_positions.size();
    s.compressed_sublattices.add(s.site_registry.id(g.name), norb, nsites);
    s.hopping_blocks.add_sites(nsites);
    s.positions = concat(s.positions, new_positions);
}

void apply(HoppingGenerator const& g, System& s) {
    detail::remove_invalid(s);

    auto pairs = g.make(s);
    s.hopping_blocks.append(s.hopping_registry.id(g.name),
                            std::move(pairs.from), std::move(pairs.to));
}

} // namespace cpb
