#include "leads/Structure.hpp"
#include "system/Foundation.hpp"

namespace cpb { namespace leads {

Structure::Structure(Foundation const& foundation, Spec const& lead)
    : system(foundation.get_lattice().site_registry(),
             foundation.get_lattice().hopping_registry()) {
    auto const& lattice = foundation.get_lattice();
    auto const& finalized_indices = foundation.get_finalized_indices();

    auto const shift = Cartesian(static_cast<float>(lead.sign) * lattice.vector(lead.axis));
    auto const junction = detail::Junction(foundation, lead);
    auto const slice = foundation[junction.slice_index];

    indices = [&]{
        auto indices = std::vector<int>();
        indices.reserve(static_cast<size_t>(junction.is_valid.count()));

        for (auto const& site : slice) {
            if (junction.is_valid[site.get_slice_idx()]) {
                indices.push_back(finalized_indices[site]);
            }
        }
        return indices;
    }();

    /*system*/ {
        auto const size = static_cast<int>(indices.size());
        system.positions.resize(size);
        system.hopping_blocks = {size, system.hopping_registry.name_map()};

        for (auto const& site : slice) {
            if (!junction.is_valid[site.get_slice_idx()]) {
                continue;
            }
            auto const index = lead_index(finalized_indices[site]);

            system.positions[index] = site.get_position() + shift;
            system.compressed_sublattices.add(SiteID{site.get_alias_id()}, site.get_norb());

            site.for_each_neighbor([&](Site neighbor, Hopping hopping) {
                auto const neighbor_index = lead_index(finalized_indices[neighbor]);
                if (neighbor_index >= 0 && !hopping.is_conjugate) {
                    system.hopping_blocks.add(hopping.family_id, index, neighbor_index);
                }
            });
        }
        system.compressed_sublattices.verify(size);
    }

    system.boundaries.push_back([&]{
        auto const size = static_cast<int>(indices.size());
        auto hopping_blocks = HoppingBlocks(size, system.hopping_registry.name_map());

        for (auto const& site : slice) {
            if (!junction.is_valid[site.get_slice_idx()]) {
                continue;
            }

            auto const shifted_site = [&]{
                Index3D shift_index = Index3D::Zero();
                shift_index[lead.axis] = lead.sign;
                return site.shifted(shift_index);
            }();

            auto const index = lead_index(finalized_indices[site]);
            shifted_site.for_each_neighbor([&](Site neighbor, Hopping hopping) {
                auto const neighbor_index = lead_index(finalized_indices[neighbor]);
                if (neighbor_index >= 0) {
                    hopping_blocks.add(hopping.family_id, index, neighbor_index);
                }
            });
        }

        return System::Boundary{hopping_blocks, shift};
    }());
}

}} // namespace cpb::leads
