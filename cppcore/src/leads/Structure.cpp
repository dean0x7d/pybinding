#include "leads/Structure.hpp"
#include "system/Foundation.hpp"

namespace cpb { namespace leads {

Structure::Structure(Foundation const& foundation, HamiltonianIndices const& hamiltonian_indices,
                     Spec const& lead)
    : system(foundation.get_lattice()) {
    auto const& lattice = foundation.get_lattice();
    auto const shift = static_cast<float>(lead.sign) * lattice.vectors[lead.axis];
    auto const junction = detail::Junction(foundation, lead);
    auto const slice = foundation[junction.slice_index];

    indices = [&]{
        auto indices = std::vector<int>();
        indices.reserve(junction.is_valid.count());

        for (auto const& site : slice) {
            if (junction.is_valid[site.get_slice_idx()]) {
                indices.push_back(hamiltonian_indices[site]);
            }
        }
        return indices;
    }();

    /*system*/ {
        auto const size = static_cast<int>(indices.size());
        system.positions.resize(size);
        system.sublattices.resize(size);
        system.hoppings.resize(size, size);

        auto const reserve_nonzeros = (lattice.max_hoppings() * size) / 2;
        auto matrix_view = compressed_inserter(system.hoppings, reserve_nonzeros);

        for (auto const& site : slice) {
            if (!junction.is_valid[site.get_slice_idx()]) {
                continue;
            }
            auto const index = lead_index(hamiltonian_indices[site]);

            system.positions[index] = site.get_position() + shift;
            system.sublattices[index] = lattice[site.get_sublattice()].alias;

            matrix_view.start_row(index);
            site.for_each_neighbour([&](Site neighbor, Hopping hopping) {
                auto const neighbor_index = lead_index(hamiltonian_indices[neighbor]);
                if (neighbor_index >= 0 && !hopping.is_conjugate) {
                    matrix_view.insert(neighbor_index, hopping.id);
                }
            });
        }
        matrix_view.compress();
    }

    system.boundaries.push_back([&]{
        auto const size = static_cast<int>(indices.size());
        auto matrix = SparseMatrixX<hop_id>(size, size);
        auto matrix_view = compressed_inserter(matrix, size * lattice.max_hoppings());

        for (auto const& site : slice) {
            if (!junction.is_valid[site.get_slice_idx()]) {
                continue;
            }

            auto const shifted_site = [&]{
                Index3D shift_index = Index3D::Zero();
                shift_index[lead.axis] = lead.sign;
                return site.shifted(shift_index);
            }();

            matrix_view.start_row();
            shifted_site.for_each_neighbour([&](Site neighbor, Hopping hopping) {
                auto const index = lead_index(hamiltonian_indices[neighbor]);
                if (index >= 0) {
                    matrix_view.insert(index, hopping.id);
                }
            });
        }
        matrix_view.compress();

        return System::Boundary{matrix, shift};
    }());
}

}} // namespace cpb::leads
