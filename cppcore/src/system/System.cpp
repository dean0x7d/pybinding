#include "system/System.hpp"

#include "system/Foundation.hpp"
#include "system/Symmetry.hpp"

namespace cpb {

System::System(Foundation const& foundation, HamiltonianIndices const& hamiltonian_indices,
               TranslationalSymmetry const& symmetry, HoppingGenerators const& hopping_generators)
    : lattice(foundation.get_lattice()) {
    detail::populate_system(*this, foundation, hamiltonian_indices);
    if (symmetry) {
        detail::populate_boundaries(*this, foundation, hamiltonian_indices, symmetry);
    }

    if (!hopping_generators.empty()) {
        for (auto const& gen : hopping_generators) {
            detail::add_extra_hoppings(*this, gen);
        }
        hoppings.makeCompressed();
        has_unbalanced_hoppings = true;
    }

    if (num_sites() == 0)
        throw std::runtime_error{"Impossible system: built 0 lattice sites"};
}

int System::find_nearest(Cartesian target_position, sub_id target_sublattice) const {
    auto nearest_index = 0;
    auto min_distance = (positions[0] - target_position).norm();

    for (auto i = 1; i < num_sites(); ++i) {
        if (target_sublattice >= 0 && sublattices[i] != target_sublattice)
            continue; // only check the target sublattice (if any)

        auto const distance = (positions[i] - target_position).norm();
        if (distance < min_distance) {
            min_distance = distance;
            nearest_index = i;
        }
    }
    
    return nearest_index;
}

namespace detail {

void populate_system(System& system, Foundation const& foundation,
                     HamiltonianIndices const& hamiltonian_indices) {
    auto const size = hamiltonian_indices.size();
    system.positions.resize(size);
    system.sublattices.resize(size);
    system.hoppings.resize(size, size);

    auto const& lattice = foundation.get_lattice();
    auto const reserve_nonzeros = (lattice.max_hoppings() * size) / 2;
    auto matrix_view = compressed_inserter(system.hoppings, reserve_nonzeros);

    for (auto const& site : foundation) {
        auto const index = hamiltonian_indices[site];
        if (index < 0)
            continue; // invalid site

        system.positions[index] = site.get_position();
        system.sublattices[index] = lattice[site.get_sublattice()].alias;

        matrix_view.start_row(index);
        site.for_each_neighbour([&](Site neighbor, Hopping hopping) {
            auto const neighbor_index = hamiltonian_indices[neighbor];
            if (neighbor_index < 0)
                return; // invalid

            if (!hopping.is_conjugate) // only make half the matrix, other half is the conjugate
                matrix_view.insert(neighbor_index, hopping.id);
        });
    }
    matrix_view.compress();
}

void populate_boundaries(System& system, Foundation const& foundation,
                         HamiltonianIndices const& hamiltonian_indices,
                         TranslationalSymmetry const& symmetry) {
    // a boundary is added first to prevent copying of Eigen::SparseMatrix
    // --> revise when Eigen types become movable

    auto const size = hamiltonian_indices.size();
    auto const& lattice = foundation.get_lattice();

    system.boundaries.emplace_back();
    for (const auto& translation : symmetry.translations(foundation)) {
        auto& boundary = system.boundaries.back();

        boundary.shift = translation.shift_lenght;
        boundary.hoppings.resize(size, size);

        // the reservation number is intentionally overestimated
        auto const reserve_nonzeros = [&]{
            auto nz = static_cast<int>(lattice.sublattices.size() * lattice.max_hoppings() / 2);
            for (auto i = 0; i < translation.boundary_slice.ndims(); ++i) {
                if (translation.boundary_slice[i].end < 0)
                    nz *= foundation.get_size()[i];
            }
            return nz;
        }();
        auto boundary_matrix_view = compressed_inserter(boundary.hoppings, reserve_nonzeros);

        for (auto const& site : foundation[translation.boundary_slice]) {
            if (!site.is_valid())
                continue;

            boundary_matrix_view.start_row(hamiltonian_indices[site]);

            auto const shifted_site = site.shifted(translation.shift_index);
            // the site is shifted to the opposite edge of the translation unit
            shifted_site.for_each_neighbour([&](Site neighbor, Hopping hopping) {
                auto const neighbor_index = hamiltonian_indices[neighbor];
                if (neighbor_index < 0)
                    return; // invalid

                boundary_matrix_view.insert(neighbor_index, hopping.id);
            });
        }
        boundary_matrix_view.compress();

        if (boundary.hoppings.nonZeros() > 0)
            system.boundaries.emplace_back();
    }
    system.boundaries.pop_back();
}

void add_extra_hoppings(System& system, HoppingGenerator const& gen) {
    auto const& lattice = system.lattice;
    auto const pairs = gen.make(system.positions, {system.sublattices, lattice.sub_name_map});

    system.hoppings.reserve([&]{
        auto reserve = ArrayXi(ArrayXi::Zero(system.num_sites()));
        for (auto i = 0; i < pairs.from.size(); ++i) {
            auto const row = std::min(pairs.from[i], pairs.to[i]); // upper triangular format
            reserve[row] += 1;
        }
        return reserve;
    }());

    auto const hopping_id = [&]{
        auto const it = lattice.hop_name_map.find(gen.name);
        assert(it != lattice.hop_name_map.end());
        return it->second;
    }();

    for (auto i = 0; i < pairs.from.size(); ++i) {
        auto m = pairs.from[i];
        auto n = pairs.to[i];
        if (m > n) { // ensure that only the upper triangle of the matrix is populated
            std::swap(m, n);
        }
        system.hoppings.coeffRef(m, n) = hopping_id;
    }
}

} // namespace detail


ArrayXi nonzeros_per_row(SparseMatrixX<hop_id> const& hoppings,  bool has_onsite_energy) {
    assert(hoppings.isCompressed());
    auto const outer_size = hoppings.outerSize();
    auto nnz = has_onsite_energy ? ArrayXi(ArrayXi::Ones(outer_size))
                                 : ArrayXi(ArrayXi::Zero(outer_size));

    auto const indptr = hoppings.outerIndexPtr();
    for (auto i = 0; i < outer_size; ++i) {
        nnz[i] += indptr[i + 1] - indptr[i];
    }

    // The hop_id sparse matrix is triangular, so the total number of non-zeros per row
    // must also include an extra 1 per column index to account for the other triangle.
    auto const indices_size = hoppings.nonZeros();
    auto const indices = hoppings.innerIndexPtr();
    for (auto i = 0; i < indices_size; ++i) {
        nnz[indices[i]] += 1;
    }

    return nnz;
}

} // namespace cpb
