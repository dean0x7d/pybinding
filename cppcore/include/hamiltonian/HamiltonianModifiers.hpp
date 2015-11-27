#pragma once
#include "system/System.hpp"

#include "support/dense.hpp"
#include "support/sparse.hpp"

#include <vector>
#include <algorithm>
#include <memory>

namespace tbm {

/**
Abstract base class for onsite potential.
*/
class OnsiteModifierImpl {
public:
    virtual ~OnsiteModifierImpl() = default;
    virtual bool is_complex() const { return false; }

    /// Get the value of the potential at the given coordinates.
    virtual void apply(ArrayXf& potential, CartesianArray const& position,
                       ArrayX<sub_id> const& sublattices) const = 0;
    virtual void apply(ArrayXcf& potential, CartesianArray const& position,
                       ArrayX<sub_id> const& sublattices) const = 0;
    virtual void apply(ArrayXd& potential, CartesianArray const& position,
                       ArrayX<sub_id> const& sublattices) const = 0;
    virtual void apply(ArrayXcd& potential, CartesianArray const& position,
                       ArrayX<sub_id> const& sublattices) const = 0;
};

/**
Abstract base class for hopping energy modifiers.
*/
class HoppingModifierImpl {
public:
    virtual ~HoppingModifierImpl() = default;
    virtual bool is_complex() const { return false; }

    virtual void apply(ArrayXf& hopping, CartesianArray const& pos1, CartesianArray const& pos2,
                       ArrayX<hop_id> const& id) const = 0;
    virtual void apply(ArrayXd& hopping, CartesianArray const& pos1, CartesianArray const& pos2,
                       ArrayX<hop_id> const& id) const = 0;
    virtual void apply(ArrayXcf& hopping,  CartesianArray const& pos1, CartesianArray const& pos2,
                       ArrayX<hop_id> const& id) const = 0;
    virtual void apply(ArrayXcd& hopping,  CartesianArray const& pos1, CartesianArray const& pos2,
                       ArrayX<hop_id> const& id) const = 0;
};

using OnsiteModifier = std::shared_ptr<OnsiteModifierImpl const>;
using HoppingModifier = std::shared_ptr<HoppingModifierImpl const>;

class HamiltonianModifiers {
public:
    bool add_unique(OnsiteModifier const& m);
    bool add_unique(HoppingModifier const& m);

    /// Do any of the modifiers require complex numbers?
    bool any_complex() const;
    
    /// Apply onsite modifiers to the given system and pass results to function:
    ///     lambda(int i, scalar_t onsite)
    template<class scalar_t, class Fn>
    void apply_to_onsite(System const& system, Fn lambda) const;

    /// Apply hopping modifiers to the given system (or boundary) and pass results to:
    ///     lambda(int i, int j, scalar_t hopping)
    template<class scalar_t, class SystemOrBoundary, class Fn>
    void apply_to_hoppings(SystemOrBoundary const& system, Fn lambda) const;

    void clear();
    
public:
    // Keep modifiers as unique elements but insertion order must be preserved (don't use std::set)
    std::vector<OnsiteModifier> onsite;
    std::vector<HoppingModifier> hopping;
};

template<class scalar_t, class Fn>
void HamiltonianModifiers::apply_to_onsite(System const& system, Fn lambda) const {
    auto const num_sites = system.num_sites();
    auto potential = ArrayX<scalar_t>{};

    if (system.lattice.has_onsite_energy) {
        potential.resize(num_sites);
        transform(system.sublattices, potential, [&](sub_id id) {
            using real_t = num::get_real_t<scalar_t>;
            return static_cast<real_t>(system.lattice[id].onsite);
        });
    }

    if (!onsite.empty()) {
        if (potential.size() == 0)
            potential.setZero(num_sites);

        for (auto const& modifier : onsite)
            modifier->apply(potential, system.positions, system.sublattices);
    }

    if (potential.size() > 0) {
        for (int i = 0; i < num_sites; ++i) {
            if (potential[i] != scalar_t{0})
                lambda(i, potential[i]);
        }
    }
}

template<class scalar_t, class SystemOrBoundary, class Fn>
void HamiltonianModifiers::apply_to_hoppings(SystemOrBoundary const& system, Fn lambda) const {
    auto const& base_hopping_energies = system.lattice.hopping_energies;
    auto hopping_csr_matrix = sparse::make_loop(system.hoppings);

    if (hopping.empty()) {
        // fast path: modifiers don't need to be applied
        hopping_csr_matrix.for_each([&](int row, int col, hop_id id) {
            lambda(row, col, num::complex_cast<scalar_t>(base_hopping_energies[id]));
        });
    } else {
        /*
         Applying modifiers to each hopping individually would be slow.
         Passing all the values in one call would require a lot of memory.
         The loop below buffers hoppings to balance performance and memory usage.
        */
        // TODO: experiment with buffer_size -> currently: hoppings + pos1 + pos2 is about 3MB
        auto const buffer_size = [&]{
            constexpr auto max_buffer_size = 100000;
            auto const max_hoppings = system.hoppings.nonZeros();
            return std::min(max_hoppings, max_buffer_size);
        }();

        auto hoppings = ArrayX<scalar_t>{buffer_size};
        auto pos1 = CartesianArray{buffer_size};
        auto pos2 = CartesianArray{buffer_size};

        // TODO: Hopping IDs can be mapped directly from the matrix, thus removing the need for
        //       this temporary buffer. However `apply()` needs to accept an array map.
        auto hop_ids = ArrayX<hop_id>{buffer_size};

        hopping_csr_matrix.buffered_for_each(
            buffer_size,
            // fill buffer
            [&](int row, int col, hop_id id, int n) {
                hoppings[n] = num::complex_cast<scalar_t>(base_hopping_energies[id]);
                std::make_pair(pos1[n], pos2[n]) = system.get_position_pair(row, col);
                hop_ids[n] = id;
            },
            // process buffer
            [&](int start_row, int start_idx, int size) {
                if (size < buffer_size) {
                    hoppings.conservativeResize(size);
                    pos1.conservativeResize(size);
                    pos2.conservativeResize(size);
                    hop_ids.conservativeResize(size);
                }

                for (auto const& modifier : hopping)
                    modifier->apply(hoppings, pos1, pos2, hop_ids);

                hopping_csr_matrix.slice_for_each(
                    start_row, start_idx, size,
                    [&](int row, int col, hop_id, int n) {
                        if (hoppings[n] != scalar_t{0})
                            lambda(row, col, hoppings[n]);
                    }
                );
            }
        );
    }
}

} // namespace tbm
