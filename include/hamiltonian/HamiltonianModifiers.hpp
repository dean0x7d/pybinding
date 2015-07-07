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
class OnsiteModifier {
public:
    virtual ~OnsiteModifier() = default;
    virtual bool is_complex() const { return false; }

    /// Get the value of the potential at the given coordinates.
    virtual void apply(ArrayXf& potential, const CartesianArray& position) const = 0;
    virtual void apply(ArrayXcf& potential, const CartesianArray& position) const = 0;
    virtual void apply(ArrayXd& potential, const CartesianArray& position) const = 0;
    virtual void apply(ArrayXcd& potential, const CartesianArray& position) const = 0;
};

/**
Abstract base class for hopping energy modifiers.
*/
class HoppingModifier {
public:
    virtual ~HoppingModifier() = default;
    virtual bool is_complex() const { return false; }

    virtual void apply(ArrayXf& hopping, const CartesianArray& p1, const CartesianArray& p2) const = 0;
    virtual void apply(ArrayXd& hopping, const CartesianArray& p1, const CartesianArray& p2) const = 0;
    virtual void apply(ArrayXcf& hopping, const CartesianArray& p1, const CartesianArray& p2) const = 0;
    virtual void apply(ArrayXcd& hopping, const CartesianArray& p1, const CartesianArray& p2) const = 0;
};


class HamiltonianModifiers {
public:
    bool add_unique(const std::shared_ptr<const OnsiteModifier>& m);
    bool add_unique(const std::shared_ptr<const HoppingModifier>& m);

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
    std::vector<std::shared_ptr<const OnsiteModifier>> onsite;
    std::vector<std::shared_ptr<const HoppingModifier>> hopping;
};

template<class scalar_t, class Fn>
void HamiltonianModifiers::apply_to_onsite(System const& system, Fn lambda) const {
    auto const num_sites = system.num_sites();
    auto potential = ArrayX<scalar_t>{};

    if (system.lattice.has_onsite_potential) {
        potential.resize(num_sites);
        transform(system.sublattices, potential, [&](sub_id id) {
            return static_cast<scalar_t>(system.lattice[id].onsite);
        });
    }

    if (!onsite.empty()) {
        if (potential.size() == 0)
            potential.setZero(num_sites);

        for (auto const& modifier : onsite)
            modifier->apply(potential, system.positions);
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
    /*
     Applying modifiers to each hopping individually would be slow.
     Passing all the values in one call would require a lot of memory.
     The loop below buffers hoppings to balance performance and memory usage.
    */
    // TODO: experiment with buffer_size -> currently: hoppings + pos1 + pos2 is about 3MB
    static constexpr auto buffer_size = 100000;
    auto hoppings = ArrayX<scalar_t>{buffer_size};
    auto pos1 = CartesianArray{buffer_size};
    auto pos2 = CartesianArray{buffer_size};

    auto const& base_hopping_energies = system.lattice.hopping_energies;
    auto hopping_csr_matrix = sparse::make_loop(system.hoppings);

    hopping_csr_matrix.buffered_for_each(
        buffer_size,
        // fill buffer
        [&](int row, int col, hop_id id, int n) {
            hoppings[n] = num::complex_cast<scalar_t>(base_hopping_energies[id]);
            std::make_pair(pos1[n], pos2[n]) = system.get_position_pair(row, col);
        },
        // process buffer
        [&](int start_row, int start_idx, int size) {
            if (size < buffer_size) {
                hoppings.conservativeResize(size);
                pos1.conservativeResize(size);
                pos2.conservativeResize(size);
            }

            for (auto const& modifier : hopping)
                modifier->apply(hoppings, pos1, pos2);

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

} // namespace tbm
