#pragma once
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
    
    /// Apply hopping modifiers to the given system and pass the results to the lambda function
    template<class scalar_t, class S, class Fn>
    void apply_to_hoppings(const S& system, Fn lambda) const;

    void clear();
    
public:
    static constexpr auto chunks_size = 100000;
    // Keep modifiers as unique elements but insertion order must be preserved (don't use std::set)
    std::vector<std::shared_ptr<const OnsiteModifier>> onsite;
    std::vector<std::shared_ptr<const HoppingModifier>> hopping;
};

template<class scalar_t, class S, class Fn>
void HamiltonianModifiers::apply_to_hoppings(const S& system, Fn lambda) const {
    // from/to hopping indices and values
    auto hopping_indices = std::vector<std::tuple<int, int>>(chunks_size);
    ArrayX<scalar_t> hopping_values = ArrayX<scalar_t>::Zero(chunks_size);
    // current sparse matrix row
    auto row = 0;
    // positions of sites i and j
    auto pi = CartesianArray{chunks_size};
    auto pj = CartesianArray{chunks_size};
    
    /* 
     Applying modifiers to each hopping individually would require a large number of virtual
     function calls (slow). Passing all the values in one call would be fast but it would require
     a lot of memory. To balance memory usage and performance the hoppings are processed in chunks.
    */
    auto next_chunk = [&]() {
        auto n = 0;
        const auto limit = chunks_size - system.lattice.max_hoppings();
        for (;row < system.hoppings.rows() && n <= limit; ++row) {
            for (auto it = sparse_row(system.hoppings, row); it; ++it, ++n) {
                hopping_indices[n] = std::make_tuple(it.row(), it.col());
                auto const& hopping_energy = system.lattice.hopping_energies[it.value()];
                hopping_values[n] = num::complex_cast<scalar_t>(hopping_energy);
                std::make_pair(pi[n], pj[n]) = system.get_position_pair(it.row(), it.col());
            }
        }
        
        for (const auto& modifier : hopping)
            modifier->apply(hopping_values, pi, pj);
        
        return n; // final size of this chunk
    };
    
    while (auto size = next_chunk()) {
        for (int n = 0; n < size; ++n) {
            int i, j;
            std::tie(i, j) = hopping_indices[n];
            lambda(i, j, hopping_values[n]);
        }
    }
}

} // namespace tbm
