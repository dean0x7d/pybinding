#pragma once
#include "system/System.hpp"

#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"

#include "detail/algorithm.hpp"

#include <vector>
#include <memory>

namespace cpb {

/**
 Modify the onsite energy, e.g. to apply an electric field
*/
class OnsiteModifier {
public:
    using Function = std::function<void(ComplexArrayRef energy, CartesianArrayConstRef positions,
                                        SubIdRef sublattice)>;

    Function apply; ///< to be user-implemented
    bool is_complex = false; ///< the modeled effect requires complex values
    bool is_double = false; ///< the modeled effect requires double precision

    OnsiteModifier(Function const& apply, bool is_complex = false, bool is_double = false)
        : apply(apply), is_complex(is_complex), is_double(is_double) {}

    explicit operator bool() const { return static_cast<bool>(apply); }
};

/**
 Modify the hopping energy, e.g. to apply a magnetic field
*/
class HoppingModifier {
public:
    using Function = std::function<void(ComplexArrayRef /*energy*/, CartesianArray const& /*pos1*/,
                                        CartesianArray const& /*pos2*/, HopIdRef /*hopping_id*/)>;
    Function apply; ///< to be user-implemented
    bool is_complex = false; ///< the modeled effect requires complex values
    bool is_double = false; ///< the modeled effect requires double precision

    HoppingModifier(Function const& apply, bool is_complex = false, bool is_double = false)
        : apply(apply), is_complex(is_complex), is_double(is_double) {}

    explicit operator bool() const { return static_cast<bool>(apply); }
};

/**
 Container with some convenience functions
 */
struct HamiltonianModifiers {
    std::vector<OnsiteModifier> onsite;
    std::vector<HoppingModifier> hopping;

    /// Do any of the modifiers require complex numbers?
    bool any_complex() const;

    /// Do any of the modifiers require double precision?
    bool any_double() const;

    /// Remove all modifiers
    void clear();

    /// Apply onsite modifiers to the given system and pass results to function:
    ///     lambda(int i, scalar_t onsite)
    template<class scalar_t, class Fn>
    void apply_to_onsite(System const& system, Fn lambda) const;

    /// Apply hopping modifiers to the given system (or boundary) and pass results to:
    ///     lambda(int i, int j, scalar_t hopping)
    template<class scalar_t, class Fn>
    void apply_to_hoppings(System const& system, Fn fn) const {
        apply_to_hoppings_impl<scalar_t>(system, system.positions, system.lattice, fn);
    };

    template<class scalar_t, class Fn>
    void apply_to_hoppings(System const& system, size_t boundary_index, Fn fn) const {
        apply_to_hoppings_impl<scalar_t>(system.boundaries[boundary_index],
                                         system.positions, system.lattice, fn);
    };

private:
    template<class scalar_t, class SystemOrBoundary, class Fn>
    void apply_to_hoppings_impl(SystemOrBoundary const& system, CartesianArray const& positions,
                                Lattice const& lattice, Fn lambda) const;
};

namespace detail {
    inline Cartesian shifted(Cartesian pos, System const&) { return pos; }
    inline Cartesian shifted(Cartesian pos, System::Boundary const& b) { return pos - b.shift; }
}

template<class scalar_t, class Fn>
void HamiltonianModifiers::apply_to_onsite(System const& system, Fn lambda) const {
    auto const& lattice = system.lattice;
    auto const has_intrinsic_onsite = lattice.has_onsite_energy();
    if (!has_intrinsic_onsite && onsite.empty()) {
        return;
    }

    for (auto const& sub : system.compressed_sublattices) {
        auto onsite_energy = ArrayX<scalar_t>::Zero(sub.ham_size()).eval();

        if (lattice.has_onsite_energy()) {
            // Intrinsic lattice onsite energy -- just replicate the value at each site
            auto const vector = lattice[sub.alias_id()].energy_vector_as<scalar_t>();
            onsite_energy = vector.replicate(sub.num_sites(), 1);
        }

        if (!onsite.empty()) {
            // Apply all user-defined onsite modifier functions
            auto onsite_ref = arrayref(onsite_energy.data(), sub.num_sites(), sub.num_orbitals());
            if (sub.num_orbitals() == 1) { onsite_ref.ndim = 1; } // squeeze dimensions

            auto const position_ref = system.positions.segment(sub.sys_start(), sub.num_sites());
            auto const sub_ids = ArrayX<sub_id>::Constant(sub.num_sites(), sub.alias_id());

            for (auto const& modifier : onsite) {
                modifier.apply(onsite_ref, position_ref, {sub_ids, lattice.sub_name_map()});
            }
        }

        for (auto n = idx_t{0}; n < onsite_energy.size(); ++n) {
            auto const v = onsite_energy[n];
            if (v != scalar_t{0}) {
                lambda(sub.ham_start() + n, v);
            }
        }
    }
}

template<class scalar_t, class SystemOrBoundary, class Fn>
void HamiltonianModifiers::apply_to_hoppings_impl(SystemOrBoundary const& system,
                                                  CartesianArray const& positions,
                                                  Lattice const& lattice, Fn lambda) const {
    if (hopping.empty()) {
        // fast path: modifiers don't need to be applied
        for (auto const& block : system.hopping_blocks) {
            auto const family_id = static_cast<hop_id>(block.family_id());
            auto const energy = lattice(family_id).energy_matrix_as<scalar_t>()(0, 0);

            for (auto const& coo : block.coordinates()) {
                lambda(coo.row, coo.col, energy);
            }
        }

        return;
    }

    /*
     Applying modifiers to each hopping individually would be slow.
     Passing all the values in one call would require a lot of memory.
     The loop below buffers hoppings to balance performance and memory usage.
    */

    // TODO: experiment with buffer_size -> currently: hoppings + pos1 + pos2 is about 3MB
    auto const buffer_size = [&]{
        constexpr auto max_buffer_size = idx_t{100000};
        auto const max_hoppings = system.hopping_blocks.nnz();
        return std::min(max_hoppings, max_buffer_size);
    }();

    for (auto const& block : system.hopping_blocks) {
        auto const family_id = static_cast<hop_id>(block.family_id());
        auto const energy = lattice(family_id).energy_matrix_as<scalar_t>()(0, 0);

        auto hoppings = ArrayX<scalar_t>::Constant(buffer_size, energy).eval();
        auto hop_ids = ArrayX<hop_id>::Constant(buffer_size, family_id).eval();
        auto pos1 = CartesianArray(buffer_size);
        auto pos2 = CartesianArray(buffer_size);

        buffered_for_each(
            block.coordinates(),
            buffer_size,
            /*fill buffer*/ [&](HoppingBlocks::COO const& coo, idx_t n) {
                pos1[n] = positions[coo.row];
                pos2[n] = detail::shifted(positions[coo.col], system);
            },
            /*filled*/ [&](idx_t size) {
                if (size < buffer_size) { // only true for the last batch
                    hoppings.conservativeResize(size);
                    hop_ids.conservativeResize(size);
                    pos1.conservativeResize(size);
                    pos2.conservativeResize(size);
                }

                for (auto const& modifier : hopping) {
                    modifier.apply(arrayref(hoppings), pos1, pos2,
                                   {hop_ids, lattice.hop_name_map()});
                }
            },
            /*process buffer*/ [&](HoppingBlocks::COO const& coo, idx_t n) {
                if (hoppings[n] != scalar_t{0}) {
                    lambda(coo.row, coo.col, hoppings[n]);
                }
            }
        ); // buffered_for_each
    }
}

} // namespace cpb
