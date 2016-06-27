#pragma once
#include "system/System.hpp"

#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"

#include <vector>
#include <algorithm>
#include <memory>

namespace cpb {

/**
 Modify the onsite energy, e.g. to apply an electric field
*/
class OnsiteModifier {
public:
    using Function = std::function<void(ComplexArrayRef /*energy*/, CartesianArray const& /*pos*/,
                                        SubIdRef /*sublattice*/)>;
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

        for (auto const& modifier : onsite) {
            modifier.apply(arrayref(potential), system.positions,
                           {system.sublattices, system.lattice.sub_name_map});
        }
    }

    if (potential.size() > 0) {
        for (int i = 0; i < num_sites; ++i) {
            if (potential[i] != scalar_t{0})
                lambda(i, potential[i]);
        }
    }
}

template<class scalar_t, class SystemOrBoundary, class Fn>
void HamiltonianModifiers::apply_to_hoppings_impl(SystemOrBoundary const& system,
                                                  CartesianArray const& positions,
                                                  Lattice const& lattice, Fn lambda) const {
    auto const& base_hopping_energies = lattice.hopping_energies;
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
                pos1[n] = positions[row];
                pos2[n] = detail::shifted(positions[col], system);
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

                for (auto const& modifier : hopping) {
                    modifier.apply(arrayref(hoppings), pos1, pos2,
                                   {hop_ids, lattice.hop_name_map});
                }

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

} // namespace cpb
