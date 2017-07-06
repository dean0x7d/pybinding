#pragma once
#include "system/System.hpp"

#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"

#include "detail/macros.hpp"
#include "detail/algorithm.hpp"

#include <vector>
#include <memory>

namespace cpb {

/**
 Thrown by a modifier if it determines that complex numbers must be
 returned even though it was given real input data. The model will
 catch this and switch the scalar type to complex.
 */
class ComplexOverride : public std::exception {
public:
    char const* what() const noexcept override {
        return "Trying to return a complex result from a real modifier.";
    }
};

/**
 Modify the onsite energy, e.g. to apply an electric field
*/
class OnsiteModifier {
public:
    using Function = std::function<void(ComplexArrayRef energy, CartesianArrayConstRef positions,
                                        string_view sublattice)>;

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
    using Function = std::function<void(ComplexArrayRef energy, CartesianArrayConstRef pos1,
                                        CartesianArrayConstRef pos2, string_view hopping_family)>;
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
        apply_to_hoppings_impl<scalar_t>(system, system, fn);
    };

    template<class scalar_t, class Fn>
    void apply_to_hoppings(System const& system, size_t boundary_index, Fn fn) const {
        apply_to_hoppings_impl<scalar_t>(system, system.boundaries[boundary_index], fn);
    };

private:
    template<class scalar_t, class SystemOrBoundary, class Fn>
    void apply_to_hoppings_impl(System const& system, SystemOrBoundary const& system_or_boundary,
                                Fn lambda) const;
};

namespace detail {
    inline Cartesian shifted(Cartesian pos, System const&) { return pos; }
    inline Cartesian shifted(Cartesian pos, System::Boundary const& b) { return pos - b.shift; }
}

template<class scalar_t, class Fn>
void HamiltonianModifiers::apply_to_onsite(System const& system, Fn lambda) const {
    auto const has_intrinsic_onsite = system.site_registry.has_nonzero_energy();
    if (!has_intrinsic_onsite && onsite.empty()) {
        return;
    }

    for (auto const& sub : system.compressed_sublattices) {
        auto const nsites = sub.num_sites();
        auto const norb = sub.num_orbitals();
        auto onsite_energy = ArrayX<scalar_t>::Zero(nsites * norb * norb).eval();

        if (has_intrinsic_onsite) {
            // Intrinsic lattice onsite energy -- just replicate the value at each site
            auto const intrinsic_energy = num::force_cast<scalar_t>(
                system.site_registry.energy(sub.id())
            );

            auto start = idx_t{0};
            for (auto const& value : intrinsic_energy) {
                onsite_energy.segment(start, nsites).setConstant(value);
                start += nsites;
            }
        }

        if (!onsite.empty()) {
            // Apply all user-defined onsite modifier functions
            auto onsite_ref = (norb == 1) ? arrayref(onsite_energy.data(), nsites)
                                          : arrayref(onsite_energy.data(), norb, norb, nsites);
            auto const position_ref = system.positions.segment(sub.sys_start(), nsites);
            auto const sub_name = system.site_registry.name(sub.id());

            for (auto const& modifier : onsite) {
                modifier.apply(onsite_ref, position_ref, sub_name);
            }
        }

        // Pass along each onsite value at the correct Hamiltonian row and column indices
        auto const* data = onsite_energy.data();
        for (auto i = idx_t{0}; i < norb; ++i) {
            for (auto j = idx_t{0}; j < norb; ++j) {
                for (auto idx = sub.ham_start(), end = sub.ham_end(); idx < end; idx += norb) {
                    auto const value = *data++;
                    if (value != scalar_t{0}) {
                        lambda(i + idx, j + idx, value);
                    }
                }
            }
        }
    } // for (auto const& sub : system.compressed_sublattices)
}

/**
 Translate System indices into Hamiltonian indices
 */
class IndexTranslator {
public:
    /// A translator is specific to a system and hopping family -- the matrix dimensions
    IndexTranslator(System const& system, MatrixXcd const& hopping_matrix)
        : term_size(hopping_matrix.rows(), hopping_matrix.cols()),
          sys_start(system.compressed_sublattices.start_index(term_size.row),
                    system.compressed_sublattices.start_index(term_size.col)),
          ham_start(system.to_hamiltonian_indices(sys_start.row)[0],
                    system.to_hamiltonian_indices(sys_start.col)[0]) {}

    /// Loop over all Hamiltonian indices matching the System indices given in `coordinates`:
    ///     lambda(idx_t row, idx_t col, scalar_t value)
    /// where `row` and `col` are the Hamiltonian indices.
    template<class C, class F, class V, class scalar_t = typename V::Scalar> CPB_ALWAYS_INLINE
    void for_each(C const& coordinates, V const& hopping_buffer, F lambda) const {
        auto const* data = hopping_buffer.data();
        for (auto i = ham_start.row; i < ham_start.row + term_size.row; ++i) {
            for (auto j = ham_start.col; j < ham_start.col + term_size.col; ++j) {
                for (auto const& coo : coordinates) {
                    auto const ham_row = i + (coo.row - sys_start.row) * term_size.row;
                    auto const ham_col = j + (coo.col - sys_start.col) * term_size.col;

                    auto const value = *data++;
                    if (value != scalar_t{0}) {
                        lambda(ham_row, ham_col, value);
                    }
                }
            }
        }
    }

private:
    COO term_size; ///< size the hopping matrix (single hopping term)
    COO sys_start; ///< start index in System coordinates
    COO ham_start; ///< start index in Hamiltonian coordinates
};

/**
 Buffer for intermediate hoppings and positions required by hopping modifiers

 Applying modifiers to each hopping individually would be slow.
 Passing all the values in one call would require a lot of memory.
 Buffering the hoppings to balances performance and memory usage.
*/
template<class scalar_t>
struct HoppingBuffer {
    static constexpr auto max_buffer_size = idx_t{100000};

    idx_t size; ///< number of elements in the buffer
    MatrixX<scalar_t> unit_hopping; ///< to be replicated `size` times
    ArrayX<scalar_t> hoppings; ///< actually a 3D array: `size` * `unit.rows()` * `unit.cols()`
    CartesianArray pos1; ///< hopping source position
    CartesianArray pos2; ///< hopping destination position

    HoppingBuffer(MatrixXcd const& unit_hopping, idx_t block_size)
        : size(std::min(max_buffer_size / unit_hopping.size(), block_size)),
          unit_hopping(num::force_cast<scalar_t>(unit_hopping)),
          hoppings(size * unit_hopping.size()),
          pos1(size), pos2(size) {}

    /// Replicate each value from the `unit_hopping` matrix `num` times
    void reset_hoppings(idx_t num) {
        auto start = idx_t{0};
        for (auto const& value : unit_hopping) {
            hoppings.segment(start, num).setConstant(value);
            start += num;
        }
    }

    /// Return an `arrayref` of the first `num` elements (each element is a hopping matrix)
    ComplexArrayRef hoppings_ref(idx_t num) {
        auto const rows = unit_hopping.rows();
        auto const cols = unit_hopping.cols();
        return (rows == 1 && cols == 1) ? arrayref(hoppings.data(), num)
                                        : arrayref(hoppings.data(), rows, cols, num);
    }
};

template<class scalar_t, class SystemOrBoundary, class Fn>
void HamiltonianModifiers::apply_to_hoppings_impl(System const& system,
                                                  SystemOrBoundary const& system_or_boundary,
                                                  Fn lambda) const {
    auto const& hopping_registry = system.hopping_registry;

    // Fast path: Modifiers don't need to be applied and the single-orbital model
    // allows direct mapping between sites and Hamiltonian matrix elements.
    if (hopping.empty() && !hopping_registry.has_multiple_orbitals()) {
        for (auto const& block : system_or_boundary.hopping_blocks) {
            auto const energy = num::force_cast<scalar_t>(
                hopping_registry.energy(block.family_id())
            );

            auto const value = energy(0, 0); // single orbital
            for (auto const& coo : block.coordinates()) {
                lambda(coo.row, coo.col, value);
            }
        }

        return;
    }

    // Slow path: Apply modifiers and/or consider multiple orbitals which
    // require translating between site and Hamiltonian matrix indices.
    for (auto const& block : system_or_boundary.hopping_blocks) {
        if (block.size() == 0) { continue; }

        auto const& hopping_energy = hopping_registry.energy(block.family_id());
        auto const hopping_name = hopping_registry.name(block.family_id());
        auto const index_translator = IndexTranslator(system, hopping_energy);

        auto buffer = HoppingBuffer<scalar_t>(hopping_energy, block.size());
        for (auto const coo_slice : sliced(block.coordinates(), buffer.size)) {
            auto size = idx_t{0};
            for (auto const& coo : coo_slice) {
                buffer.pos1[size] = system.positions[coo.row];
                buffer.pos2[size] = detail::shifted(system.positions[coo.col], system_or_boundary);
                ++size;
            }

            buffer.reset_hoppings(size);
            for (auto const& modifier : hopping) {
                modifier.apply(buffer.hoppings_ref(size), buffer.pos1.head(size),
                               buffer.pos2.head(size), hopping_name);
            }

            index_translator.for_each(coo_slice, buffer.hoppings, lambda);
        }
    }
}

} // namespace cpb
