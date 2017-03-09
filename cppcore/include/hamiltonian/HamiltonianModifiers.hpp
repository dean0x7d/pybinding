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
    using Function = std::function<void(ComplexArrayRef energy, CartesianArrayConstRef pos1,
                                        CartesianArrayConstRef pos2, HopIdRef hopping_ids)>;
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
    auto const& lattice = system.lattice;
    auto const has_intrinsic_onsite = lattice.has_onsite_energy();
    if (!has_intrinsic_onsite && onsite.empty()) {
        return;
    }

    for (auto const& sub : system.compressed_sublattices) {
        auto onsite_energy = ArrayX<scalar_t>::Zero(sub.ham_size() * sub.num_orbitals()).eval();

        if (has_intrinsic_onsite) {
            // Intrinsic lattice onsite energy -- just replicate the value at each site
            auto const energy = lattice[sub.alias_id()].energy_as<scalar_t>();
            auto const flat = Eigen::Map<ArrayX<scalar_t> const>(energy.data(), energy.size());
            onsite_energy = flat.replicate(sub.num_sites(), 1);
        }

        if (!onsite.empty()) {
            // Apply all user-defined onsite modifier functions
            auto onsite_ref = arrayref(onsite_energy.data(), sub.num_sites(),
                                       sub.num_orbitals(), sub.num_orbitals());
            if (sub.num_orbitals() == 1) { onsite_ref.ndim = 1; } // squeeze dimensions

            auto const position_ref = system.positions.segment(sub.sys_start(), sub.num_sites());
            auto const sub_ids = ArrayX<sub_id>::Constant(sub.num_sites(), sub.alias_id());

            for (auto const& modifier : onsite) {
                modifier.apply(onsite_ref, position_ref, {sub_ids, lattice.sub_name_map()});
            }
        }

        auto const* data = onsite_energy.data();
        if (sub.num_orbitals() == 1) {
            // Fast path: `onsite_energy` corresponds directly with the diagonal of the Hamiltonian
            for (auto idx = sub.ham_start(), end = sub.ham_end(); idx < end; ++idx) {
                auto const value = *data++;
                if (value != scalar_t{0}) { lambda(idx, idx, value); }
            }
        } else {
            // Slow path: `onsite_energy` holds matrices: (norb x norb)
            auto const norb = sub.num_orbitals();

            for (auto start = sub.ham_start(), end = sub.ham_end(); start < end; start += norb) {
                for (auto row = start; row < start + norb; ++row) {
                    for (auto col = start; col < start + norb; ++col) {
                        auto const value = *data++;
                        if (value != scalar_t{0}) { lambda(row, col, value); }
                    }
                }
            }
        } // if (sub.num_orbitals() == 1)
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
          ham_start(system.to_hamiltonian_index(sys_start.row),
                    system.to_hamiltonian_index(sys_start.col)) {}

    /// Loop over all Hamiltonian indices matching the System indices given in `coo`:
    ///     lambda(idx_t row, idx_t col, scalar_t value)
    /// where `row` and `col` are the Hamiltonian indices and `value` is an element
    /// of the given `matrix` argument.
    template<class Matrix, class F> CPB_ALWAYS_INLINE
    void for_each(COO coo, Matrix const& matrix, F lambda) const {
        auto const ham_row = ham_start.row + (coo.row - sys_start.row) * term_size.row;
        auto const ham_col = ham_start.col + (coo.col - sys_start.col) * term_size.col;

        for (auto i = 0; i < term_size.row; ++i) {
            for (auto j = 0; j < term_size.col; ++j) {
                lambda(ham_row + i, ham_col + j, matrix(i, j));
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

    HoppingBuffer(MatrixX<scalar_t> const& unit_hopping, idx_t block_size)
        : size(std::min(max_buffer_size / unit_hopping.size(), block_size)),
          unit_hopping(unit_hopping), pos1(size), pos2(size) { reset_hoppings(); }

    /// Replicate the `unit_hopping` matrix `size` times
    void reset_hoppings() {
        using FlatMap = Eigen::Map<ArrayX<scalar_t> const>; // map 2D matrix to 1D array
        auto const flat_energy = FlatMap(unit_hopping.data(), unit_hopping.size());
        hoppings = ArrayX<scalar_t>(flat_energy.replicate(size, 1));
    }

    /// Return an `arrayref` of the first `num` elements (each element is a hopping matrix)
    ComplexArrayRef hoppings_ref(idx_t num) {
        auto ref = arrayref(hoppings.data(), num, unit_hopping.rows(), unit_hopping.cols());
        if (unit_hopping.rows() == 1 && unit_hopping.cols() == 1) { ref.ndim = 1; } // squeeze
        return ref;
    }

    /// Return a single hopping matrix from hoppings
    Eigen::Map<MatrixX<scalar_t> const> hopping_matrix(idx_t n) {
        return {hoppings.data() + n * unit_hopping.size(),
                unit_hopping.rows(), unit_hopping.cols()};
    }
};

template<class scalar_t, class SystemOrBoundary, class Fn>
void HamiltonianModifiers::apply_to_hoppings_impl(System const& system,
                                                  SystemOrBoundary const& system_or_boundary,
                                                  Fn lambda) const {
    if (hopping.empty()) {
        // Fast path: modifiers don't need to be applied
        for (auto const& block : system_or_boundary.hopping_blocks) {
            auto const& hopping_family = system.lattice(static_cast<hop_id>(block.family_id()));
            auto const index_translator = IndexTranslator(system, hopping_family.energy);
            auto const energy = hopping_family.energy_as<scalar_t>();

            for (auto const& coo : block.coordinates()) {
                index_translator.for_each(coo, energy, [&](idx_t row, idx_t col, scalar_t value) {
                    if (value != scalar_t{0}) {
                        lambda(row, col, value);
                    }
                });
            }
        }

        return;
    }

    // Slow path: apply modifiers
    for (auto const& block : system_or_boundary.hopping_blocks) {
        auto const family_id = static_cast<hop_id>(block.family_id());
        auto const& hopping_family = system.lattice(family_id);
        auto const index_translator = IndexTranslator(system, hopping_family.energy);

        auto buffer = HoppingBuffer<scalar_t>(hopping_family.energy_as<scalar_t>(), block.size());
        auto hop_ids = ArrayX<hop_id>::Constant(buffer.size, family_id).eval();

        buffered_for_each(
            block.coordinates(),
            buffer.size,
            /*fill buffer*/ [&](COO const& coo, idx_t n) {
                buffer.pos1[n] = system.positions[coo.row];
                buffer.pos2[n] = detail::shifted(system.positions[coo.col], system_or_boundary);
            },
            /*filled*/ [&](idx_t size) {
                if (size < buffer.size) { hop_ids.conservativeResize(size); }

                for (auto const& modifier : hopping) {
                    modifier.apply(buffer.hoppings_ref(size),
                                   buffer.pos1.head(size), buffer.pos2.head(size),
                                   {hop_ids, system.lattice.hop_name_map()});
                }
            },
            /*process buffer*/ [&](COO const& coo, idx_t n) {
                auto const energy = buffer.hopping_matrix(n);
                index_translator.for_each(coo, energy, [&](idx_t row, idx_t col, scalar_t value) {
                    if (value != scalar_t{0}) {
                        lambda(row, col, value);
                    }
                });
            },
            /*processed*/ [&]() { buffer.reset_hoppings(); }
        );
    }
}

} // namespace cpb
