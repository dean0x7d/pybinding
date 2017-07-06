#pragma once
#include "hamiltonian/Hamiltonian.hpp"

namespace cpb { namespace leads {

namespace detail {
    template<class scalar_t>
    Hamiltonian make_h0(System const& lead_system, Lattice const& lattice,
                        HamiltonianModifiers const& modifiers) {
        auto h0 = std::make_shared<SparseMatrixX<scalar_t>>();
        cpb::detail::build_main(*h0, lead_system, lattice, modifiers, /*simple_build*/true);
        h0->makeCompressed();
        cpb::detail::throw_if_invalid(*h0);
        return h0;
    }

    template<class scalar_t>
    Hamiltonian make_h1(System const& system, Lattice const& lattice,
                        HamiltonianModifiers const& modifiers) {
        auto h1 = std::make_shared<SparseMatrixX<scalar_t>>();
        auto& matrix = *h1;

        auto const size = system.hamiltonian_size();
        matrix.resize(size, size);
        matrix.reserve(ArrayXi::Constant(size, lattice.max_hoppings()));

        modifiers.apply_to_hoppings<scalar_t>(system, 0, [&](idx_t i, idx_t j, scalar_t hopping) {
            matrix.insert(i, j) = hopping;
        });

        h1->makeCompressed();
        cpb::detail::throw_if_invalid(*h1);

        return h1;
    }
} // namespace detail

/**
 Pair of Hamiltonians which describe the periodic structure of a lead
 */
struct HamiltonianPair {
    Hamiltonian h0; ///< hoppings within the unit cell
    Hamiltonian h1; ///< hoppings between unit cells

    HamiltonianPair(System const& lead_system, Lattice const& lattice,
                    HamiltonianModifiers const& modifiers, bool is_double, bool is_complex) {
        if (is_double) {
            if (is_complex) {
                h0 = detail::make_h0<std::complex<double>>(lead_system, lattice, modifiers);
                h1 = detail::make_h1<std::complex<double>>(lead_system, lattice, modifiers);
            } else {
                h0 = detail::make_h0<double>(lead_system, lattice, modifiers);
                h1 = detail::make_h1<double>(lead_system, lattice, modifiers);
            }
        } else {
            if (is_complex) {
                h0 = detail::make_h0<std::complex<float>>(lead_system, lattice, modifiers);
                h1 = detail::make_h1<std::complex<float>>(lead_system, lattice, modifiers);
            } else {
                h0 = detail::make_h0<float>(lead_system, lattice, modifiers);
                h1 = detail::make_h1<float>(lead_system, lattice, modifiers);
            }
        }
    }
};

}} // namespace cpb::leads
