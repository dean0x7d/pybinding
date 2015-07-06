#include "hamiltonian/Hamiltonian.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"
#include "system/System.hpp"

#include "utils/Log.hpp"
#include "utils/Chrono.hpp"
#include "support/cpp14.hpp"
#include "support/physics.hpp"
#include "support/format.hpp"

using namespace tbm;
using physics::i1;

template<typename scalar_t>
HamiltonianT<scalar_t>::~HamiltonianT()
{
    Log::d("~Hamiltonian<" + num::scalar_name<scalar_t>() + ">()");
}

template<typename scalar_t>
HamiltonianT<scalar_t>::HamiltonianT(System const& system, HamiltonianModifiers const& modifiers,
                                     Cartesian k_vector) {
    auto build_time = Chrono{};
    build_main(system, modifiers);
    build_periodic(system, modifiers);
    set(k_vector);
    report = fmt::format("The Hamiltonian has {} non-zero values, {}",
                         fmt::with_suffix(non_zeros()), build_time.toc());
}

template<typename scalar_t>
void HamiltonianT<scalar_t>::build_main(System const& system,
                                        HamiltonianModifiers const& modifiers) {
    auto const num_sites = system.num_sites();
    auto const& lattice = system.lattice;

    matrix.resize(num_sites, num_sites);
    // number of hoppings plus 1 (optional) for the on-site potential
    auto const non_zeros_per_row = lattice.max_hoppings()
                                   + (lattice.has_onsite_potential || !modifiers.onsite.empty());
    matrix.reserve(VectorXi::Constant(num_sites, non_zeros_per_row));
    
    // insert onsite potential terms
    auto potential = ArrayX<scalar_t>{};
    if (lattice.has_onsite_potential) {
        potential.resize(num_sites);
        for (int i = 0; i < num_sites; ++i) {
            potential[i] = static_cast<scalar_t>(lattice[system.sublattices[i]].onsite);
        }
    }

    if (!modifiers.onsite.empty()) {
        if (potential.size() == 0)
            potential.setZero(num_sites);

        for (const auto& onsite_modifier : modifiers.onsite)
            onsite_modifier->apply(potential, system.positions);
        
        for (int i = 0; i < num_sites; ++i) {
            if (potential[i] != scalar_t{0}) // conserve space in the sparse matrix
                matrix.insert(i, i) = potential[i];
        }
    }
    
    // insert hopping terms
    modifiers.apply_to_hoppings<scalar_t>(system, [&](int i, int j, scalar_t hopping) {
        matrix.insert(i, j) = hopping;
        matrix.insert(j, i) = num::conjugate(hopping);
    });

    // remove any extra reserved space from the sparse matrices
    matrix.makeCompressed();
}

template<typename scalar_t>
void HamiltonianT<scalar_t>::build_periodic(System const& system,
                                            HamiltonianModifiers const& modifiers) {
    auto const num_boundaries = static_cast<int>(system.boundaries.size());
    boundary_matrices.resize(num_boundaries);
    boundary_lengths.resize(num_boundaries);
    
    for (int p = 0; p < num_boundaries; ++p) {
        auto const& boundary = system.boundaries[p];
        auto& b_matrix = boundary_matrices[p];
        boundary_lengths[p] = boundary.shift;
        
        // set the size of the matrix
        auto const num_sites = system.num_sites();
        b_matrix.resize(num_sites, num_sites);
        b_matrix.reserve(VectorXi::Constant(num_sites, system.lattice.max_hoppings()));

        modifiers.apply_to_hoppings<scalar_t>(boundary, [&](int i, int j, scalar_t hopping) {
            b_matrix.insert(i, j) = hopping;
        });
        
        b_matrix.makeCompressed();
    }
}

template<typename scalar_t, class... Args>
cpp14::enable_if_t<!num::is_complex<scalar_t>(), void>
set_helper(Args...)
{
    // pass - a real Hamiltonian can't have periodic boundary conditions
}

template<typename scalar_t, class SparseMatrix, class M, class L>
cpp14::enable_if_t<num::is_complex<scalar_t>(), void>
set_helper(SparseMatrix& matrix, const M& boundary_matrices, const L& lengths, Cartesian k)
{
    // sum boundary matrices in all periodic directions
    for (std::size_t i = 0; i < boundary_matrices.size(); ++i) {
        auto b_matrix1 = boundary_matrices[i] * exp(i1 * k.dot(lengths[i]));
        auto b_matrix2 = static_cast<SparseMatrix>(b_matrix1.adjoint());
        matrix += b_matrix1 + b_matrix2;
    }
}

template<typename scalar_t>
void HamiltonianT<scalar_t>::set(Cartesian k_vector)
{
    set_helper<scalar_t>(matrix, boundary_matrices, boundary_lengths, k_vector);
}


template class tbm::HamiltonianT<float>;
template class tbm::HamiltonianT<std::complex<float>>;
//template class tbm::HamiltonianT<double>;
//template class tbm::HamiltonianT<std::complex<double>>;
