#include "hamiltonian/Hamiltonian.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"

#include "utils/Log.hpp"
#include "support/cpp14.hpp"
#include "support/physics.hpp"

using namespace tbm;
using physics::i1;

template<class scalar_t>
HamiltonianT<scalar_t>::~HamiltonianT()
{
    Log::d("~Hamiltonian<" + num::scalar_name<scalar_t>() + ">()");
}

template<class scalar_t>
HamiltonianT<scalar_t>::HamiltonianT(System const& system, HamiltonianModifiers const& modifiers,
                                     Cartesian k_vector) {
    build_main(system, modifiers);
    build_periodic(system, modifiers);
    set(k_vector);
}

template<class scalar_t>
void HamiltonianT<scalar_t>::throw_if_invalid(SparseMatrix const& m) {
    Eigen::Map<ArrayX<scalar_t> const> data{m.valuePtr(), m.nonZeros()};
    if (!data.allFinite())
        throw std::runtime_error{"The Hamiltonian contains invalid values: NaN or INF.\n"
                                 "Check the lattice and/or modifier functions."};
}

template<class scalar_t>
void HamiltonianT<scalar_t>::build_main(System const& system,
                                        HamiltonianModifiers const& modifiers) {
    auto const num_sites = system.num_sites();
    matrix.resize(num_sites, num_sites);

    auto const non_zeros_per_row = system.lattice.max_hoppings() +
        (system.lattice.has_onsite_energy || !modifiers.onsite.empty());
    matrix.reserve(VectorXi::Constant(num_sites, non_zeros_per_row));
    
    modifiers.apply_to_onsite<scalar_t>(system, [&](int i, scalar_t onsite) {
        matrix.insert(i, i) = onsite;
    });
    
    modifiers.apply_to_hoppings<scalar_t>(system, [&](int i, int j, scalar_t hopping) {
        matrix.insert(i, j) = hopping;
        matrix.insert(j, i) = num::conjugate(hopping);
    });

    matrix.makeCompressed();
    throw_if_invalid(matrix);
}

template<class scalar_t>
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
        throw_if_invalid(b_matrix);
    }
}

template<class scalar_t, class... Args>
cpp14::enable_if_t<!num::is_complex<scalar_t>(), void>
set_helper(Args...)
{
    // pass - a real Hamiltonian can't have periodic boundary conditions
}

template<class scalar_t, class SparseMatrix, class M, class L>
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

template<class scalar_t>
void HamiltonianT<scalar_t>::set(Cartesian k_vector)
{
    set_helper<scalar_t>(matrix, boundary_matrices, boundary_lengths, k_vector);
}


template class tbm::HamiltonianT<float>;
template class tbm::HamiltonianT<std::complex<float>>;
//template class tbm::HamiltonianT<double>;
//template class tbm::HamiltonianT<std::complex<double>>;
