#include "hamiltonian/Hamiltonian.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"

#include "utils/Log.hpp"
#include "numeric/constant.hpp"

using namespace tbm;
using constant::i1;

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

    auto const has_onsite_energy = system.lattice.has_onsite_energy || !modifiers.onsite.empty();
    if (system.has_unbalanced_hoppings) {
        matrix.reserve(nonzeros_per_row(system.hoppings, has_onsite_energy));
    } else {
        auto const num_per_row = system.lattice.max_hoppings() + has_onsite_energy;
        matrix.reserve(ArrayXi::Constant(num_sites, num_per_row));
    }

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
    
    for (auto n = 0; n < num_boundaries; ++n) {
        auto& b_matrix = boundary_matrices[n];
        boundary_lengths[n] = system.boundaries[n].shift;
        
        // set the size of the matrix
        auto const num_sites = system.num_sites();
        b_matrix.resize(num_sites, num_sites);
        b_matrix.reserve(VectorXi::Constant(num_sites, system.lattice.max_hoppings()));

        modifiers.apply_to_hoppings<scalar_t>(system, n, [&](int i, int j, scalar_t hopping) {
            b_matrix.insert(i, j) = hopping;
        });

        b_matrix.makeCompressed();
        throw_if_invalid(b_matrix);
    }
}

template<class scalar_t, class... Args>
std14::enable_if_t<!num::is_complex<scalar_t>(), void>
set_helper(Args...) {
    // pass - a real Hamiltonian can't have periodic boundary conditions
}

template<class scalar_t, class SparseMatrix, class M, class L>
std14::enable_if_t<num::is_complex<scalar_t>(), void>
set_helper(SparseMatrix& matrix, const M& boundary_matrices, const L& lengths, Cartesian k) {
    // sum boundary matrices in all periodic directions
    for (std::size_t i = 0; i < boundary_matrices.size(); ++i) {
        auto const phase = static_cast<scalar_t>(exp(i1 * k.dot(lengths[i])));
        auto b_matrix1 = boundary_matrices[i] * phase;
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
template class tbm::HamiltonianT<double>;
template class tbm::HamiltonianT<std::complex<double>>;
