#pragma once
#include "hamiltonian/HamiltonianModifiers.hpp"

#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"
#include "numeric/traits.hpp"
#include "numeric/constant.hpp"

#include "support/variant.hpp"

namespace cpb {

template<class scalar_t>
using SparseMatrixRC = std::shared_ptr<SparseMatrixX<scalar_t> const>;

/**
 Stores a tight-binding Hamiltonian as a sparse matrix variant
 with real or complex scalar type and single or double precision.
 */
class Hamiltonian {
    using Variant = var::variant<SparseMatrixRC<float>, SparseMatrixRC<std::complex<float>>,
                                 SparseMatrixRC<double>, SparseMatrixRC<std::complex<double>>>;
    Variant variant_matrix;

public:
    Hamiltonian() = default;
    template<class scalar_t>
    Hamiltonian(std::shared_ptr<SparseMatrixX<scalar_t>> p) : variant_matrix(std::move(p)) {}
    template<class scalar_t>
    Hamiltonian(std::shared_ptr<SparseMatrixX<scalar_t> const> p) : variant_matrix(std::move(p)) {}

    Variant const& get_variant() const { return variant_matrix; }

    explicit operator bool() const;
    void reset();

    ComplexCsrConstRef csrref() const;
    int non_zeros() const;
    int rows() const;
    int cols() const;
};

namespace detail {

template<class scalar_t>
void build_main(SparseMatrixX<scalar_t>& matrix, System const& system,
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
}

template<class scalar_t>
void build_periodic(SparseMatrixX<scalar_t>& matrix, System const& system,
                    HamiltonianModifiers const& modifiers, Cartesian k_vector) {
    for (auto n = size_t{0}, size = system.boundaries.size(); n < size; ++n) {
        using constant::i1;
        auto const& d = system.boundaries[n].shift;
        auto const phase = num::complex_cast<scalar_t>(exp(i1 * k_vector.dot(d)));

        modifiers.apply_to_hoppings<scalar_t>(system, n, [&](int i, int j, scalar_t hopping) {
            matrix.coeffRef(i, j) += hopping * phase;
            matrix.coeffRef(j, i) += num::conjugate(hopping * phase);
        });
    }
}

/// Check that all the values in the matrix are finite
template<class scalar_t>
void throw_if_invalid(SparseMatrixX<scalar_t> const& m) {
    auto const data = Eigen::Map<ArrayX<scalar_t> const>(m.valuePtr(), m.nonZeros());
    if (!data.allFinite()) {
        throw std::runtime_error("The Hamiltonian contains invalid values: NaN or INF.\n"
                                 "Check the lattice and/or modifier functions.");
    }
}

} // namespace detail

namespace ham {

template<class scalar_t>
inline SparseMatrixRC<scalar_t> get_shared_ptr(Hamiltonian const& h) {
    return var::get<SparseMatrixRC<scalar_t>>(h.get_variant());
}

template<class scalar_t>
inline SparseMatrixX<scalar_t> const& get_reference(Hamiltonian const& h) {
    return *var::get<SparseMatrixRC<scalar_t>>(h.get_variant());
}

template<class scalar_t>
inline bool is(Hamiltonian const& h) {
    return h.get_variant().template is<SparseMatrixRC<scalar_t>>();
}

template<class scalar_t>
Hamiltonian make(System const& system, HamiltonianModifiers const& modifiers, Cartesian k_vector) {
    auto matrix = std::make_shared<SparseMatrixX<scalar_t>>();

    detail::build_main(*matrix, system, modifiers);
    detail::build_periodic(*matrix, system, modifiers, k_vector);

    matrix->makeCompressed();
    detail::throw_if_invalid(*matrix);

    return matrix;
}

} // namespace ham
} // namespace cpb
