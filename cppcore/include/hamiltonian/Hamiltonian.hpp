#pragma once
#include "support/dense.hpp"
#include "support/sparse.hpp"
#include "support/traits.hpp"
#include "support/sparseref.hpp"

namespace tbm {

struct System;
class HamiltonianModifiers;

/**
 Builds and stores a tight-binding Hamiltonian. Abstract base.
 */
class Hamiltonian {
public:
    virtual ~Hamiltonian() = default;

    virtual SparseURef matrix_union() const = 0;
    virtual int non_zeros() const = 0;
    virtual int rows() const = 0;
    virtual int cols() const = 0;
};


/// Concrete hamiltonian with a specific scalar type.
template<class scalar_t>
class HamiltonianT : public Hamiltonian {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;
    using SparseMatrix = SparseMatrixX<scalar_t>;

public:
    HamiltonianT(System const&, HamiltonianModifiers const&, Cartesian k_vector);
    virtual ~HamiltonianT() override;

    /// Get a const reference to the matrix.
    const SparseMatrix& get_matrix() const { return matrix; }

    virtual SparseURef matrix_union() const override { return matrix; }
    virtual int non_zeros() const override { return matrix.nonZeros(); }
    virtual int rows() const override { return matrix.rows(); }
    virtual int cols() const override { return matrix.cols(); }

private: // build the Hamiltonian
    void build_main(System const&, HamiltonianModifiers const&);
    void build_periodic(System const&, HamiltonianModifiers const&);
    void set(Cartesian k_vector);

    /// Check that all the values in the matrix are finite
    static void throw_if_invalid(SparseMatrix const& m);

private:
    SparseMatrix matrix; ///< the sparse matrix that holds the data
    std::vector<SparseMatrix> boundary_matrices;
    std::vector<Cartesian> boundary_lengths;
};

extern template class HamiltonianT<float>;
extern template class HamiltonianT<std::complex<float>>;
//extern template class HamiltonianT<double>;
//extern template class HamiltonianT<std::complex<double>>;

} // namespace tbm
