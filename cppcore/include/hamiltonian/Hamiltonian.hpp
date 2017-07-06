#pragma once
#include "hamiltonian/HamiltonianModifiers.hpp"

#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"
#include "numeric/traits.hpp"
#include "numeric/constant.hpp"

#include "support/variant.hpp"

namespace cpb {

/**
 Stores a CSR matrix variant of various scalar types: real or complex, single or double precision.
 The internal storage is reference counted which makes instances of this class relatively cheap
 to copy. The matrix itself is immutable (for safety with the reference counting).
 */
class VariantCSR {
    using Variant = var::complex<SparseMatrixX>;
    std::shared_ptr<Variant const> ptr;

public:
    VariantCSR() = default;
    template<class scalar_t>
    VariantCSR(SparseMatrixX<scalar_t> const& m) : ptr(std::make_shared<Variant>(m)) {}
    template<class scalar_t>
    VariantCSR(SparseMatrixX<scalar_t>&& m) : ptr(std::make_shared<Variant>(m.markAsRValue())) {}

    explicit operator bool() const { return static_cast<bool>(ptr); }
    void reset() { ptr.reset(); }

    template<class scalar_t>
    auto get() const -> decltype(ptr->template get<SparseMatrixX<scalar_t>>()) {
        return ptr->template get<SparseMatrixX<scalar_t>>();
    }

    template<class... Args>
    auto match(Args&&... args) const -> decltype(ptr->match(std::forward<Args>(args)...)) {
        return ptr->match(std::forward<Args>(args)...);
    }
};

template<class scalar_t>
using SparseMatrixRC = std::shared_ptr<SparseMatrixX<scalar_t> const>;

/**
 Stores a tight-binding Hamiltonian as a sparse matrix variant
 with real or complex scalar type and single or double precision.
 */
class Hamiltonian {
public:
    Hamiltonian() = default;
    template<class scalar_t>
    Hamiltonian(std::shared_ptr<SparseMatrixX<scalar_t>> p) : variant_matrix(std::move(p)) {}
    template<class scalar_t>
    Hamiltonian(std::shared_ptr<SparseMatrixX<scalar_t> const> p) : variant_matrix(std::move(p)) {}

    var::complex<SparseMatrixRC> const& get_variant() const { return variant_matrix; }

    explicit operator bool() const;
    void reset();

    ComplexCsrConstRef csrref() const;
    idx_t non_zeros() const;
    idx_t rows() const;
    idx_t cols() const;

private:
    var::complex<SparseMatrixRC> variant_matrix;
};

namespace detail {

template<class scalar_t>
void build_main(SparseMatrixX<scalar_t>& matrix, System const& system,
                HamiltonianModifiers const& modifiers, bool simple_build) {
    auto const size = system.hamiltonian_size();
    matrix.resize(size, size);

    if (simple_build) {
        // Fast path: No generators were used (only unit cell replication + modifiers)
        // so we can easily predict the maximum number of non-zero values per row.
        auto const has_diagonal = system.lattice.has_diagonal_terms() || !modifiers.onsite.empty();
        auto const num_per_row = system.lattice.max_hoppings() + has_diagonal;
        matrix.reserve(ArrayXi::Constant(size, num_per_row));

        modifiers.apply_to_onsite<scalar_t>(system, [&](idx_t i, idx_t j, scalar_t onsite) {
            matrix.insert(i, j) = onsite;
        });

        modifiers.apply_to_hoppings<scalar_t>(system, [&](idx_t i, idx_t j, scalar_t hopping) {
            matrix.insert(i, j) = hopping;
            matrix.insert(j, i) = num::conjugate(hopping);
        });
    } else {
        // Slow path: Users can do anything with generators which makes the number of non-zeros
        // per row difficult to count (possible but not worth it over building from triplets).
        auto triplets = std::vector<Eigen::Triplet<scalar_t>>();
        triplets.reserve(system.hamiltonian_nnz());

        // Helper lambda which does a `idx_t` -> `storage_idx_t` cast for the indices.
        auto to_triplet = [](idx_t i, idx_t j, scalar_t value) -> Eigen::Triplet<scalar_t> {
            return {static_cast<storage_idx_t>(i), static_cast<storage_idx_t>(j), value};
        };

        modifiers.apply_to_onsite<scalar_t>(system, [&](idx_t i, idx_t j, scalar_t onsite) {
            triplets.push_back(to_triplet(i, j, onsite));
        });

        modifiers.apply_to_hoppings<scalar_t>(system, [&](idx_t i, idx_t j, scalar_t hopping) {
            triplets.push_back(to_triplet(i, j, hopping));
            triplets.push_back(to_triplet(j, i, num::conjugate(hopping)));
        });

        matrix.setFromTriplets(triplets.begin(), triplets.end());
    }
}

template<class scalar_t>
void build_periodic(SparseMatrixX<scalar_t>& matrix, System const& system,
                    HamiltonianModifiers const& modifiers, Cartesian k_vector) {
    for (auto n = size_t{0}, size = system.boundaries.size(); n < size; ++n) {
        using constant::i1;
        auto const& d = system.boundaries[n].shift;
        auto const phase = num::force_cast<scalar_t>(exp(i1 * k_vector.dot(d)));

        modifiers.apply_to_hoppings<scalar_t>(system, n, [&](idx_t i, idx_t j, scalar_t hopping) {
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
Hamiltonian make(System const& system, HamiltonianModifiers const& modifiers,
                 Cartesian k_vector, bool simple_build) {
    auto matrix = std::make_shared<SparseMatrixX<scalar_t>>();

    detail::build_main(*matrix, system, modifiers, simple_build);
    detail::build_periodic(*matrix, system, modifiers, k_vector);

    matrix->makeCompressed();
    detail::throw_if_invalid(*matrix);

    return matrix;
}

} // namespace ham
} // namespace cpb
