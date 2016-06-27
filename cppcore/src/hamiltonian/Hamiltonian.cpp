#include "hamiltonian/Hamiltonian.hpp"

namespace cpb {
namespace {

struct IsValid {
    template<class scalar_t>
    bool operator()(SparseMatrixRC<scalar_t> const& p) const { return p != nullptr; }
};

struct Reset {
    template<class scalar_t>
    void operator()(SparseMatrixRC<scalar_t>& p) const { p.reset(); }
};

struct GetSparseRef {
    template<class scalar_t>
    ComplexCsrConstRef operator()(SparseMatrixRC<scalar_t> const& m) const { return csrref(*m); }
};

struct NonZeros {
    template<class scalar_t>
    int operator()(SparseMatrixRC<scalar_t> const& m) const { return m->nonZeros(); }
};

struct Rows {
    template<class scalar_t>
    int operator()(SparseMatrixRC<scalar_t> const& m) const { return m->rows(); }
};

struct Cols {
    template<class scalar_t>
    int operator()(SparseMatrixRC<scalar_t> const& m) const { return m->cols(); }
};

} // namespace

Hamiltonian::operator bool() const {
    return var::apply_visitor(IsValid(), variant_matrix);
}

void Hamiltonian::reset() {
    return var::apply_visitor(Reset(), variant_matrix);
}

ComplexCsrConstRef Hamiltonian::csrref() const {
    return var::apply_visitor(GetSparseRef(), variant_matrix);
}

int Hamiltonian::non_zeros() const {
    return var::apply_visitor(NonZeros(), variant_matrix);
}

int Hamiltonian::rows() const {
    return var::apply_visitor(Rows(), variant_matrix);
}

int Hamiltonian::cols() const {
    return var::apply_visitor(Cols(), variant_matrix);
}

} // namespace cpb
