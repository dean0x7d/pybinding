#include "kpm/Moments.hpp"

namespace cpb { namespace kpm {

namespace {

struct InitMatrix {
    idx_t size;

    template<class scalar_t>
    var::Complex<MatrixX> operator()(var::tag<scalar_t>) const {
        return MatrixX<scalar_t>::Zero(size, size).eval();
    }
};

struct ArrayAdd {
    var::Complex<ArrayX> const& other;

    template<class scalar_t>
    void operator()(ArrayX<scalar_t>& result) const {
        result += other.template get<ArrayX<scalar_t>>();
    }
};

struct MatrixMulAdd {
    var::Complex<MatrixX>& result;
    var::Complex<MatrixX> const& a;

    template<class scalar_t>
    void operator()(MatrixX<scalar_t> const& b) {
        using T = MatrixX<scalar_t>;
        result.template get<T>() += a.template get<T>() * b.adjoint();
    }
};

struct Div {
    idx_t n;

    template<class T, class real_t = num::get_real_t<typename T::Scalar>>
    void operator()(T& x) const { x /= static_cast<real_t>(n); }
};

} // anonymous namespace

void MomentAccumulator::add(var::Complex<ArrayX> const& other) {
    if (_count == 0) {
        data = other;
    } else {
        var::apply_visitor(ArrayAdd{other}, data);
    }

    ++_count;
    if (_count == total && total != 1) {
        var::apply_visitor(Div{total}, data);
    }
}

MomentMultiplication::MomentMultiplication(idx_t num_moments, var::scalar_tag tag)
    : data(var::apply_visitor(InitMatrix{num_moments}, tag)) {}

void  MomentMultiplication::matrix_mul_add(DenseMatrixMoments const& a,
                                           DenseMatrixMoments const& b) {
    var::apply_visitor(MatrixMulAdd{data, a.data}, b.data);
}

void MomentMultiplication::normalize(idx_t total) {
    var::apply_visitor(Div{total}, data);
}

struct Velocity {
    ArrayXf const& alpha;

    template<class scalar_t>
    VariantCSR operator()(SparseMatrixRC<scalar_t> const& ham) const {
        auto result = *ham;
        auto const data = result.valuePtr();
        auto const indices = result.innerIndexPtr();
        auto const indptr = result.outerIndexPtr();

        auto const size = result.rows();
        for (auto row = idx_t{0}; row < size; ++row) {
            for (auto n = indptr[row]; n < indptr[row + 1]; ++n) {
                const auto col = indices[n];
                data[n] *= static_cast<scalar_t>(alpha[row] - alpha[col]);
            }
        }

        return std::move(result);
    }
};

VariantCSR velocity(Hamiltonian const& hamiltonian, ArrayXf const& alpha) {
    return var::apply_visitor(Velocity{alpha}, hamiltonian.get_variant());
}

}} // namespace cpb::kpm
