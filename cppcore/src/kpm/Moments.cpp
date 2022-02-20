#include "kpm/Moments.hpp"

namespace cpb { namespace kpm {

namespace {

struct BatchAccumulatorImpl {
    BatchData& var_result;
    idx_t idx;
    idx_t num_vectors;
    idx_t& count;

    template<class scalar_t>
    void operator()(ArrayX<scalar_t> const& a) {
        if (count == 0) {
            var_result = a;
        } else {
            auto& result = var_result.template get<ArrayX<scalar_t>>();
            result += a;
        }

        ++count;
        if (count >= num_vectors && num_vectors != 1) {
            auto& result = var_result.template get<ArrayX<scalar_t>>();
            result /= static_cast<num::get_real_t<scalar_t>>(num_vectors);
            count = 0;
        }
    }

    template<class scalar_t>
    void operator()(ArrayXX<scalar_t> const& a) {
        if (count == 0) {
            var_result = ArrayX<scalar_t>::Zero(a.rows()).eval();
        }

        auto const batch_size = a.cols();
        auto const remaining = num_vectors - idx;
        auto cols = remaining > batch_size ? batch_size : remaining;

        auto& result = var_result.template get<ArrayX<scalar_t>>();
        result += a.leftCols(cols).rowwise().sum();

        count += batch_size;
        if (count >= num_vectors && num_vectors != 1) {
            result /= static_cast<num::get_real_t<scalar_t>>(num_vectors);
            count = 0;
        }
    }
};

struct BatchConcatenatorImpl {
    BatchData& var_result;
    idx_t idx;
    idx_t num_vectors;
    idx_t& count;

    template<class scalar_t>
    void operator()(ArrayX<scalar_t> const& a) {
        if (count == 0) {
            var_result = ArrayXX<scalar_t>(a.size(), num_vectors);
        }
        auto& data = var_result.template get<ArrayXX<scalar_t>>();
        data.col(idx) = a;
        ++count;
    }

    template<class scalar_t>
    void operator()(ArrayXX<scalar_t> const& a) {
        if (count == 0) {
            var_result = ArrayXX<scalar_t>(a.rows(), num_vectors);
        }

        auto const batch_size = a.cols();
        auto const remaining_cols = num_vectors - idx;
        auto const cols = remaining_cols > batch_size ? batch_size : remaining_cols;

        auto& data = var_result.template get<ArrayXX<scalar_t>>();
        data.block(0, idx, data.rows(), cols) = a.leftCols(cols);
        count += cols;
    }
};

struct InitMatrix {
    idx_t size;

    template<class scalar_t>
    var::complex<MatrixX> operator()(var::tag<scalar_t>) const {
        return MatrixX<scalar_t>::Zero(size, size).eval();
    }
};

struct MatrixMulAdd {
    var::complex<MatrixX>& result;
    var::complex<MatrixX> const& a;

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

void BatchAccumulator::operator()(BatchData& result, BatchData const& nd, idx_t idx, idx_t nvec) {
    var::apply_visitor(BatchAccumulatorImpl{result, idx, nvec, count}, nd);
}

void BatchConcatenator::operator()(BatchData& result, BatchData const& nd, idx_t idx, idx_t nvec) {
    var::apply_visitor(BatchConcatenatorImpl{result, idx, nvec, count}, nd);
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

        return result;
    }
};

VariantCSR velocity(Hamiltonian const& hamiltonian, ArrayXf const& alpha) {
    return var::apply_visitor(Velocity{alpha}, hamiltonian.get_variant());
}

}} // namespace cpb::kpm
