#include "kpm/default/collectors.hpp"

namespace cpb { namespace kpm {

template<class scalar_t>
void DiagonalCollector<scalar_t>::initial(VectorRef r0, VectorRef r1) {
    m0 = moments[0] = r0.squaredNorm() * scalar_t{0.5};
    m1 = moments[1] = r1.dot(r0);
}

template<class scalar_t>
void DiagonalCollector<scalar_t>::operator()(idx_t n, scalar_t m2, scalar_t m3) {
    moments[2 * (n - 1)] = scalar_t{2} * (m2 - m0);
    moments[2 * (n - 1) + 1] = scalar_t{2} * m3 - m1;
}

template<class scalar_t>
void BatchDiagonalCollector<scalar_t>::initial(VectorRef r0, VectorRef r1) {
    auto const size = m0.size();
    for (auto i = size_t{0}; i < size; ++i) {
        moments(0, i) = m0[i] = r0.col(i).squaredNorm() * scalar_t{0.5};
        moments(1, i) = m1[i] = r1.col(i).dot(r0.col(i));
    }
}

template<class scalar_t>
void BatchDiagonalCollector<scalar_t>::operator()(idx_t n, simd::array<scalar_t> m2,
                                                  simd::array<scalar_t> m3) {
    auto const size = m0.size();
    for (auto i = size_t{0}; i < size; ++i) {
        moments(2 * (n - 1), i) = scalar_t{2} * (m2[i] - m0[i]);
        moments(2 * (n - 1) + 1, i) = scalar_t{2} * m3[i] - m1[i];
    }
}

template<class scalar_t>
GenericCollector<scalar_t>::GenericCollector(idx_t num_moments, OptimizedHamiltonian const& oh,
                                             VectorXcd const& alpha_, VectorXcd const& beta_,
                                             SparseMatrixXcd const& op_) : moments(num_moments) {
    beta = num::force_cast<scalar_t>(beta_.size() != 0 ? beta_ : alpha_);
    oh.reorder(beta);
    if (op_.size() != 0){
        op = num::force_cast<scalar_t>(op_);
        oh.reorder(op);
    }
}

template<class scalar_t>
void GenericCollector<scalar_t>::initial(VectorRef r0, VectorRef r1) {
    moments[0] = (op.size() != 0) ? beta.dot(op * r0) : beta.dot(r0);
    moments[0] *= 0.5f;
    moments[1] = (op.size() != 0) ? beta.dot(op * r1) : beta.dot(r1);
}

template<class scalar_t>
void GenericCollector<scalar_t>::operator()(idx_t n, VectorRef r1) {
    moments[n] = (op.size() != 0) ? beta.dot(op * r1) : beta.dot(r1);
}

template<class scalar_t>
void MultiUnitCollector<scalar_t>::initial(VectorRef r0, VectorRef r1) {
    using real_t = num::get_real_t<scalar_t>;

    for (auto i = 0; i < idx.dest.size(); ++i) {
        moments[i][0] = r0[idx.dest[i]] * real_t{0.5}; // 0.5 is special the moment zero
        moments[i][1] = r1[idx.dest[i]];
    }
}

template<class scalar_t>
void MultiUnitCollector<scalar_t>::operator()(idx_t n, VectorRef r1) {
    for (auto i = 0; i < idx.dest.size(); ++i) {
        moments[i][n] = r1[idx.dest[i]];
    }
}

template<class scalar_t>
DenseMatrixCollector<scalar_t>::DenseMatrixCollector(
    idx_t num_moments, OptimizedHamiltonian const& oh, VariantCSR const& op_
) : moments(num_moments, oh.size()) {
    if (op_) {
        op = op_.template get<scalar_t>();
        oh.reorder(op);
    }
}

template<class scalar_t>
void DenseMatrixCollector<scalar_t>::initial(VectorRef r0, VectorRef r1) {
    using real_t = num::get_real_t<scalar_t>;
    if (op.size() != 0){
        moments.row(0) = op * r0 * real_t{0.5}; // 0.5 is special for the moment zero
        moments.row(1) = op * r1;
    } else {
        moments.row(0) = r0 * real_t{0.5}; // 0.5 is special for the moment zero
        moments.row(1) = r1;
    }
}

template<class scalar_t>
void DenseMatrixCollector<scalar_t>::operator()(idx_t n, VectorRef r1) {
    if (op.size() != 0) {
        moments.row(n) = op * r1;
    } else {
        moments.row(n) = r1;
    }
}

CPB_INSTANTIATE_TEMPLATE_CLASS(DiagonalCollector)
CPB_INSTANTIATE_TEMPLATE_CLASS(BatchDiagonalCollector)
CPB_INSTANTIATE_TEMPLATE_CLASS(GenericCollector)
CPB_INSTANTIATE_TEMPLATE_CLASS(MultiUnitCollector)
CPB_INSTANTIATE_TEMPLATE_CLASS(DenseMatrixCollector)

}} // namespace cpb::kpm
