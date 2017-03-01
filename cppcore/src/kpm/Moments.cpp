#include "kpm/Moments.hpp"

namespace cpb { namespace kpm {

template<class scalar_t>
void DiagonalMoments<scalar_t>::collect_initial(VectorRef r0, VectorRef r1) {
    m0 = moments[0] = r0.squaredNorm() * scalar_t{0.5};
    m1 = moments[1] = r1.dot(r0);
}

template<class scalar_t>
void DiagonalMoments<scalar_t>::collect(idx_t n, VectorRef r0, VectorRef r1) {
    collect(n, r0.squaredNorm(), r1.dot(r0));
}

template<class scalar_t>
void DiagonalMoments<scalar_t>::collect(idx_t n, scalar_t a, scalar_t b) {
    assert(n >= 2 && n <= size() / 2);
    moments[2 * (n - 1)] = scalar_t{2} * (a - m0);
    moments[2 * (n - 1) + 1] = scalar_t{2} * b - m1;
}

template<class scalar_t>
void OffDiagonalMoments<scalar_t>::collect_initial(VectorRef r0, VectorRef r1) {
    using real_t = num::get_real_t<scalar_t>;

    for (auto i = 0; i < idx.cols.size(); ++i) {
        auto const col = idx.cols[i];
        data[i][0] = r0[col] * real_t{0.5}; // 0.5 is special for the moment zero
        data[i][1] = r1[col];
    }
}

template<class scalar_t>
void OffDiagonalMoments<scalar_t>::collect(idx_t n, VectorRef r1) {
    assert(n >= 2 && n < data[0].size());
    for (auto i = 0; i < idx.cols.size(); ++i) {
        auto const col = idx.cols[i];
        data[i][n] = r1[col];
    }
}

CPB_INSTANTIATE_TEMPLATE_CLASS(DiagonalMoments)
CPB_INSTANTIATE_TEMPLATE_CLASS(OffDiagonalMoments)

}} // namespace cpb::kpm
