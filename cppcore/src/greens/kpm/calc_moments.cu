#include "greens/kpm/calc_moments_cuda.hpp"

#include "numeric/traits.cuh"
#include "support/thrust.hpp"
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

namespace cpb { namespace cuda {

/**
 Return the square of a real or complex number (the result is always real)
 */
struct Square {
    template<class real_t> __host__ __device__
    real_t operator()(real_t a) const { return a * a; }

    template<class real_t> __host__ __device__
    real_t operator()(thr::complex<real_t> a) const {
        return a.real() * a.real() + a.imag() * a.imag();
    }
};

/**
 Squared norm of a vector
 
 Note: squared_norm(a) is the same as dotc(a, a), just faster.
 */
template<class thr_scalar_t, class real_t = num::get_real_t<thr_scalar_t>>
real_t squared_norm(int start, int end, thr::device_vector<thr_scalar_t> const& x) {
    return thr::transform_reduce(x.data() + start, x.data() + end, Square(),
                                 real_t{0}, thr::plus<real_t>());
}

/**
 Return conj(a) * b for real or complex arguments
 */
struct Mulc {
    template<class real_t> __host__ __device__
    real_t operator()(real_t a, real_t b) const { return a * b; }

    template<class real_t> __host__ __device__
    thr::complex<real_t> operator()(thr::complex<real_t> a, thr::complex<real_t> b) const {
        return {a.real() * b.real() + a.imag() * b.imag(),
                a.real() * b.imag() - a.imag() * b.real()};
    }
};

/**
 Complex dot product of two vectors
 */
template<class thr_scalar_t>
thr_scalar_t dotc(int start, int end, thr::device_vector<thr_scalar_t> const& x,
                  thr::device_vector<thr_scalar_t> const& y) {
    return thr::inner_product(x.data() + start, x.data() + end, y.data() + start, 
                              thr_scalar_t{0}, thr::plus<thr_scalar_t>(), Mulc());
}

/**
 ELLPACK matrix stored in GPU memory
 */
template<class scalar_t, class index_t = int,
         class thr_scalar_t = num::get_thrust_t<scalar_t>>
class EllMatrix {
    static constexpr auto align_bytes = 256;

    index_t _rows, _cols, _nnz_per_row, _pitch;
    thr::device_vector<thr_scalar_t> _data;
    thr::device_vector<index_t> _indices;

public:
    using Scalar = scalar_t;
    using ThrustScalar = thr_scalar_t;
    using Index = index_t;
    
    EllMatrix(num::EllConstRef<scalar_t> ref)
        : _rows(ref.rows), _cols(ref.cols), _nnz_per_row(ref.nnz_per_row),
          _pitch(num::aligned_size<thr_scalar_t, align_bytes>(ref.rows)) {        
        _data.resize(nnz());
        _indices.resize(nnz());        
        auto const ref_data = static_cast<thr_scalar_t const*>(ref.void_data);
        for (auto n = 0; n < ref.nnz_per_row; ++n) {
            thr::copy(ref_data + n * ref.pitch, ref_data + (n + 1) * ref.pitch,
                      _data.begin() + n * _pitch);
            thr::copy(ref.indices + n * ref.pitch, ref.indices + (n + 1) * ref.pitch,
                      _indices.begin() + n * _pitch);
        }
    }
    
    index_t rows() const { return _rows; }
    index_t cols() const { return _cols; }
    index_t nnz_per_row() const { return _nnz_per_row; }
    index_t pitch() const { return _pitch; }
    index_t nnz() const { return _pitch * _nnz_per_row; }
    
    thr_scalar_t const* data() const { return thr::raw_pointer_cast(_data.data()); }
    index_t const* indices() const { return thr::raw_pointer_cast(_indices.data()); }
};

/**
 Return the KPM r0 vector with all zeros except for the source index
 */
template<class Matrix, class thr_scalar_t = typename Matrix::ThrustScalar>
thr::device_vector<thr_scalar_t> make_r0(Matrix const& h2, int i) {
    auto r0 = thr::device_vector<thr_scalar_t>(h2.rows(), thr_scalar_t{0});
    r0[i] = thr_scalar_t{1};
    return r0;
}

/**
 Return the KPM r1 vector which is equal to the Hamiltonian matrix column at the source index
 */
template<class scalar_t, class thr_scalar_t = num::get_thrust_t<scalar_t>>
thr::device_vector<thr_scalar_t> make_r1(num::EllConstRef<scalar_t> const& h2, int i) {
    auto r1 = thr::host_vector<thr_scalar_t>(h2.rows, thr_scalar_t{0});
    for (auto n = 0; n < h2.nnz_per_row; ++n) {
        auto const col = h2.indices[i + n * h2.pitch];
        auto const value = h2.data()[i + n * h2.pitch];
        r1[col] = num::conjugate(value) * scalar_t{0.5};
    }
    return r1;
}

/**
 GPU function component of the KPM compute kernel for ELLPACK matrix

 Equivalent to: y = matrix * x - y
 */
template<class index_t, class thr_scalar_t>
__global__ void kpm_kernel_device(index_t num_rows, index_t nnz_per_row, index_t pitch,
                                  thr_scalar_t const* data, index_t const* indices,
                                  thr_scalar_t const* x, thr_scalar_t* y) {
    auto const thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto const grid_size = gridDim.x * blockDim.x;

    for(auto row = thread_id; row < num_rows; row += grid_size) {
        auto sum = -y[row];

        auto idx = row;
        for(auto n = 0; n < nnz_per_row; ++n) {
            auto const col = indices[idx];
            auto const value = data[idx];
            sum += value * x[col];
            
            idx += pitch;
        }
        
        y[row] = sum;
    }
}

/**
 Return the number of blocks needed to compute `size` elements at `block_size`
 */
constexpr int required_num_blocks(int size, int block_size) noexcept {
    return (size + (block_size - 1)) / block_size;
}

/**
 KPM compute kernel for ELLPACK matrix

 Equivalent to: y = matrix * x - y
 */
template<class scalar_t, class thr_scalar_t> CPB_ALWAYS_INLINE
void kpm_kernel(int start, int end, EllMatrix<scalar_t> const& ell,
                thr::device_vector<thr_scalar_t> const& x,
                thr::device_vector<thr_scalar_t>& y) {
    auto const size = end - start;
    if (size <= 0) {
        return;
    }

    static constexpr auto block_size = 256;
    auto const num_blocks = required_num_blocks(size, block_size);
    
    kpm_kernel_device<<<num_blocks, block_size>>>(
        size, ell.nnz_per_row(), ell.pitch(), 
        ell.data() + start,
        ell.indices() + start,
        thr::raw_pointer_cast(x.data()),
        thr::raw_pointer_cast(y.data()) + start
    );
}

template<class scalar_t>
thr::host_vector<scalar_t> 
I<scalar_t>::calc_diag_moments0(num::EllConstRef<scalar_t> ellref, int i, int num_moments) {
    auto const h2 = EllMatrix<scalar_t>(ellref);    
    auto r0 = make_r0(h2, i);
    auto r1 = make_r1(ellref, i);

    using thr_scalar_t = num::get_thrust_t<scalar_t>;
    auto const m0 = static_cast<thr_scalar_t>(r0[i]) * real_t{0.5};
    auto const m1 = static_cast<thr_scalar_t>(r1[i]);

    auto moments = thr::host_vector<scalar_t>(num_moments);
    moments[0] = m0;
    moments[1] = m1;
    
    auto const size = h2.rows();
    assert(num_moments % 2 == 0);
    for (auto n = 2; n <= num_moments / 2; ++n) {
        cuda::kpm_kernel(0, size, h2, r1, r0);
        r1.swap(r0);
        
        moments[2 * (n - 1)] = real_t{2} * (squared_norm(0, size, r0) - m0);       
        moments[2 * (n - 1) + 1] = real_t{2} * dotc(0, size, r0, r1) - m1;
    }
    
    return moments;
}

template<class scalar_t>
thr::host_vector<scalar_t> 
I<scalar_t>::calc_diag_moments1(num::EllConstRef<scalar_t> ellref, int i, int num_moments,
                                kpm::OptimizedSizes const& sizes) {
    auto const h2 = EllMatrix<scalar_t>(ellref);    
    auto r0 = make_r0(h2, i);
    auto r1 = make_r1(ellref, i);

    using thr_scalar_t = num::get_thrust_t<scalar_t>;
    auto const m0 = static_cast<thr_scalar_t>(r0[i]) * real_t{0.5};
    auto const m1 = static_cast<thr_scalar_t>(r1[i]);

    auto moments = thr::host_vector<scalar_t>(num_moments);
    moments[0] = m0;
    moments[1] = m1;

    assert(num_moments % 2 == 0);
    for (auto n = 2; n <= num_moments / 2; ++n) {
        auto const opt_size = sizes.optimal(n, num_moments);
        cuda::kpm_kernel(0, opt_size, h2, r1, r0);
        r1.swap(r0);
        
        moments[2 * (n - 1)] = real_t{2} * (squared_norm(0, opt_size, r0) - m0);       
        moments[2 * (n - 1) + 1] = real_t{2} * dotc(0, opt_size, r0, r1) - m1;
    }

    return moments;
}

CPB_INSTANTIATE_TEMPLATE_CLASS(I)

}} // namespace cpb::cuda
