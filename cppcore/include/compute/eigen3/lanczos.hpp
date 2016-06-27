#pragma once
#include "numeric/dense.hpp"
#include <Eigen/Jacobi>

namespace cpb { namespace compute {

namespace detail {
    template<class real_t>
    static void tridiagonal_qr_step(real_t* diag, real_t* subdiag, int start, int end) {
        auto td = (diag[end-1] - diag[end]) * real_t(0.5);
        auto e = subdiag[end-1];
        auto mu = diag[end];
        if (td == 0) {
            mu -= std::abs(e);
        }
        else {
            auto e2 = Eigen::numext::abs2(subdiag[end-1]);
            auto h = Eigen::numext::hypot(td, e);
            if (e2 == 0)
                mu -= (e / (td + (td>0 ? 1 : -1))) * (e / h);
            else
                mu -= e2 / (td + (td>0 ? h : -h));
        }

        auto x = diag[start] - mu;
        auto z = subdiag[start];
        for (auto k = start; k < end; ++k) {
            Eigen::JacobiRotation<real_t> rot;
            rot.makeGivens(x, z);

            // do T = G' T G
            auto sdk = rot.s() * diag[k] + rot.c() * subdiag[k];
            auto dkp1 = rot.s() * subdiag[k] + rot.c() * diag[k+1];

            diag[k] = rot.c() * (rot.c() * diag[k] - rot.s() * subdiag[k])
                - rot.s() * (rot.c() * subdiag[k] - rot.s() * diag[k+1]);
            diag[k+1] = rot.s() * sdk + rot.c() * dkp1;
            subdiag[k] = rot.c() * sdk - rot.s() * dkp1;

            if (k > start)
                subdiag[k - 1] = rot.c() * subdiag[k-1] - rot.s() * z;

            x = subdiag[k];
            if (k < end - 1) {
                z = -rot.s() * subdiag[k+1];
                subdiag[k + 1] = rot.c() * subdiag[k+1];
            }
        }
    }
}

template<class Derived, class scalar_t = typename Derived::Scalar>
inline ArrayX<scalar_t> tridiagonal_eigenvalues(const DenseBase<Derived>& alpha,
                                                const DenseBase<Derived>& beta)
{
    ArrayX<scalar_t> eigenvalues = alpha;
    ArrayX<scalar_t> temp = beta;

    auto start = 0;
    auto end = static_cast<int>(eigenvalues.size()) - 1;
    auto iter = 0;
    constexpr auto max_iterations = 30;

    while (end > 0) {
        for (auto i = start; i < end; ++i) {
            auto a = std::abs(temp[i]);
            auto b = std::abs(eigenvalues[i]) + std::abs(eigenvalues[i + 1]);
            // if a is much smaller than b
            if (a < b * std::numeric_limits<scalar_t>::epsilon())
                temp[i] = 0;
        }

        while (end > 0 && temp[end-1] == 0)
            end--;

        if (end <= 0)
            break;

        if (++iter > max_iterations * eigenvalues.size())
            throw std::runtime_error{"Tridiagonal QR error"};

        start = end - 1;
        while (start > 0 && temp[start-1] != 0)
            start--;

        detail::tridiagonal_qr_step(eigenvalues.data(), temp.data(), start, end);
    }

    return eigenvalues;
}

}} // namespace cpb::compute
