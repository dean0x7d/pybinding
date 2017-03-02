#pragma once
#include "numeric/dense.hpp"

namespace cpb { namespace kpm {

/**
 Put the kernel in *Kernel* Polynomial Method

 This provides the general kernel interface. For concrete implementations
 see the `lorentz_kernel` and `jackson_kernel` functions below.
 */
struct Kernel {
    /// Produce the KPM damping coefficients which depend on the number of expansion moments
    std::function<ArrayXd(int num_moments)> damping_coefficients;
    /// The number of moments required to reconstruct a function at the specified scaled broadening
    std::function<int(double scaled_broadening)> required_num_moments;

    /// Apply the kernel damping to an array of moments
    template<class scalar_t>
    void apply(ArrayX<scalar_t>& moments) const {
        using real_t = num::get_real_t<scalar_t>;
        auto const N = static_cast<int>(moments.size());
        moments *= damping_coefficients(N).template cast<real_t>();
    }

    /// Apply the kernel damping to a matrix of moments
    template<class scalar_t>
    void apply(MatrixX<scalar_t>& moments) const {
        using real_t = num::get_real_t<scalar_t>;
        assert(moments.rows() == moments.cols());

        auto const N = static_cast<int>(moments.rows());
        auto const g = damping_coefficients(N).template cast<real_t>().eval();
        moments.array() *= g.replicate(1, N).rowwise() * g.transpose();
    }
};

/**
 The Jackson kernel

 This is a good general-purpose kernel, appropriate for most applications. It imposes a
 Gaussian broadening of `sigma = pi / N`. Therefore, the resolution of the reconstructed
 function will improve directly with the number of moments N.
*/
Kernel jackson_kernel();

/**
 The Lorentz kernel

 This kernel is most appropriate for the expansion of the Green’s function because it most
 closely mimics the divergences near the true eigenvalues of the Hamiltonian. The lambda
 value is found empirically to be between 3 and 5, and it may be used to fine-tune the
 smoothness of the convergence. The Lorentzian broadening is given by `lambda / N`.
 */
Kernel lorentz_kernel(double lambda = 4.0);

}} // namespace cpb::kpm
