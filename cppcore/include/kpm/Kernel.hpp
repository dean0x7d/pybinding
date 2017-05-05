#pragma once
#include "numeric/dense.hpp"

namespace cpb { namespace kpm {

/// Moment calculations at higher optimization levels require specific rounding.
/// `n - 2` considers only moments in the main KPM loop. Divisible by 4 because
/// that is the strictest requirement imposed by `opt_size_and_interleaved`.
inline idx_t round_num_moments(idx_t n) {
    if (n < 2) { return 2; }
    while ((n - 2) % 4 != 0) { ++n; }
    return n;
}

/**
 Put the kernel in *Kernel* Polynomial Method

 This provides the general kernel interface. For concrete implementations
 see the `lorentz_kernel` and `jackson_kernel` functions below.
 */
struct Kernel {
    /// Produce the KPM damping coefficients which depend on the number of expansion moments
    std::function<ArrayXd(idx_t num_moments)> damping_coefficients;
    /// The number of moments required to reconstruct a function at the specified scaled broadening
    std::function<idx_t(double scaled_broadening)> required_num_moments;

    /// Apply the kernel damping to an array of moments
    template<class scalar_t>
    void operator()(ArrayX<scalar_t>& moments) const {
        using real_t = num::get_real_t<scalar_t>;
        auto const N = static_cast<int>(moments.size());
        moments *= damping_coefficients(N).template cast<real_t>();
    }

    template<class scalar_t>
    void operator()(std::vector<ArrayX<scalar_t>>& moments) const {
        for (auto& m : moments) { operator()(m); }
    }

    template<class scalar_t>
    void operator()(ArrayXX<scalar_t>& moments) const {
        using real_t = num::get_real_t<scalar_t>;
        auto const N = static_cast<int>(moments.rows());
        moments.colwise() *= damping_coefficients(N).template cast<real_t>().eval();
    }

    /// Apply the kernel damping to a matrix of moments
    template<class scalar_t>
    void operator()(MatrixX<scalar_t>& moments) const {
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

/**
 The Dirichlet kernel

 This kernel doesn't modify the moments at all. The resulting moments represent just
 a truncated series which results in lots of oscillation in the reconstructed function.
 Therefore, this kernel should almost never be used. It's only here in case the raw
 moment values are needed for some other purpose. Note that `required_num_moments()`
 returns `N = pi / sigma` for compatibility with the Jackson kernel, but there is no
 actual broadening associated with the Dirichlet kernel.
 */
Kernel dirichlet_kernel();

}} // namespace cpb::kpm
