#pragma once
#include "numeric/dense.hpp"
#include "numeric/constant.hpp"

#include "detail/macros.hpp"

namespace cpb { namespace kpm {

namespace detail {
    /// Put the kernel in *Kernel* polynomial method
    template<class scalar_t, class real_t>
    void apply_lorentz_kernel(ArrayX<scalar_t>& moments, real_t lambda) {
        auto const N = moments.size();

        auto lorentz_kernel = [=](real_t n) { // n is real_t to get proper fp division
            using std::sinh;
            return sinh(lambda * (1 - n / N)) / sinh(lambda);
        };

        for (auto n = 0; n < N; ++n) {
            moments[n] *= lorentz_kernel(static_cast<real_t>(n));
        }
    }

    /// Calculate the final Green's function for `scaled_energy` using the KPM `moments`
    template<class scalar_t, class real_t, class complex_t = num::get_complex_t<scalar_t>>
    ArrayX<complex_t> calculate_greens(ArrayX<real_t> const& scaled_energy,
                                       ArrayX<scalar_t> const& moments) {
        // Note that this integer array has real type values
        auto ns = ArrayX<real_t>(moments.size());
        for (auto n = 0; n < ns.size(); ++n) {
            ns[n] = static_cast<real_t>(n);
        }

        // G = -2*i / sqrt(1 - E^2) * sum( moments * exp(-i*ns*acos(E)) )
        auto greens = ArrayX<complex_t>(scaled_energy.size());
        transform(scaled_energy, greens, [&](real_t E) {
            using std::acos;
            using constant::i1;
            auto const norm = -real_t{2} * complex_t{i1} / sqrt(1 - E*E);
            return norm * sum(moments * exp(-complex_t{i1} * ns * acos(E)));
        });

        return greens;
    }
} // namespace detail

/**
 Stores KPM moments (size `num_moments`) computed for each index (size of `indices`)
 */
template<class scalar_t>
class MomentsMatrix {
    using real_t = num::get_real_t<scalar_t>;

    ArrayXi indices;
    std::vector<ArrayX<scalar_t>> data;

public:
    MomentsMatrix(int num_moments, ArrayXi const& indices)
        : indices(indices), data(indices.size()) {
        for (auto& moments : data) {
            moments.resize(num_moments);
        }
    }

    /// Collect the first 2 moments which are computer outside the main KPM loop
    void collect_initial(VectorX<scalar_t> const& r0, VectorX<scalar_t> const& r1) {
        for (auto i = 0; i < indices.size(); ++i) {
            auto const idx = indices[i];
            data[i][0] = r0[idx] * real_t{0.5}; // 0.5 is special for the moment zero
            data[i][1] = r1[idx];
        }
    }

    /// Collect moment `n` from result vector `r` for each index. Expects `n >= 2`.
    void collect(int n, VectorX<scalar_t> const& r) {
        assert(n >= 2 && n < data[0].size());
        for (auto i = 0; i < indices.size(); ++i) {
            auto const idx = indices[i];
            data[i][n] = r[idx];
        }
    }

    /// Put the kernel in *Kernel* polynomial method
    void apply_lorentz_kernel(real_t lambda) {
        for (auto& moments : data) {
            detail::apply_lorentz_kernel(moments, lambda);
        }
    }

    /// Calculate the final Green's function at all indices for `scaled_energy`
    std::vector<ArrayXcd> calc_greens(ArrayX<real_t> const& scaled_energy) const {
        auto greens = std::vector<ArrayXcd>();
        greens.reserve(indices.size());
        for (auto const& moments : data) {
            auto const g = detail::calculate_greens(scaled_energy, moments);
            greens.push_back(g.template cast<std::complex<double>>());
        }
        return greens;
    }
};

}} // namespace cpb::kpm
