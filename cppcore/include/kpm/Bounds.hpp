#pragma once
#include "hamiltonian/Hamiltonian.hpp"
#include "utils/Chrono.hpp"

namespace cpb { namespace kpm {

/**
 The KPM scaling factors `a` and `b`
*/
template<class real_t = double>
struct Scale {
    static constexpr auto tolerance = 0.01f; ///< needed because the energy bounds are not precise

    real_t a = 0;
    real_t b = 0;

    Scale() = default;
    Scale(real_t min_energy, real_t max_energy)
        : a(0.5f * (max_energy - min_energy) * (1 + tolerance)),
          b(0.5f * (max_energy + min_energy)) {
        if (std::abs(b / a) < 0.01f * tolerance) {
            b = 0; // rounding to zero saves space in the sparse matrix
        }
    }
    template<class T>
    Scale(Scale<T> const& other)
        : a(static_cast<real_t>(other.a)), b(static_cast<real_t>(other.b)) {}

    explicit operator bool() { return a != 0; }

    /// Apply the scaling factors to a vector
    ArrayX<real_t> operator()(ArrayX<real_t> const& v) const { return (v - b) / a; }
};

/**
 Min and max eigenvalues of the Hamiltonian

 The bounds can be determined automatically using the Lanczos procedure,
 or set manually by the user. Also computes the KPM scaling factors a and b.
*/
class Bounds {
public:
    Bounds(Hamiltonian const& hamiltonian, double precision_percent)
        : hamiltonian(hamiltonian), precision_percent(precision_percent) {}
    /// Set the energy bounds manually, therefore skipping the Lanczos computation
    Bounds(double min_energy, double max_energy) : min(min_energy), max(max_energy) {}

    double min_energy() { compute_bounds(); return min; }
    double max_energy() { compute_bounds(); return max; }
    /// The KPM scaling factors a and b
    Scale<> scaling_factors() { compute_bounds(); return {min, max}; }

    /// Return an array with `size` linearly spaced values within the bounds
    ArrayXd linspaced(idx_t size) { return ArrayXd::LinSpaced(size, min_energy(), max_energy()); }

    std::string report(bool shortform = false) const;

private:
    /// Compute the scaling factors using the Lanczos procedure
    void compute_bounds();

private:
    double min = .0; ///< the lowest eigenvalue
    double max = .0; ///< the highest eigenvalue
    int lanczos_loops = 0;  ///< number of iterations needed to converge the Lanczos procedure

    Hamiltonian hamiltonian;
    double precision_percent;
    Chrono timer;
};

}} // namespace cpb::kpm
