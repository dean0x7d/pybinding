#pragma once
#include "kpm/Kernel.hpp"

namespace cpb { namespace kpm {

/// Sparse matrix format for the optimized Hamiltonian
enum class MatrixFormat { CSR, ELL };

/**
 Algorithm selection, see the corresponding functions in `calc_moments.hpp`
 */
struct AlgorithmConfig {
    bool optimal_size;
    bool interleaved;

    /// Does the Hamiltonian matrix need to be reordered?
    bool reorder() const { return optimal_size || interleaved; }
};

/**
 KPM configuration struct with defaults
 */
struct Config {
    float min_energy = 0.0f; ///< lowest eigenvalue of the Hamiltonian
    float max_energy = 0.0f; ///< highest eigenvalue of the Hamiltonian
    Kernel kernel = jackson_kernel(); ///< produces the damping coefficients

    MatrixFormat matrix_format = MatrixFormat::ELL;
    AlgorithmConfig algorithm = {/*optimal_size*/true, /*interleaved*/true};

    float lanczos_precision = 0.002f; ///< how precise should the min/max energy estimation be
};

}} // namespace cpb::kpm
