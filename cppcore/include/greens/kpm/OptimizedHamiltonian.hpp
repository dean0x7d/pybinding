#pragma once
#include "greens/kpm/Bounds.hpp"
#include "greens/kpm/OptimizedSizes.hpp"

#include "numeric/sparse.hpp"
#include "numeric/ellmatrix.hpp"

#include "support/variant.hpp"
#include "utils/Chrono.hpp"
#include "detail/macros.hpp"

namespace cpb { namespace kpm {

/**
 Indices of the Green's function matrix that will be computed

 A single KPM calculation will compute an entire `row` of the Green's matrix,
 however only some column indices are required to be saved, as indicated by `cols`.
 */
struct Indices {
    int row = -1;
    ArrayXi cols;

    Indices() = default;
    Indices(int row, int col) : row(row), cols(1) { cols[0] = col; }
    Indices(int row, ArrayXi const& cols) : row(row), cols(cols) {}
    Indices(int row, std::vector<int> const& cols) : row(row), cols(eigen_cast<ArrayX>(cols)) {}

    /// Indicates a single element on the main diagonal
    bool is_diagonal() const { return cols.size() == 1 && row == cols[0]; }

    friend bool operator==(Indices const& l, Indices const& r) {
        return l.row == r.row && all_of(l.cols == r.cols);
    }
};

/**
 Matrix configuration for `OptimizedHamiltonian`
 */
struct MatrixConfig {
    enum class Reorder { ON, OFF };
    enum class Format { CSR, ELL };

    Reorder reorder;
    Format format;
};

/**
 Stores a scaled Hamiltonian `(H - b)/a` which limits it to (-1, 1) boundaries required for KPM.
 In addition, three optimisations are applied (last two are optional, see `MatrixConfig`):

 1) The matrix is multiplied by 2. This benefits most calculations (e.g. `y = 2*H*x - y`),
    because the 2x multiplication is done only once, but it will need to be divided by 2
    when the original element values are needed (very rarely).

 2) Reorder the elements so that target indices are placed at the start of the matrix.
    This produces the `optimized_sizes` vector which may be used to reduce calculation
    time by skipping sparse matrix-vector multiplication of zero values.

 3) Convert the sparse matrix into the ELLPACK format. The sparse matrix-vector
    multiplication algorithm for this format is much easier to vectorize compared
    to the classic CSR format.
 */
template<class scalar_t>
class OptimizedHamiltonian {
    using real_t = num::get_real_t<scalar_t>;
    using OptMatrix = var::variant<SparseMatrixX<scalar_t>, num::EllMatrix<scalar_t>>;

    OptMatrix optimized_matrix; ///< reordered for faster compute
    Indices optimized_idx; ///< reordered target indices in the optimized matrix
    OptimizedSizes optimized_sizes; ///< optimal matrix sizes for each KPM iteration

    SparseMatrixX<scalar_t> const* original_matrix;
    Indices original_idx; ///< original target indices for which the optimization was done

    MatrixConfig config;
    Chrono timer;

public:
    OptimizedHamiltonian(SparseMatrixX<scalar_t> const* m, MatrixConfig const& config)
        : optimized_sizes(m->rows()), original_matrix(m), config(config) {}

    /// Create the optimized Hamiltonian targeting specific indices and scale factors
    void optimize_for(Indices const& idx, Scale<real_t> scale);

    Indices const& idx() const { return optimized_idx; }
    OptimizedSizes const& sizes() const { return optimized_sizes; }

    SparseMatrixX<scalar_t> const& csr() const {
        return optimized_matrix.template get<SparseMatrixX<scalar_t>>();
    }

    num::EllMatrix<scalar_t> const& ell() const {
        return optimized_matrix.template get<num::EllMatrix<scalar_t>>();
    }

    /// The unoptimized compute area is matrix.nonZeros() * num_moments
    std::uint64_t optimized_area(int num_moments) const;
    /// The number of mul + add operations needed to compute `num_moments` of this Hamiltonian
    std::uint64_t operations(int num_moments) const;

    std::string report(int num_moments, bool shortform = false) const;

private:
    /// Just scale the Hamiltonian: H2 = (H - I*b) * (2/a)
    void create_scaled(Indices const& idx, Scale<real_t> scale);
    /// Scale and reorder the Hamiltonian so that idx is at the start of the optimized matrix
    void create_reordered(Indices const& idx, Scale<real_t> scale);
    /// Convert CSR matrix into ELLPACK format
    static num::EllMatrix<scalar_t> convert_to_ellpack(SparseMatrixX<scalar_t> const& csr);
    /// Get optimized indices which map to the given originals
    static Indices reorder_indices(Indices const& original_idx, ArrayXi const& reorder_map);
};

CPB_EXTERN_TEMPLATE_CLASS(OptimizedHamiltonian)

}} // namespace cpb::kpm
