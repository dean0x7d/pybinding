#pragma once
#include "kpm/Bounds.hpp"
#include "kpm/Config.hpp"
#include "kpm/Stats.hpp"

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
 Optimized slice mapping for `optimal_size` and `interleaved` KPM algorithms.
 */
class SliceMap {
    std::vector<int> data; ///< optimized Hamiltonian indices marking slice borders
    int offset = 0; ///< needed to correctly compute off-diagonal elements (i != j)

public:
    /// Simple constructor for non-optimized case -> single slice equal to full system size
    explicit SliceMap(int system_size) { data = {system_size}; }
    /// Map for optimized matrix. The `optimized_idx` is needed to compute the `offset`
    SliceMap(std::vector<int> border_indices, Indices const& optimized_idx);

    /// Return an index into `data`, indicating the optimal system size for
    /// the calculation of KPM moment number `n` out of total `num_moments`
    int index(int n, int num_moments) const {
        assert(n < num_moments);

        auto const max = std::min(last_index(), num_moments / 2);
        if (n < max) {
            return n; // size grows in the beginning
        } else { // constant in the middle and shrinking near the end as reverse `n`
            return std::min(max, num_moments - 1 - n + offset);
        }
    }

    /// Last index into `data`
    int last_index() const { return static_cast<int>(data.size()) - 1; }

    /// Return the optimal system size for KPM moment number `n` out of total `num_moments`
    int optimal_size(int n, int num_moments) const {
        return data[index(n, num_moments)];
    }

    /// Would calculating this number of moments ever do a full matrix-vector multiplication?
    bool uses_full_system(int num_moments) const {
        return static_cast<int>(data.size()) < num_moments / 2;
    }

    int operator[](int i) const { return data[i]; }
    std::vector<int> const& get_data() const { return data; }
    int get_offset() const { return offset; }
};

/**
 Stores a scaled Hamiltonian `(H - b)/a` which limits it to (-1, 1) boundaries required for KPM.
 In addition, three optimisations are applied (last two are optional, see `MatrixConfig`):

 1) The matrix is multiplied by 2. This benefits most calculations (e.g. `y = 2*H*x - y`),
    because the 2x multiplication is done only once, but it will need to be divided by 2
    when the original element values are needed (very rarely).

 2) Reorder the elements so that target indices are placed at the start of the matrix.
    This produces a `SliceMap` which may be used to reduce calculation time by skipping
    sparse matrix-vector multiplication of zero values or by interleaving calculations
    of neighboring slices.

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
    SliceMap slice_map; ///< slice border indices
    std::vector<int> reorder_map; ///< mapping from original matrix indices to reordered indices

    SparseMatrixX<scalar_t> const* original_matrix;
    Indices original_idx; ///< original target indices for which the optimization was done

    MatrixFormat matrix_format;
    bool reorder;
    Chrono timer;

public:
    OptimizedHamiltonian(SparseMatrixX<scalar_t> const* m, MatrixFormat const& mf, bool reorder)
        : slice_map(m->rows()), original_matrix(m), matrix_format(mf), reorder(reorder) {}

    /// Create the optimized Hamiltonian targeting specific indices and scale factors
    void optimize_for(Indices const& idx, Scale<real_t> scale);

    /// Apply new Hamiltonian index ordering to a vector
    template<class Vector>
    Vector reorder_vector(Vector const& v) const {
        if (reorder_map.empty()) { return v; }
        assert(reorder_map.size() == v.size());

        auto result = Vector(v.size());
        for (auto i = 0; i < result.size(); ++i) {
            result[reorder_map[i]] = v[i];
        }
        return result;
    }

    int size() const { return original_matrix->rows(); }
    Indices const& idx() const { return optimized_idx; }
    SliceMap const& map() const { return slice_map; }
    OptMatrix const& matrix() const { return optimized_matrix; }

    void populate_stats(Stats& s, int num_moments, AlgorithmConfig const& c) const;

private:
    /// Just scale the Hamiltonian: H2 = (H - I*b) * (2/a)
    void create_scaled(Indices const& idx, Scale<real_t> scale);
    /// Scale and reorder the Hamiltonian so that idx is at the start of the optimized matrix
    void create_reordered(Indices const& idx, Scale<real_t> scale);
    /// Convert CSR matrix into ELLPACK format
    static num::EllMatrix<scalar_t> convert_to_ellpack(SparseMatrixX<scalar_t> const& csr);
    /// Get optimized indices which map to the given originals
    static Indices reorder_indices(Indices const& original_idx,
                                   std::vector<int> const& reorder_map);

    /// Total non-zeros processed when computing `num_moments` with or without size optimizations
    size_t num_nonzeros(int num_moments, bool optimal_size) const;
    /// Same as above but with vector elements instead of sparse matrix non-zeros
    size_t num_vec_elements(int num_moments, bool optimal_size) const;
};

CPB_EXTERN_TEMPLATE_CLASS(OptimizedHamiltonian)

}} // namespace cpb::kpm
