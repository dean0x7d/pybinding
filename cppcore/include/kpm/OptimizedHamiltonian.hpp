#pragma once
#include "hamiltonian/Hamiltonian.hpp"

#include "kpm/Bounds.hpp"
#include "kpm/Config.hpp"

#include "numeric/sparse.hpp"
#include "numeric/ellmatrix.hpp"

#include "support/variant.hpp"
#include "utils/Chrono.hpp"
#include "detail/macros.hpp"

namespace cpb { namespace kpm {

/**
 Source and destination indices for Hamiltonian optimization and local KPM calculations
 */
struct Indices {
    ArrayXi src;
    ArrayXi dest;

    Indices() = default;
    Indices(idx_t source, idx_t destination) : src(1), dest(1) {
        src[0] = static_cast<storage_idx_t>(source);
        dest[0] = static_cast<storage_idx_t>(destination);
    }

    Indices(idx_t source, ArrayXi destination)
        : src(1), dest(std::move(destination)) { src[0] = static_cast<storage_idx_t>(source); }
    Indices(idx_t source, std::vector<idx_t> const& destination)
        : Indices(source, eigen_cast<ArrayX>(destination).cast<storage_idx_t>()) {}

    Indices(ArrayXi source, ArrayXi destination)
        : src(std::move(source)), dest(std::move(destination)) {}
    Indices(std::vector<idx_t> const& source, std::vector<idx_t> const& destination)
        : Indices(eigen_cast<ArrayX>(source).cast<storage_idx_t>(),
                  eigen_cast<ArrayX>(destination).cast<storage_idx_t>()) {}

    /// Indicates a single element on the main diagonal
    bool is_diagonal() const {
        return src.size() == dest.size() && (src == dest).all();
    }

    friend bool operator==(Indices const& l, Indices const& r) {
        return l.src.size() == r.src.size() && (l.src == r.src).all()
               && l.dest.size() == r.dest.size() && (l.dest == r.dest).all();
    }
};

/**
 Optimized slice mapping for `optimal_size` and `interleaved` KPM algorithms.
 */
class SliceMap {
    std::vector<storage_idx_t> data; ///< optimized Hamiltonian indices marking slice borders
    idx_t src_offset = 0; ///< needed when there are multiple source indices (start at offset)
    idx_t dest_offset = 0; ///< indicates the slice of the highest destination index

public:
    /// Simple constructor for non-optimized case -> single slice equal to full system size
    explicit SliceMap(idx_t system_size) { data = {static_cast<storage_idx_t>(system_size)}; }
    /// Map for optimized matrix. The `optimized_idx` is needed to compute the `offset`
    SliceMap(std::vector<storage_idx_t> border_indices, Indices const& optimized_idx);

    /// Return an index into `data`, indicating the optimal system size for
    /// the calculation of KPM moment number `n` out of total `num_moments`
    idx_t index(idx_t n, idx_t num_moments) const {
        assert(n < num_moments);

        auto const mid = (num_moments - 1 + dest_offset - src_offset) / 2;
        auto const max = std::min(last_index(), mid + src_offset);
        if (n < mid) {
            return std::min(max, n + src_offset); // the size grows in the beginning
        } else { // constant in the middle and shrinking near the end as reverse `n`
            return std::min(max, num_moments - 1 - n + dest_offset);
        }
    }

    /// Last index into `data`
    idx_t last_index() const { return static_cast<idx_t>(data.size()) - 1; }

    /// Return the optimal system size for KPM moment number `n` out of total `num_moments`
    idx_t optimal_size(idx_t n, idx_t num_moments) const {
        return data[index(n, num_moments)];
    }

    /// Would calculating this number of moments ever do a full matrix-vector multiplication?
    bool uses_full_system(idx_t num_moments) const {
        return static_cast<idx_t>(data.size()) < num_moments / 2;
    }

    idx_t operator[](idx_t i) const { return data[i]; }
    std::vector<storage_idx_t> const& get_data() const { return data; }
    idx_t get_src_offset() const { return src_offset; }
    idx_t get_dest_offset() const { return dest_offset; }
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
class OptimizedHamiltonian {
public:
    using VariantMatrix = var::complex<SparseMatrixX, num::EllMatrix>;

    OptimizedHamiltonian(Hamiltonian const& h, MatrixFormat const& mf, bool reorder)
        : original_h(h), slice_map(h.rows()), matrix_format(mf), is_reordered(reorder) {}

    /// Create the optimized Hamiltonian targeting specific indices and scale factors
    void optimize_for(Indices const& idx, Scale<> scale);

    /// Apply new Hamiltonian index ordering to a container
    template<class Vector>
    void reorder(Vector& v) const {
        if (reorder_map.empty()) { return; }
        assert(reorder_map.size() == static_cast<size_t>(v.size()));

        auto reordered_v = Vector(v.size());
        for (auto i = idx_t{0}; i < reordered_v.size(); ++i) {
            reordered_v[reorder_map[i]] = v[i];
        }
        v.swap(reordered_v);
    }

    template<class scalar_t>
    void reorder(SparseMatrixX<scalar_t>& matrix) const {
        if (reorder_map.empty()) { return; }

        auto reordered_matrix = SparseMatrixX<scalar_t>(matrix.rows(), matrix.cols());
        auto const reserve_per_row = static_cast<int>(sparse::max_nnz_per_row(matrix));
        reordered_matrix.reserve(ArrayXi::Constant(matrix.rows(), reserve_per_row));

        sparse::make_loop(matrix).for_each([&](idx_t row, idx_t col, scalar_t value) {
            reordered_matrix.insert(reorder_map[row], reorder_map[col]) = value;
        });

        reordered_matrix.makeCompressed();
        matrix.swap(reordered_matrix);
    }

    idx_t size() const { return original_h.rows(); }
    Indices const& idx() const { return optimized_idx; }
    SliceMap const& map() const { return slice_map; }
    VariantMatrix const& matrix() const { return optimized_matrix; }
    var::scalar_tag scalar_tag() const { return tag; }

private:
    /// Just scale the Hamiltonian: H2 = (H - I*b) * (2/a)
    template<class scalar_t>
    void create_scaled(Indices const& idx, Scale<> scale);
    /// Scale and reorder the Hamiltonian so that idx is at the start of the optimized matrix
    template<class scalar_t>
    void create_reordered(Indices const& idx, Scale<> scale);
    /// Get optimized indices which map to the given originals
    static Indices reorder_indices(Indices const& original_idx,
                                   std::vector<storage_idx_t> const& reorder_map);

    /// Total non-zeros processed when computing `num_moments` with or without size optimizations
    size_t num_nonzeros(idx_t num_moments, bool optimal_size) const;
    /// Same as above but with vector elements instead of sparse matrix non-zeros
    size_t num_vec_elements(idx_t num_moments, bool optimal_size) const;
    /// The amount of memory (in bytes) used by the Hamiltonian matrix
    size_t matrix_memory() const;
    /// Memory used by a single KPM vector
    size_t vector_memory() const;

private:
    Hamiltonian original_h; ///< original unoptimized Hamiltonian
    Indices original_idx; ///< original target indices for which the optimization was done

    VariantMatrix optimized_matrix; ///< reordered for faster compute
    var::scalar_tag tag; ///< indicates the scalar type of the matrix
    Indices optimized_idx; ///< reordered target indices in the optimized matrix
    SliceMap slice_map; ///< slice border indices
    std::vector<storage_idx_t> reorder_map; ///< mapping from original matrix indices to reordered indices

    MatrixFormat matrix_format;
    bool is_reordered;
    Chrono timer;

    friend struct Stats;
    friend struct Optimize;
};

}} // namespace cpb::kpm
