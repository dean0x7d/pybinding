#pragma once
#include "Lattice.hpp"

#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"

namespace cpb {

/// Alternative CSR mapping of hopping IDs
using HoppingCSR = SparseMatrixX<storage_idx_t>;

/**
 A simple row and column index pair
 */
struct COO {
    storage_idx_t row;
    storage_idx_t col;

    COO() = default;
    COO(idx_t row, idx_t col)
        : row(static_cast<storage_idx_t>(row)),
          col(static_cast<storage_idx_t>(col)) {}

    friend bool operator==(COO const& a, COO const& b) {
        return std::tie(a.row, a.col) == std::tie(b.row, b.col);
    }
    friend bool operator<(COO const& a, COO const& b) {
        return std::tie(a.row, a.col) < std::tie(b.row, b.col);
    }
};

/**
 Hopping coordinates arranged in per-family blocks

 A hopping here represents a connection between two sites, not orbitals.
 The `row` and `col` index sites which are connected by a hopping family
 represented by numerical ID (the sparse matrix `data`). If a hopping
 family has an energy matrix (instead of a scalar) then it will need to
 be expanded to get the full orbital-to-orbital hoppings. But this happens
 at a later stage. This data structure is only concerned with site-to-site
 hoppings arranged in per-family blocks.

 Each block corresponds to a COO sparse matrix where all the elements in
 the data array are the same and correspond to the index of the block,
 i.e. the hopping family ID:

         block 0                 block 1                 block 2
     row | col | data        row | col | data        row | col | data
     ----------------        ----------------        ----------------
      0  |  1  |  0           0  |  4  |  1           1  |  3  |  2
      0  |  4  |  0           2  |  3  |  1           4  |  4  |  2
      1  |  2  |  0           2  |  0  |  1           7  |  9  |  2
      3  |  2  |  0          ----------------         8  |  1  |  2
      7  |  5  |  0                                  ----------------
     ----------------

 Because the data array is trivial, it doesn't actually need to be stored.
 The full COO sparse matrix can be reconstructed by appending all the blocks
 and reconstructing the implicit data array.

 The row-col coordinate pairs are unique (over all blocks) and sorted to
 maintain an upper triangular matrix per block (implied hermiticity supplies
 the lower triangular portion and is not actually stored in memory).
 */
class HoppingBlocks {
public:
    using Block = std::vector<COO>;
    using Blocks = std::vector<Block>;
    using SerializedBlocks = std::vector<std::pair<ArrayXi, ArrayXi>>; // format for saving to file

    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Iterator;
        using reference = value_type const&;
        using pointer = value_type const*;

        Iterator(Blocks::const_iterator it) : it(it) {}

        HopID family_id() const { return HopID(id); }
        Block const& coordinates() const { return *it; }
        idx_t size() const { return static_cast<idx_t>(it->size()); }

        reference operator*() { return *this; }
        pointer operator->() { return this; }
        Iterator& operator++() { ++it; ++id; return *this; }

        friend bool operator==(Iterator const& a, Iterator const& b) { return a.it == b.it; }
        friend bool operator!=(Iterator const& a, Iterator const& b) { return !(a == b); }

    private:
        Blocks::const_iterator it;
        storage_idx_t id = 0;
    };

public:
    HoppingBlocks() = default;
    HoppingBlocks(idx_t num_sites, NameMap name_map)
        : num_sites(num_sites), blocks(name_map.size()), name_map(std::move(name_map)) {}
    /// Internal: construct from serialized data
    HoppingBlocks(idx_t num_sites, SerializedBlocks const& data, NameMap name_map);

    /// Internal: return serialized data
    idx_t get_num_sites() const { return num_sites; }
    SerializedBlocks get_serialized_blocks() const;
    NameMap const& get_name_map() const { return name_map; }

    Iterator begin() const { return blocks.begin(); }
    Iterator end() const { return blocks.end(); }

    /// Number of non-zeros in this COO sparse matrix, i.e. the total number of hoppings.
    /// This only includes the upper triangular part (i.e. does not include 2x for hermiticity).
    idx_t nnz() const;

    /// Return the number of neighbors for each site
    ArrayXi count_neighbors() const;

    /// Reserve space the given number of hoppings per family
    void reserve(ArrayXi const& counts);

    /// Add a single coordinate pair to the given family block
    void add(HopID family_id, idx_t row, idx_t col) {
        blocks[family_id.as<size_t>()].push_back({row, col});
    }

    /// Append a range of coordinates to the given family block
    void append(HopID family_id, ArrayXi&& rows, ArrayXi&& cols);

    /// Remove sites for which `keep == false`
    void filter(VectorX<bool> const& keep);

    /// Account for the addition of new sites (no new hoppings)
    void add_sites(idx_t num_new_sites);

    /// Return the matrix in the CSR sparse matrix format
    HoppingCSR tocsr() const;

private:
    idx_t num_sites; ///< number of lattice sites, i.e. the size of the square matrix
    Blocks blocks; ///< the coordinate blocks indexed by hopping family ID
    NameMap name_map; ///< map from friendly hopping family names to their numeric IDs
};

} // namespace cpb
