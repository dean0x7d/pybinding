#pragma once
#include "Lattice.hpp"

#include "detail/slice.hpp"
#include "numeric/dense.hpp"
#include "support/cppfuture.hpp"

namespace cpb {

class Primitive;
class Shape;
class Site;
class Foundation;

namespace detail {
    /// Return the lower and upper bounds of the shape in lattice vector coordinates
    std::pair<Index3D, Index3D> find_bounds(Shape const& shape, Lattice const& lattice);
    /// Generate real space coordinates for a block of lattice sites
    CartesianArray generate_positions(Cartesian origin, Index3D size, Lattice const& lattice);
    /// Initialize the neighbor count for each site
    ArrayXi count_neighbors(Foundation const& foundation);
    /// Reduce this site's neighbor count to zero and inform its neighbors of the change
    void clear_neighbors(Site& site, ArrayXi& neighbor_count, int min_neighbors);
} // namespace detail

/// Remove sites which have a neighbor count lower than `min_neighbors`
void remove_dangling(Foundation& foundation, int min_neighbors);

/**
 Keeps the final indices of valid `Foundation` sites as they should appear in `System`
 */
class FinalizedIndices {
public:
    FinalizedIndices() = default;
    FinalizedIndices(ArrayXi indices, ArrayXi hopping_counts, idx_t total_valid_sites);

    explicit operator bool() const { return total_valid_sites != 0; }

    /// Return the System index for the given site
    storage_idx_t operator[](Site const& site) const;
    /// Size of the Hamiltonian matrix
    idx_t size() const { return total_valid_sites; }
    /// Upper limit for the number of hoppings, indexed by family ID (useful for reservation)
    ArrayXi const& max_hoppings_per_family() const { return hopping_counts; }

private:
    ArrayXi indices;
    ArrayXi hopping_counts;
    idx_t total_valid_sites = 0;
};

/**
 The foundation class creates a lattice-vector-aligned set of sites. The number of sites is high
 enough to encompass the given shape. After creation, the foundation can be cut down to the shape.
 */
class Foundation {
    template<bool is_const> class Iterator;
    using ConstIterator = Iterator<true>;
    using NonConstIterator = Iterator<false>;

    template<bool is_const> class SpatialSlice;
    using ConstSpatialSlice = SpatialSlice<true>;
    using NonConstSpatialSlice = SpatialSlice<false>;

    template<bool is_const> class SublatticeSlice;
    using ConstSublatticeSlice = SublatticeSlice<true>;
    using NonConstSublatticeSlice = SublatticeSlice<false>;

public:
    Foundation(Lattice const& lattice, Primitive const& shape);
    Foundation(Lattice const& lattice, Shape const& shape);

    ConstIterator begin() const;
    ConstIterator end() const;
    NonConstIterator begin();
    NonConstIterator end();

    ConstSpatialSlice operator[](SliceIndex3D const& index) const;
    NonConstSpatialSlice operator[](SliceIndex3D const& index);
    ConstSublatticeSlice operator[](SubID id) const;
    NonConstSublatticeSlice operator[](SubID id);

    /// Total number of sites: product of all sizes (3D space and sublattice)
    idx_t size() const { return spatial_size.prod() * sub_size; }

    Lattice const& get_lattice() const { return lattice; }
    OptimizedUnitCell const& get_optimized_unit_cell() const { return unit_cell; }
    std::pair<Index3D, Index3D> const& get_bounds() const { return bounds; }
    Index3D const& get_spatial_size() const { return spatial_size; }
    idx_t get_sub_size() const { return sub_size; }

    CartesianArray const& get_positions() const { return positions; }
    CartesianArray& get_positions() { return positions; }
    ArrayX<bool> const& get_states() const { return is_valid; }
    ArrayX<bool>& get_states() { return is_valid; }

    FinalizedIndices const& get_finalized_indices() const;

private:
    Lattice const& lattice;
    OptimizedUnitCell unit_cell;
    std::pair<Index3D, Index3D> bounds; ///< in lattice vector coordinates
    Index3D spatial_size; ///< number of unit cells in each lattice vector direction
    idx_t sub_size; ///< number of sites in a unit cell (sublattices)

    CartesianArray positions; ///< real space coordinates of lattice sites
    ArrayX<bool> is_valid; ///< indicates if the site should be included in the final system

    mutable FinalizedIndices finalized_indices;

    friend class Site;
};

/// Convenient alias
using Hopping = OptimizedUnitCell::Hopping;

/**
 Describes a site on the lattice foundation

 Proxy type for a single index in the foundation arrays.
 */
class Site {
public:
    /// Direct index assignment
    Site(Foundation* foundation, Index3D spatial_idx, idx_t sub_idx, idx_t idx)
        : foundation(foundation), spatial_idx(spatial_idx), sub_idx(sub_idx), flat_idx(idx) {}
    /// Compute flat index based on 3D space + sub_idx
    Site(Foundation* foundation, Index3D spatial_idx, idx_t sub_idx)
        : foundation(foundation), spatial_idx(spatial_idx), sub_idx(sub_idx) { reset_idx(); }

    Index3D const& get_spatial_idx() const { return spatial_idx; }
    idx_t get_sub_idx() const { return sub_idx; }
    idx_t get_flat_idx() const { return flat_idx; }

    SubAliasID get_alias_id() const { return foundation->unit_cell[sub_idx].alias_id; }
    storage_idx_t get_norb() const { return foundation->unit_cell[sub_idx].norb; }

    Cartesian get_position() const { return foundation->positions[flat_idx]; }
    bool is_valid() const { return foundation->is_valid[flat_idx]; }
    void set_valid(bool state) {foundation->is_valid[flat_idx] = state; }

    /// Return a new site which has a shifted spatial index
    Site shifted(Index3D shift) const { return {foundation, spatial_idx + shift, sub_idx}; }

    /// Loop over all neighbours of this site
    template<class Fn>
    void for_each_neighbor(Fn lambda) const  {
        auto const spatial_size = foundation->spatial_size.array();

        for (auto const& hopping : foundation->unit_cell[sub_idx].hoppings) {
            auto const neighbor_index = Array3i(spatial_idx + hopping.relative_index);
            if ((neighbor_index < 0).any() || (neighbor_index >= spatial_size).any())
                continue; // out of bounds

            lambda(Site(foundation, neighbor_index, hopping.to_sub_idx), hopping);
        }
    }

    friend bool operator==(Site const& l, Site const& r) { return l.flat_idx == r.flat_idx; }
    friend bool operator!=(Site const& l, Site const& r) { return !(l == r); }

protected:
    /// Recalculate `flat_idx`
    void reset_idx() {
        auto const& i = spatial_idx;
        auto const& size = foundation->spatial_size;
        flat_idx = ((sub_idx * size[2] + i[2]) * size[1] + i[1]) * size[0] + i[0];
    }

protected:
    Foundation* foundation; ///< the site's parent foundation
    Index3D spatial_idx; ///< unit cell spatial index
    idx_t sub_idx; ///< sublattice index
    idx_t flat_idx; ///< directly corresponds to array elements
};

/**
 Iterate over all foundation sites
 */
template<bool is_const>
class Foundation::Iterator : public Site {
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std14::conditional_t<is_const, Site const, Site>;
    using reference = value_type&;
    using pointer = value_type*;

public:
    Iterator(Foundation* foundation, idx_t flat_idx) : Site(foundation, {0, 0, 0}, 0, flat_idx) {}

    reference operator*() { return *this; }
    pointer operator->() { return this; }

    Iterator& operator++() {
        ++flat_idx;
        ++spatial_idx[0];
        if (spatial_idx[0] == foundation->spatial_size[0]) {
            spatial_idx[0] = 0;
            ++spatial_idx[1];
            if (spatial_idx[1] == foundation->spatial_size[1]) {
                spatial_idx[1] = 0;
                ++spatial_idx[2];
                if (spatial_idx[2] == foundation->spatial_size[2]) {
                    spatial_idx[2] = 0;
                    ++sub_idx;
                }
            }
        }
        return *this;
    }
};

/**
 Iterate only over sites inside the spatial slice
 */
template<bool is_const>
class SpatialSliceIterator : public Site {
    using It = SpatialSliceIterator<is_const>;
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std14::conditional_t<is_const, It const, It>;
    using reference = value_type&;
    using pointer = value_type*;

public:
    SpatialSliceIterator(Foundation* foundation, SliceIndex3D range, idx_t slice_idx)
        : Site(foundation, {range[0].start, range[1].start, range[2].start}, 0),
          range(range), slice_idx(slice_idx) {}

    /// Flat index within the slice: `0 < slice_idx < slice_size`
    idx_t get_slice_idx() const { return slice_idx; }

    reference operator*() { return *this; }
    pointer operator->() { return this; }

    It& operator++() {
        ++slice_idx;
        ++spatial_idx[0];
        if (spatial_idx[0] == range[0].end) {
            spatial_idx[0] = range[0].start;
            ++spatial_idx[1];
            if (spatial_idx[1] == range[1].end) {
                spatial_idx[1] = range[1].start;
                ++spatial_idx[2];
                if (spatial_idx[2] == range[2].end) {
                    spatial_idx[2] = range[2].start;
                    ++sub_idx;
                }
            }
        }
        reset_idx();
        return *this;
    }

    friend bool operator==(It const& l, It const& r) { return l.slice_idx == r.slice_idx; }
    friend bool operator!=(It const& l, It const& r) { return !(l == r); }

private:
    SliceIndex3D range; ///< slice start and end indices in 3 dimensions
    idx_t slice_idx; ///< flat index within the slice
};

/**
 A 3D slice view of a foundation
 */
template<bool is_const>
class Foundation::SpatialSlice {
    using Iterator = SpatialSliceIterator<is_const>;

public:
    SpatialSlice(Foundation* foundation, SliceIndex3D const& range)
        : foundation(foundation), range(range) { normalize(); }

    Iterator begin() const { return {foundation, range, 0}; }
    Iterator end() const { return {foundation, range, size()}; }

    idx_t size() const { return range.size() * foundation->get_sub_size(); }
    SliceIndex const& operator[](int n) const { return range[n]; }
    SliceIndex& operator[](int n) { return range[n]; }

    /// Replace open ended indices [0, -1) with proper [0, size) indices
    void normalize() {
        for (auto i = 0; i < range.ndims(); ++i) {
            if (range[i].end < 0) {
                range[i].end = foundation->get_spatial_size()[i];
            }
        }
    }

    CartesianArray positions() const {
        auto positions = CartesianArray(size());
        for (auto const& site : *this) {
            positions[site.get_slice_idx()] = site.get_position();
        }
        return positions;
    }

private:
    Foundation* foundation;
    SliceIndex3D range;
};

/**
 A single sublattice slice view of a foundation
 */
template<bool is_const>
class Foundation::SublatticeSlice {
    using Iterator = SpatialSliceIterator<is_const>;

public:
    SublatticeSlice(Foundation* foundation, SubID unique_id) : foundation(foundation) {
        using CellSite = OptimizedUnitCell::Site;
        auto const& unit_cell = foundation->get_optimized_unit_cell();
        auto const it = std::find_if(unit_cell.begin(), unit_cell.end(),
                                     [&](CellSite const& s) { return s.unique_id == unique_id; });
        if (it == unit_cell.end()) {
            throw std::runtime_error("Foundation::SublatticeSlice: invalid sublattice unique_id");
        }

        slice_size = foundation->get_spatial_size().prod();
        start_idx = (it - unit_cell.begin()) * slice_size;
    }

    idx_t size() const { return slice_size; }

    Foundation::Iterator<is_const> begin() const { return {foundation, start_idx}; }
    Foundation::Iterator<is_const> end() const { return {foundation, start_idx + slice_size}; }

    Eigen::Ref<ArrayX<bool> const> get_states() const {
        return foundation->is_valid.segment(start_idx, slice_size);
    }

    Eigen::Ref<ArrayX<bool>> get_states() {
        return foundation->is_valid.segment(start_idx, slice_size);
    }

    CartesianArrayConstRef get_positions() const {
        return foundation->positions.segment(start_idx, slice_size);
    }

    CartesianArrayRef get_positions() {
        return foundation->positions.segment(start_idx, slice_size);
    }

private:
    Foundation* foundation;
    idx_t start_idx;
    idx_t slice_size;
};

inline storage_idx_t FinalizedIndices::operator[](Site const& site) const {
    return indices[site.get_flat_idx()];
}

inline Foundation::ConstIterator Foundation::begin() const {
    return {const_cast<Foundation*>(this), 0};
}

inline Foundation::ConstIterator Foundation::end() const {
    return {const_cast<Foundation*>(this), size()};
}

inline Foundation::NonConstIterator Foundation::begin() {
    return {this, 0};
}

inline Foundation::NonConstIterator Foundation::end() {
    return {this, size()};
}

inline Foundation::ConstSpatialSlice Foundation::operator[](SliceIndex3D const& index) const {
    return {const_cast<Foundation*>(this), index};
}

inline Foundation::NonConstSpatialSlice Foundation::operator[](SliceIndex3D const& index) {
    return {this, index};
}

inline Foundation::ConstSublatticeSlice Foundation::operator[](SubID unique_id) const {
    return {const_cast<Foundation*>(this), unique_id};
}

inline Foundation::NonConstSublatticeSlice Foundation::operator[](SubID unique_id) {
    return {this, unique_id};
}

} // namespace cpb
