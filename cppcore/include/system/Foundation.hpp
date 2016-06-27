#pragma once
#include "system/Lattice.hpp"

#include "detail/slice.hpp"
#include "numeric/dense.hpp"
#include "support/cppfuture.hpp"

#include <array>
#include <vector>
#include <cstdint>

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
    ArrayX<int16_t> count_neighbors(Foundation const& foundation);
    /// Reduce this site's neighbor count to zero and inform its neighbors of the change
    void clear_neighbors(Site& site, ArrayX<int16_t>& neighbor_count, int min_neighbors);
    /// Make an array of sublattice ids for the entire foundation
    ArrayX<sub_id> make_sublattice_ids(Foundation const& foundation);
} // namespace detail

/// Remove sites which have a neighbor count lower than `min_neighbors`
void remove_dangling(Foundation& foundation, int min_neighbors);

/**
 The foundation class creates a lattice-vector-aligned set of sites. The number of sites is high
 enough to encompass the given shape. After creation, the foundation can be cut down to the shape.
 */
class Foundation {
    Lattice const& lattice;
    std::pair<Index3D, Index3D> bounds; ///< in lattice vector coordinates
    Index3D size; ///< number of unit cells in each lattice vector direction
    int size_n; ///< sublattice size (number of sites in each unit cell)
    int num_sites; ///< total number of sites: product of all sizes (3D and sublattice)

    CartesianArray positions; ///< real space coordinates of lattice sites
    ArrayX<bool> is_valid; ///< indicates if the site should be included in the final system

    friend class Site;

private:
    template<bool is_const> class Iterator;
    using ConstIterator = Iterator<true>;
    using NonConstIterator = Iterator<false>;

    template<bool is_const> class Slice;
    using ConstSlice = Slice<true>;
    using NonConstSlice = Slice<false>;

public:
    Foundation(Lattice const& lattice, Primitive const& shape);
    Foundation(Lattice const& lattice, Shape const& shape);

    ConstIterator begin() const;
    ConstIterator end() const;
    NonConstIterator begin();
    NonConstIterator end();

    ConstSlice operator[](SliceIndex3D const& index) const;
    NonConstSlice operator[](SliceIndex3D const& index);

    Lattice const& get_lattice() const { return lattice; }
    std::pair<Index3D, Index3D> const& get_bounds() const { return bounds; }
    Index3D const& get_size() const { return size; }
    int get_num_sublattices() const { return size_n; }
    int get_num_sites() const { return num_sites; }

    CartesianArray const& get_positions() const { return positions; }
    CartesianArray& get_positions() { return positions; }
    ArrayX<bool> const& get_states() const { return is_valid; }
    ArrayX<bool>& get_states() { return is_valid; }
};

/**
 Describes a site on the lattice foundation

 Proxy type for a single index in the foundation arrays.
 */
class Site {
protected:
    Foundation* foundation; ///< the site's parent foundation
    Index3D index; ///< unit cell spatial index
    int sublattice; ///< sublattice index
    int idx; ///< flat index for array addressing

    /// Recalculate flat `idx` from spatial `index` and `sublattice`
    void reset_idx() {
        auto const& size = foundation->size;
        auto const& size_n = foundation->size_n;
        idx = ((index[0] * size[1] + index[1]) * size[2] + index[2]) * size_n + sublattice;
    }

public:
    Site(Foundation* foundation, Index3D index, int sublattice, int idx)
        : foundation(foundation), index(index), sublattice(sublattice), idx(idx) {}
    Site(Foundation* foundation, Index3D index, int sublattice)
        : foundation(foundation), index(index), sublattice(sublattice) {
        reset_idx();
    }

    Index3D const& get_index() const { return index; }
    int get_sublattice() const { return sublattice; }
    int get_idx() const { return idx; }
    Lattice const& get_lattice() const { return foundation->lattice; }

    Cartesian get_position() const { return foundation->positions[idx]; }

    bool is_valid() const { return foundation->is_valid[idx]; }
    void set_valid(bool state) {foundation->is_valid[idx] = state; }

    Site shifted(Index3D shift) const { return {foundation, index + shift, sublattice}; }

    /// Loop over all neighbours of this site
    template<class Fn>
    void for_each_neighbour(Fn lambda) const  {
        for (auto const& hopping : foundation->lattice[sublattice].hoppings) {
            Array3i const neighbor_index = (index + hopping.relative_index).array();
            if (any_of(neighbor_index < 0) || any_of(neighbor_index >= foundation->size.array()))
                continue; // out of bounds

            lambda(Site(foundation, neighbor_index, hopping.to_sublattice), hopping);
        }
    }

    friend bool operator==(Site const& l, Site const& r) { return l.idx == r.idx; }
    friend bool operator!=(Site const& l, Site const& r) { return !(l == r); }
};

/**
 Hamiltonian indices of valid Foundation sites
 */
class HamiltonianIndices {
    ArrayX<int> indices;
    int num_valid_sites;

public:
    HamiltonianIndices(Foundation const& foundation);

    /// Return the Hamiltonian matrix index for the given site
    int operator[](Site const& site) const { return indices[site.get_idx()]; }
    /// Size of the Hamiltonian matrix
    int size() const { return num_valid_sites; }
};

template<bool is_const>
class Foundation::Iterator
    : public Site,
      public std::iterator<std::input_iterator_tag,
                           std14::conditional_t<is_const, Site const, Site>> {
    using Ref = std14::conditional_t<is_const, Site const&, Site&>;

public:
    Iterator(Foundation* foundation, int idx) : Site(foundation, {0, 0, 0}, 0, idx) {}

    Ref operator*() { return *this; }

    Iterator& operator++() {
        ++idx;
        ++sublattice;
        if (sublattice == foundation->size_n) {
            ++index[2];
            if (index[2] == foundation->size[2]) {
                ++index[1];
                if (index[1] == foundation->size[1]) {
                    ++index[0];
                    index[1] = 0;
                }
                index[2] = 0;
            }
            sublattice = 0;
        }
        return *this;
    }
};

template<bool is_const>
class Foundation::Slice {
    Foundation* foundation;
    SliceIndex3D index;

private:
    class Iterator
        : public Site,
          public std::iterator<std::input_iterator_tag,
                               std14::conditional_t<is_const, Iterator const, Iterator>> {
        using Ref = std14::conditional_t<is_const, Iterator const&, Iterator&>;

        SliceIndex3D slice_index;
        int slice_idx;

    public:
        Iterator(Foundation* foundation, SliceIndex3D index, int slice_idx)
            : Site(foundation, {index[0].start, index[1].start, index[2].start}, 0),
              slice_index(index), slice_idx(slice_idx) {}

        int get_slice_idx() const { return slice_idx; }

        Ref operator*() { return *this; }

        Iterator& operator++() {
            ++slice_idx;
            ++sublattice;
            if (sublattice == foundation->size_n) {
                ++index[2];
                if (index[2] == slice_index[2].end) {
                    ++index[1];
                    if (index[1] == slice_index[1].end) {
                        ++index[0];
                        index[1] = slice_index[1].start;
                    }
                    index[2] = slice_index[2].start;
                }
                sublattice = 0;
            }
            reset_idx();
            return *this;
        }

        friend bool operator==(Iterator const& l, Iterator const& r) {
            return l.slice_idx == r.slice_idx;
        }
        friend bool operator!=(Iterator const& l, Iterator const& r) { return !(l == r); }
    };

public:
    Slice(Foundation* foundation, SliceIndex3D const& index)
        : foundation(foundation), index(index) {
        normalize();
    }

    Iterator begin() const { return {foundation, index, 0}; }
    Iterator end() const { return {foundation, index, size()}; }

    int size() const { return index.size() * foundation->get_num_sublattices(); }
    SliceIndex const& operator[](int n) const { return index[n]; }
    SliceIndex& operator[](int n) { return index[n]; }

    /// Replace open ended indices [0, -1) with proper [0, size) indices
    void normalize() {
        for (auto i = 0; i < index.ndims(); ++i) {
            if (index[i].end < 0) {
                index[i].end = foundation->get_size()[i];
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
};


inline Foundation::ConstIterator Foundation::begin() const {
    return {const_cast<Foundation*>(this), 0};
}

inline Foundation::ConstIterator Foundation::end() const {
    return {const_cast<Foundation*>(this), num_sites};
}

inline Foundation::NonConstIterator Foundation::begin() {
    return {this, 0};
}

inline Foundation::NonConstIterator Foundation::end() {
    return {this, num_sites};
}

inline Foundation::ConstSlice Foundation::operator[](SliceIndex3D const& index) const {
    return {const_cast<Foundation*>(this), index};
}

inline Foundation::NonConstSlice Foundation::operator[](SliceIndex3D const& index) {
    return {this, index};
}

} // namespace cpb
