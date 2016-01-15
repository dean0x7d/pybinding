#pragma once
#include "system/Lattice.hpp"

#include "detail/slice.hpp"
#include "support/dense.hpp"

#include <array>
#include <vector>
#include <cstdint>

namespace tbm {

class Primitive;
class Shape;
class Site;
class Foundation;
class FoundationConstIterator;
class FoundationIterator;
class SliceIterator;

namespace detail {
    /// Return the lower and upper bounds of the shape in lattice vector coordinates
    std::pair<Index3D, Index3D> find_bounds(Shape const& shape, Lattice const& lattice);
    /// Generate real space coordinates for a block of lattice sites
    CartesianArray generate_positions(Cartesian origin, Index3D size, Lattice const& lattice);
    /// Initialize the neighbor count for each site
    ArrayX<int16_t> count_neighbors(Foundation const& foundation);
    /// Reduce this site's neighbor count to zero and inform its neighbors of the change
    void clear_neighbors(Site& site, ArrayX<int16_t>& neighbor_count);
    /// Remove edge sites which have a neighbor count lower than the lattice minimum
    void trim_edges(Foundation& foundation);
    /// Make an array of sublattice ids for the entire foundation
    ArrayX<sub_id> make_sublattice_ids(Foundation const& foundation);
} // namespace detail

/**
 The foundation class creates a lattice-vector-aligned set of sites. The number of sites is high
 enough to encompass the given shape. After creation, the foundation can be cut down to the shape.
 */
class Foundation {
public:
    Lattice const& lattice;
    Index3D size; ///< number of unit cells in each lattice vector direction
    int size_n; ///< sublattice size (number of sites in each unit cell)
    int num_sites; ///< total number of sites: product of all sizes (3D and sublattice)

    CartesianArray positions; ///< real space coordinates of lattice sites
    ArrayX<bool> is_valid; ///< indicates if the site should be included in the final system

public:
    struct Slice {
        Foundation* const foundation;
        SliceIndex3D index;

        SliceIterator begin();
        SliceIterator end();
    };

public:
    Foundation(Lattice const& lattice, Primitive const& shape);
    Foundation(Lattice const& lattice, Shape const& shape);

    FoundationConstIterator begin() const;
    FoundationConstIterator end() const;
    FoundationIterator begin();
    FoundationIterator end();

    Slice operator[](SliceIndex3D const& index) { return {this, index}; }
};

/**
 Describes a site on the lattice foundation

 Proxy type for a single index in the foundation arrays.
 */
class Site {
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

    friend class Foundation;
    friend class FoundationConstIterator;
    friend class FoundationIterator;
    friend class SliceIterator;

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

class FoundationConstIterator {
protected:
    Site site;

public:
    FoundationConstIterator(Foundation* foundation, int idx)
        : site{foundation, {0, 0, 0}, 0, idx} {}

    Site const& operator*() const { return site; }

    FoundationConstIterator& operator++() {
        ++site.idx;
        ++site.sublattice;
        if (site.sublattice == site.foundation->size_n) {
            ++site.index[2];
            if (site.index[2] == site.foundation->size[2]) {
                ++site.index[1];
                if (site.index[1] == site.foundation->size[1]) {
                    ++site.index[0];
                    site.index[1] = 0;
                }
                site.index[2] = 0;
            }
            site.sublattice = 0;
        }
        return *this;
    }

    FoundationConstIterator operator++(int) {
        auto const copy = *this;
        this->operator++();
        return copy;
    }

    friend bool operator==(FoundationConstIterator const& l, FoundationConstIterator const& r) {
        return l.site.get_idx() == r.site.get_idx();
    }
    friend bool operator!=(FoundationConstIterator const& l, FoundationConstIterator const& r) {
        return !(l == r);
    }
};

class FoundationIterator : public FoundationConstIterator {
public:
    using FoundationConstIterator::FoundationConstIterator;

    Site& operator*() { return site; }
};

class SliceIterator {
    Site site;
    SliceIndex3D slice_index;

    static SliceIndex3D normalize(SliceIndex3D const& index, Index3D const& size) {
        auto ret = index;
        for (auto i = 0u; i < 3; ++i) {
            if (ret[i].end < 0) {
                ret[i].end = size[i];
            }
        }
        return ret;
    }

public:
    SliceIterator(Foundation* foundation) : site{foundation, {}, 0, -1} {}
    SliceIterator(Foundation* foundation, SliceIndex3D index)
        : site(foundation, {index[0].start, index[1].start, index[2].start}, 0),
          slice_index(normalize(index, foundation->size)) {}

    Site& operator*() { return site; }

    SliceIterator& operator++() {
        ++site.sublattice;
        if (site.sublattice == site.foundation->size_n) {
            ++site.index[2];
            if (site.index[2] == slice_index[2].end) {
                ++site.index[1];
                if (site.index[1] == slice_index[1].end) {
                    ++site.index[0];
                    if (site.index[0] == slice_index[0].end) {
                        site.idx = -1;
                        return *this;
                    }
                    site.index[1] = slice_index[1].start;
                }
                site.index[2] = slice_index[2].start;
            }
            site.sublattice = 0;
        }

        site.reset_idx();
        return *this;
    }

    SliceIterator operator++(int) {
        auto const copy = *this;
        this->operator++();
        return copy;
    }

    friend bool operator==(SliceIterator const& l, SliceIterator const& r) {
        return l.site.get_idx() == r.site.get_idx();
    }
    friend bool operator!=(SliceIterator const& l, SliceIterator const& r) { return !(l == r); }
};

} // namespace tbm
