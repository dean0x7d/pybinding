#pragma once
#include "detail/slice.hpp"
#include "numeric/dense.hpp"
#include <vector>

namespace cpb {

class Foundation;

struct SymmetryArea {
    Index3D left, right, middle;

    /// Is this lattice site contained in the symmetry unit cell?
    bool contains(Index3D const& index) const;
};

struct Translation {
    Index3D direction; ///< translation direction in number of unit cell
    SliceIndex3D boundary_slice; ///< Foundation slice which has the boundary sites
    Index3D shift_index; ///< width of a translation unit in lattice sites
    Cartesian shift_lenght; ///< width of a translation unit in nanometers
};

/**
 Translational symmetry

 The constructor takes the translation length in each lattice vector direction.
 A positive number is a valid length. A negative number disables translation in that direction.
 Zero is a special value which automatically sets the minimal translation length for the lattice, 
 i.e. the lattice vector length.
*/
class TranslationalSymmetry {
public:
    TranslationalSymmetry(float a1 = -1, float a2 = -1, float a3 = -1);

    SymmetryArea area(Foundation const& foundation) const;
    std::vector<Translation> translations(Foundation const& foundation) const;
    void apply(Foundation& foundation) const;

    explicit operator bool() const { return enabled_directions != Vector3b{false, false, false}; }

private:
    Cartesian length;
    Vector3b enabled_directions = {false, false, false};
};

namespace detail {
    /// Return all the combinations of enabled directions
    /// Example: [1, 1, 0] -> [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]
    std::vector<Index3D> make_masks(Vector3b enabled_directions, int ndim);
}

} // namespace cpb
