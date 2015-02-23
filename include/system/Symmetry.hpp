#pragma once
#include <vector>
#include "support/dense.hpp"

namespace tbm {

class Lattice;
class Foundation;

// TODO: this only works for translational symmetry
struct SymmetrySpec {
    /// Describes a symmetry translation
    struct Translation {
        Index3D direction; ///< translation direction in number of unit cell
        Index3D boundary; ///< indices of the boundary sites - to be used with for_sites method
        Index3D shift_index; ///< width of a translation unit in lattice sites
        Cartesian shift_lenght; ///< width of a translation unit in nanometers
    };

    /// Is this lattice site contained in the symmetry unit cell
    virtual bool contains(const Index3D& index) const;

    /// Add a translation in the relative_index directions
    void add_translation(const Index3D& relative_index, const Lattice& lattice);

    Index3D left, right, middle;
    std::vector<Translation> translations;
};


/**
 Abstract base symmetry.
 */
class Symmetry {
public:
    /// Build the symmetry for the given lattice and foundation size
    virtual SymmetrySpec build_for(const Foundation& foundation) const = 0;
};


/**
 Simple translational symmetry.
 The constructor takes the translation length in each lattice vector direction.
 A positive number is a valid length. A negative number disables translation in that direction.
 Zero is a special value which automatically sets the minimal translation length for the lattice, 
 i.e. the lattice vector length.
*/
class Translational : public Symmetry {
public:
    Translational(Cartesian length) : length{length} {}

    virtual SymmetrySpec build_for(const Foundation& foundation) const override;

private:
    Cartesian length = Cartesian::Zero();
};

} // namespace tbm
