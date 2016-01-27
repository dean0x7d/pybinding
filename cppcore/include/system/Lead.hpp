#pragma once
#include "system/Shape.hpp"

#include "detail/slice.hpp"
#include "support/dense.hpp"

#include <vector>

namespace tbm {

class Foundation;

/**
 Lead specification

 The direction parameter needs to be one of: 1, 2, 3, -1, -2, -3.
 The number indicates the lattice vector along which the lead is placed. It mustn't be
 bigger than the number of lattice vectors. The sign indicates if the lead with go with
 or opposite the given lattice vector.
 */
struct Lead {
    int axis; ///< the crystal axis of this lead (lattice vector direction)
    int sign; ///< +1 or -1: with or opposite the axis direction
    Shape shape; ///< determines the attachment area with the main system

    Lead(int direction, Shape const& shape);
};

using Leads = std::vector<Lead>;

/// Describes the area where the lead should be attached to the foundation
struct LeadJunction {
    SliceIndex3D slice_index; ///< slice of the foundation where the lead can be attached
    ArrayX<bool> is_valid; ///< valid lead sites within the slice (determined from the lead shape)

    LeadJunction(Foundation const& foundation, Lead const& lead);
};

/// Create a lead attachment area in the foundation
void attach(Foundation& foundation, Lead const& lead);

namespace lead {
    /// Return the slice of the foundation which contains the shape
    SliceIndex3D shape_slice(Foundation const& foundation, Shape const& shape);
    /// Compute the slice where the lead can be attached to the foundation
    SliceIndex3D attachment_slice(Foundation const& foundation, Lead const& lead);
}

} // namespace tbm
