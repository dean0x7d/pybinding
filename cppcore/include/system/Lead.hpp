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

/// Create a lead attachment area in the foundation
void attach(Foundation& foundation, Lead const& lead);

namespace lead {
    /// Compute the slice through which the lead penetrates the foundation, end to end
    SliceIndex3D passthrough_slice(Foundation const& foundation, Lead const& lead);
    /// Compute the slice where the lead can be attached to the foundation
    SliceIndex3D attachment_slice(Foundation const& foundation, Lead const& lead);
}

} // namespace tbm
