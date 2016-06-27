#pragma once
#include "system/Shape.hpp"

#include "detail/slice.hpp"
#include "numeric/dense.hpp"

namespace cpb {

class Foundation;

namespace leads {

/**
 Lead specification

 The direction parameter needs to be one of: 1, 2, 3, -1, -2, -3.
 The number indicates the lattice vector along which the lead is placed. It mustn't be
 bigger than the number of lattice vectors. The sign indicates if the lead with go with
 or opposite the given lattice vector.
 */
struct Spec {
    int axis; ///< the crystal axis of this lead (lattice vector direction)
    int sign; ///< +1 or -1: with or opposite the axis direction
    Shape shape; ///< determines the attachment area with the main system

    Spec(int direction, Shape const& shape);
};

/**
 Create a lead attachment area in the foundation
 */
void create_attachment_area(Foundation& foundation, Spec const& spec);

namespace detail {
    /// Return the slice of the foundation which contains the shape
    SliceIndex3D shape_slice(Foundation const& foundation, Shape const& shape);
    /// Compute the slice where the lead can be attached to the foundation
    SliceIndex3D attachment_slice(Foundation const& foundation, Spec const& spec);

    /// Describes the area where the lead should be attached to the foundation
    struct Junction {
        SliceIndex3D slice_index; ///< slice of the foundation where the lead can be attached
        ArrayX<bool> is_valid; ///< valid lead sites within the slice (determined from lead shape)

        Junction(Foundation const& foundation, Spec const& spec);
    };
} // namespace detail

}} // namespace cpb::leads
