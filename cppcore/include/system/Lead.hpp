#pragma once
#include "system/Shape.hpp"

#include "detail/slice.hpp"
#include "support/dense.hpp"

namespace tbm {

class Foundation;

class Lead {
public:
    Lead(int direction, Shape const& shape);

public:
    int direction;
    Shape shape;
};

void attach(Foundation& foundation, Lead const& lead);

namespace lead {
    /// Compute the slice through which the lead penetrates the foundation
    SliceIndex3D slice_index(Foundation const& foundation, Lead const& lead);
}

} // namespace tbm
