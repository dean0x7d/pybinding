#pragma once
#include <vector>
#include "support/dense.hpp"

namespace tbm {

/**
 Shape of the primitive unit cell
 */
class Primitive {
public:
    Primitive(size_t a1 = 1, size_t a2 = 1, size_t a3 = 1) : size(a1, a2, a3) {}

    Index3D size;
};


/**
 Abstract base shape

 The bounding box (bbox) specifies the maximum area (or volume) where the shape will be
 located. The entire bbox is filled with lattice sites and then the `contains` function
 decides which of those sites are actually located within the desired shape.
 It's like carving a sculpture from a block of stone.
 */
class Shape {
public:
    Shape(std::vector<Cartesian> const& bbox_vertices, Cartesian offset);

    /// Return `true` for `positions` located within the shape
    virtual ArrayX<bool> contains(CartesianArray const& positions) const = 0;

    std::vector<Cartesian> bbox_vertices; ///< bounding box defined by its vertices
    Cartesian offset; ///< offset of the lattice origin from the shape origin
};


/**
 Polygon shape defined by a list of points
 */
class Polygon : public Shape {
public:
    Polygon(std::vector<Cartesian> const& bounding_points, Cartesian offset);

    ArrayX<bool> contains(CartesianArray const& positions) const override;

    ArrayX<float> x, y;
};


/**
 Shape defined by a bounding box and a function
 */
class FreeformShape : public Shape {
public:
    using ContainsFunc = std::function<ArrayX<bool>(CartesianArray const&)>;

    FreeformShape(ContainsFunc contains_func, Cartesian width,
                  Cartesian center = Cartesian::Zero(),
                  Cartesian offset = Cartesian::Zero());

    ArrayX<bool> contains(CartesianArray const& positions) const override;

private:
    ContainsFunc contains_func;
};

} // namespace tbm
