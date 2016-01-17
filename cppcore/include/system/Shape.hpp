#pragma once
#include "support/dense.hpp"
#include <vector>
#include <functional>

namespace tbm {

/**
 Shape of the primitive unit cell
 */
class Primitive {
public:
    Primitive(int a1 = 1, int a2 = 1, int a3 = 1);

    Index3D size;
};


/**
 Shape defined by bounding vertices and `contains` function

 The bounding vertices specify the maximum area (or volume) where the shape will be located.
 The entire volume is filled with lattice sites and then the `contains` function decides which
 of those sites are actually located within the desired shape. It's like carving a sculpture
 from a block of stone.
 */
class Shape {
public:
    using Vertices = std::vector<Cartesian>;
    using Contains = std::function<ArrayX<bool>(CartesianArray const&)>;

    Shape() = default;
    Shape(Vertices const& vertices, Contains const& contains, Cartesian offset);

    /// A shape is valid if it has a `contains` function
    explicit operator bool() const { return static_cast<bool>(contains); }

    Vertices vertices; ///< bounding vertices which define the initial volume
    Contains contains; ///< return `true` for `positions` located within the shape
    Cartesian offset; ///< offset of the lattice origin from the shape origin
};

/**
 1D line
 */
class Line : public Shape {
public:
    Line(Cartesian a, Cartesian b, Cartesian offset = Cartesian::Zero());
};

/**
 Polygon shape defined by a list of points

 Strictly 2D within the xy plane.
 */
class Polygon : public Shape {
public:
    Polygon(Vertices const& vertices, Cartesian offset = Cartesian::Zero());
};

/**
 Shape defined by a bounding box and a function
 */
class FreeformShape : public Shape {
public:
    FreeformShape(Contains const& contains, Cartesian width,
                  Cartesian center = Cartesian::Zero(),
                  Cartesian offset = Cartesian::Zero());
};

namespace detail {
    // Is the angle formed by three points acute? The vertex is `b`.
    ArrayX<bool> is_acute_angle(Cartesian a, Cartesian b, CartesianArray const& c);

    /// Function object which determines if a point is within a polygon
    class WithinPolygon {
    public:
        WithinPolygon(Shape::Vertices const& vertices);
        ArrayX<bool> operator()(CartesianArray const& positions) const;

    private:
        ArrayX<float> x, y;
    };
} // namespace detail

} // namespace tbm
