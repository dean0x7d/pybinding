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
 */
class Shape {
public:
    Shape(std::vector<Cartesian> const& bounding_points, Cartesian offset)
        : bounding_points(bounding_points), offset(offset) {}

    /// Return `true` for `positions` located within the shape
    virtual ArrayX<bool> contains(CartesianArray const& positions) const = 0;

    std::vector<Cartesian> bounding_points;
    Cartesian offset;
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
 Simple circle defined by radius and center coordinates
 */
class Circle : public Shape {
public:
    Circle(float radius, Cartesian center = Cartesian::Zero(),
           Cartesian offset = Cartesian::Zero());
    
    ArrayX<bool> contains(CartesianArray const& positions) const final;

    float radius;
    Cartesian center;
};

} // namespace tbm
