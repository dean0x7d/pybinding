#pragma once
#include <vector>
#include "support/dense.hpp"

namespace tbm {

class Lattice;

struct FS {
    Index3D size;
    Cartesian offset;
};

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
    /// Return `true` for `positions` located within the shape
    virtual ArrayX<bool> contains(CartesianArray const& positions) const = 0;
    /// Location of the shape center
    virtual Cartesian center() const = 0;
    /// Return the foundation size required to hold the shape
    virtual FS foundation_size(Lattice const& lattice) const = 0;
};


/**
 Polygon shape defined by a list of points
 */
class Polygon : public Shape {
public:
    ArrayX<bool> contains(CartesianArray const& positions) const override;
    Cartesian center() const override;
    FS foundation_size(Lattice const& lattice) const override;

public:
    ArrayX<float> x, y;
    Cartesian offset;
};


/**
 Simple circle defined by radius and center coordinates
 */
class Circle : public Polygon {
public:
    Circle(float radius, Cartesian center = Cartesian::Zero());
    
    ArrayX<bool> contains(CartesianArray const& positions) const final;
    Cartesian center() const final;

public:
    float radius;
    Cartesian _center;
};

} // namespace tbm
