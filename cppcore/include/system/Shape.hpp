#pragma once
#include <vector>
#include "support/dense.hpp"

namespace tbm {

class Lattice;

/**
 Abstract base shape
 */
class Shape {
public:
    /// Return `true` for `positions` located within the shape
    virtual ArrayX<bool> contains(CartesianArray const& positions) const = 0;
    /// Location of the shape center
    virtual Cartesian center() const = 0;
    /// Returns shape outline length in lattice vector directions
    virtual Cartesian length_for(Lattice const& lattice) const;

public:
    bool has_nice_edges = true; ///< the edges should conform to lattice.min_neighbours specification
    
protected:
    virtual std::vector<Cartesian> bounding_vectors() const = 0;
};


/**
 Shape of the primitive unit cell
 */
class Primitive : public Shape {
public:
    Primitive(Cartesian length = Cartesian::Zero(), bool nanometers = false)
        : length{std::move(length)}, nanometers{nanometers} { has_nice_edges = false; }

    virtual ArrayX<bool> contains(CartesianArray const& positions) const final;
    virtual Cartesian center() const final;
    virtual Cartesian length_for(const Lattice& lattice) const final;

protected:
    virtual std::vector<Cartesian> bounding_vectors() const final;

private:
    Cartesian length;
    bool nanometers;
};


/**
 Simple circle defined by radius and center coordinates
 */
class Circle : public Shape {
public:
    Circle(float radius, Cartesian center = Cartesian::Zero())
        : radius{radius}, _center{std::move(center)} {}
    
    virtual ArrayX<bool> contains(CartesianArray const& positions) const final;
    virtual Cartesian center() const final;

protected:
    virtual std::vector<Cartesian> bounding_vectors() const final;

public:
    float radius;
    Cartesian _center;
};


/**
 Polygon shape defined by a list of points
 */
class Polygon : public Shape {
public:
    virtual ArrayX<bool> contains(CartesianArray const& positions) const final;
    virtual Cartesian center() const final;
    
protected:
    virtual std::vector<Cartesian> bounding_vectors() const final;

public:
    ArrayX<float> x, y;
    Cartesian offset;
};
    
} // namespace tbm
