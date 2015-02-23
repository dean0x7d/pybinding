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
    /// Is the point located within this shape?
    virtual bool contains(const Cartesian& point) const = 0;
    /// Location of the shape center
    virtual Cartesian center() const = 0;
    /// Returns shape outline length in lattice vector directions
    virtual Cartesian length_for(const Lattice& lattice) const;

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

    virtual bool contains(const Cartesian&) const final { return true; }
    virtual Cartesian center() const final { return Cartesian::Zero(); }
    virtual Cartesian length_for(const Lattice& lattice) const override;

protected:
    virtual std::vector<Cartesian> bounding_vectors() const { return {}; }

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
    
    virtual bool contains(const Cartesian& point) const final;
    virtual Cartesian center() const final { return _center; }

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
    virtual bool contains(const Cartesian& point) const final;
    virtual Cartesian center() const final;
    
protected:
    virtual std::vector<Cartesian> bounding_vectors() const final;

public:
    ArrayX<float> x, y;
    Cartesian offset;
};
    
} // namespace tbm
