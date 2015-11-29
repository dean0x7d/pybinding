#include "system/Shape.hpp"
#include "system/Lattice.hpp"

#include <Eigen/Dense>  // for `colPivHouseholderQr()`

namespace tbm {

Cartesian Shape::length_for(Lattice const& lattice) const {
    auto const ndim = lattice.vectors.size();
    auto const lattice_matrix = [&]{
        Eigen::MatrixXf m(ndim, ndim);
        for (auto i = 0u; i < ndim; ++i) {
            m.col(i) = lattice.vectors[i].head(ndim);
        }
        return m;
    }();

    Cartesian length = Cartesian::Zero();
    for (auto const& boundary : bounding_vectors()) {
        // solve `A*x = b`, where A is lattice_matrix
        auto const& b = boundary.head(ndim);
        VectorXf x = lattice_matrix.colPivHouseholderQr().solve(b);
        length.head(ndim) += x.cwiseAbs();
    }
    length *= 0.5f;

    for (auto i = 0u; i < ndim; ++i) {
        length[i] *= lattice.vectors[i].norm();
    }

    return length;
}


ArrayX<bool> Primitive::contains(CartesianArray const& positions) const {
    return ArrayX<bool>::Constant(positions.size(), true);
}

Cartesian Primitive::center() const {
    return Cartesian::Zero();
}

std::vector<Cartesian> Primitive::bounding_vectors() const {
    return {};
}

Cartesian Primitive::length_for(const Lattice& lattice) const {
    if (nanometers) {
        return length;
    }
    else {
        Cartesian length_nm = Cartesian::Zero();
        for (std::size_t i = 0; i < lattice.vectors.size(); ++i) {
            length_nm[i] = length[i] * lattice.vectors[i].norm();
        }
        return length_nm;
    }
}


ArrayX<bool> Circle::contains(CartesianArray const& positions) const {
    ArrayX<bool> is_within(positions.size());
    for (auto i = 0; i < positions.size(); ++i) {
        is_within[i] = (positions[i] - _center).norm() < radius;
    }
    return is_within;
}

Cartesian Circle::center() const {
    return _center;
}

std::vector<Cartesian> Circle::bounding_vectors() const {
    std::vector<Cartesian> bounding_vectors;

    bounding_vectors.emplace_back(.0f, 2*radius, .0f);
    bounding_vectors.emplace_back(2*radius, .0f, .0f);
    bounding_vectors.emplace_back(.0f, -2*radius, .0f);
    bounding_vectors.emplace_back(-2*radius, .0f , .0f);

    return bounding_vectors;
}

ArrayX<bool> Polygon::contains(CartesianArray const& positions) const {
    // Raycasting algorithm checks if `positions` are inside this polygon
    ArrayX<bool> is_within = ArrayX<bool>::Constant(positions.size(), false);

    // Loop over all the sides of the polygon (neighbouring vertices)
    auto const num_vertices = static_cast<int>(x.size());
    for (auto i = 0, j = num_vertices - 1; i < num_vertices; j = i++) {
        // Aliases for readability
        auto const& x1 = x[i]; auto const& x2 = x[j];
        auto const& y1 = y[i]; auto const& y2 = y[j];
        // The slope of this side
        auto const k = (x2 - x1) / (y2 - y1);

        // Shoot the ray along the x direction and see if it passes between `y1` and `y2`
        auto intersects_y = (y1 > positions.y) != (y2 > positions.y);

        // The ray is moving from left to right and may cross a side of the polygon
        auto x_side = k * (positions.y - y1) + x1;
        auto intersects_x = positions.x > x_side;

        // Eigen doesn't support `operator!`, so this will have to do...
        auto negate = is_within.select(
            ArrayX<bool>::Constant(is_within.size(), false),
            ArrayX<bool>::Constant(is_within.size(), true)
        );
        // Flip the states which intersect the side
        is_within = (intersects_y && intersects_x).select(negate, is_within);
    }

    return is_within;
}

Cartesian Polygon::center() const {
    Cartesian center = Cartesian::Zero();
    center.x() = (x.maxCoeff() + x.minCoeff()) / 2;
    center.y() = (y.maxCoeff() + y.minCoeff()) / 2;

    return center + offset;
}

std::vector<Cartesian> Polygon::bounding_vectors() const {
    std::vector<Cartesian> bounding_vectors;

    // loop over all sides of the polygon
    for (int i = 0, j = (int)x.size()-1; i < x.size(); j = i++) {
        bounding_vectors.emplace_back(x[i] - x[j], y[i] - y[j], .0f);
    }

    return bounding_vectors;
}

} // namespace tbm
