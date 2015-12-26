#include "system/Shape.hpp"
#include "system/Lattice.hpp"

#include <Eigen/Dense>  // for `colPivHouseholderQr()`

namespace tbm {

FS Polygon::foundation_size(Lattice const& lattice) const {
    auto const ndim = lattice.vectors.size();
    auto const lattice_matrix = [&]{
        Eigen::MatrixXf m(ndim, ndim);
        for (auto i = 0u; i < ndim; ++i) {
            m.col(i) = lattice.vectors[i].head(ndim);
        }
        return m;
    }();

    auto const num_sides = static_cast<int>(x.size());
    Cartesian const center(x.sum() / num_sides, y.sum() / num_sides, .0f);

    Cartesian mul = Cartesian::Zero();
    Cartesian offset = Cartesian::Zero();

    // loop over all sides of the polygon
    for (int i = 0, j = num_sides - 1; i < num_sides; j = i++) {
        Cartesian const side_vector(x[i] - x[j], y[i] - y[j], .0f);
        Cartesian const side_center((x[i] + x[j]) / 2, (y[i] + y[j]) / 2, .0f);

        // solve `A*x = b`, where A is lattice_matrix
        auto const& b = side_vector.head(ndim);
        VectorXf x = lattice_matrix.colPivHouseholderQr().solve(b);
        mul.head(ndim) += x.cwiseAbs();

        Cartesian t = Cartesian::Zero();
        for (auto n = 0u; n < ndim; ++n) {
            t += abs(x[n]) * lattice.vectors[n].cwiseAbs();
        }
        t.head(ndim) -= b.cwiseAbs();

        auto diff = side_center.array() > center.array();
        offset.head(ndim) += diff.select(t, -t);
    }
    mul *= 0.5f;
    offset *= 0.25f;

    Index3D size = Index3D::Constant(1);
    for (auto i = 0u; i < ndim; ++i) {
        // integer number of lattice vectors, plus one site (fencepost error otherwise)
        size[i] = static_cast<int>(std::ceil(mul[i])) + 1;
        // make sure it's an odd number, so that (size - 1) / 2 is an integer
        size[i] += !(size[i] % 2);
    }

    return {size, offset};
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

Circle::Circle(float radius, Cartesian center)
: radius{radius}, _center{center} {
    x.resize(4);
    x << .0f, 2 * radius, .0f, -2 * radius;
    y.resize(4);
    y << 2 * radius, .0f, -2 * radius, .0f;
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

} // namespace tbm
