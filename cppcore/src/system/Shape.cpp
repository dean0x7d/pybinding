#include "system/Shape.hpp"

namespace tbm {

Polygon::Polygon(std::vector<Cartesian> const& bounding_points, Cartesian offset)
    : Shape(bounding_points, offset)
{
    auto const size = bounding_points.size();
    x.resize(size);
    y.resize(size);

    for (auto i = 0u; i < size; ++i) {
        x[i] = bounding_points[i].x();
        y[i] = bounding_points[i].y();
    }
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

Circle::Circle(float r, Cartesian c, Cartesian offset)
    : Shape({{-r, -r, 0}, {-r, r, 0}, {r, r, 0}, {r, -r, 0}}, offset), radius(r), center(c) {}

ArrayX<bool> Circle::contains(CartesianArray const& positions) const {
    ArrayX<bool> is_within(positions.size());
    for (auto i = 0; i < positions.size(); ++i) {
        is_within[i] = (positions[i] - center).norm() < radius;
    }
    return is_within;
}

} // namespace tbm
