#include "system/Shape.hpp"

namespace tbm {

Shape::Shape(std::vector<Cartesian> const& bbox_vertices, Cartesian offset)
    : bbox_vertices(bbox_vertices), offset(offset)
{
    if (bbox_vertices.size() < 2)
        throw std::logic_error("Shape: The bounding box must contain at least two vertices.");
}


Polygon::Polygon(std::vector<Cartesian> const& vertices, Cartesian offset)
    : Shape(vertices, offset)
{
    auto const size = vertices.size();
    x.resize(size);
    y.resize(size);

    for (auto i = 0u; i < size; ++i) {
        x[i] = vertices[i].x();
        y[i] = vertices[i].y();
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

FreeformShape::FreeformShape(ContainsFunc contains_func, Cartesian width,
                             Cartesian center, Cartesian offset)
    : Shape({}, offset), contains_func(contains_func)
{
    Cartesian base_vertex = center + 0.5 * width;
    auto const x = base_vertex.x();
    auto const y = base_vertex.y();
    auto const z = base_vertex.z();

    bbox_vertices = {
        {x,   y,  z},
        {-x,  y,  z},
        {x,  -y,  z},
        {-x, -y,  z},
        {x,   y, -z},
        {-x,  y, -z},
        {x,  -y, -z},
        {-x, -y, -z}
    };
}

ArrayX<bool> FreeformShape::contains(CartesianArray const& positions) const {
    return contains_func(positions);
}

} // namespace tbm
