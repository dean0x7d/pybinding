#include "system/Shape.hpp"

namespace cpb {

Primitive::Primitive(int a1, int a2, int a3) : size(a1, a2, a3) {
    if (any_of(size.array() <= 0)) {
        throw std::logic_error("Primitive: The size must be at least 1 in every direction.");
    }
}

Shape::Shape(Vertices const& vertices, Contains const& contains)
    : vertices(vertices), contains(contains) {
    if (vertices.size() < 2)
        throw std::logic_error("Shape: The bounding box must contain at least two vertices.");
}

Line::Line(Cartesian a, Cartesian b) : Shape({a, b}) {
    contains = [a, b](CartesianArray const& positions) -> ArrayX<bool> {
        // Return `true` for all `positions` which are in the perpendicular space
        // between the two end points of the line
        return detail::is_acute_angle(a, b, positions)
               && detail::is_acute_angle(b, a, positions);
    };
}

namespace detail {

ArrayX<bool> is_acute_angle(Cartesian a, Cartesian b, CartesianArray const& c) {
    // Vectors BA and BC which make the angle
    auto const ba = Cartesian{a - b};
    auto const bc_x = ArrayXf{c.x - b.x()};
    auto const bc_y = ArrayXf{c.y - b.y()};
    auto const bc_z = ArrayXf{c.z - b.z()};

    // Compute the cosine between the two vectors based on the dot product
    auto const ba_dot_bc = ba.x() * bc_x + ba.y() * bc_y + ba.z() * bc_z;
    auto const ba_length = ba.norm();
    auto const bc_length = sqrt(bc_x.cwiseAbs2() + bc_y.cwiseAbs2() + bc_z.cwiseAbs2());
    auto const cos_theta = ba_dot_bc / (ba_length * bc_length);

    return cos_theta >= 0; // acute angle
};

WithinPolygon::WithinPolygon(Shape::Vertices const& vertices)
    : x(vertices.size()), y(vertices.size()) {
    for (auto i = size_t{0}, size = vertices.size(); i < size; ++i) {
        x[i] = vertices[i].x();
        y[i] = vertices[i].y();
    }
}

ArrayX<bool> WithinPolygon::operator()(CartesianArray const& positions) const {
    // Raycasting algorithm checks if `positions` are inside this polygon
    ArrayX<bool> is_within = ArrayX<bool>::Constant(positions.size(), false);

    // Loop over all the sides of the polygon (neighbouring vertices)
    auto const num_vertices = static_cast<int>(x.size());
    for (auto i = 0, j = num_vertices - 1; i < num_vertices; j = i++) {
        // Aliases for readability
        auto const& x1 = x[i]; auto const& x2 = x[j];
        auto const& y1 = y[i]; auto const& y2 = y[j];

        // Check if ray is parallel to this side of the polygon
        if (num::approx_equal(y1, y2)) {
            continue; // avoid division by zero in the next step
        }

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

} // namespace detail

Polygon::Polygon(Vertices const& vertices)
    : Shape(vertices, detail::WithinPolygon(vertices)) {}

namespace {

Shape::Vertices make_freeformshape_vertices(Cartesian width, Cartesian center) {
    auto const v1 = static_cast<Cartesian>(center - 0.5f * width);
    auto const v2 = static_cast<Cartesian>(center + 0.5f * width);
    return {
        {v1.x(), v1.y(), v1.z()},
        {v2.x(), v1.y(), v1.z()},
        {v1.x(), v2.y(), v1.z()},
        {v2.x(), v2.y(), v1.z()},
        {v1.x(), v1.y(), v2.z()},
        {v2.x(), v1.y(), v2.z()},
        {v1.x(), v2.y(), v2.z()},
        {v2.x(), v2.y(), v2.z()}
    };
}

} // anonymous namespace

FreeformShape::FreeformShape(Contains const& contains, Cartesian width, Cartesian center)
    : Shape(make_freeformshape_vertices(width, center), contains) {}

} // namespace cpb
