#include "system/Shape.hpp"
#include "system/Lattice.hpp"
using namespace tbm;

Cartesian Shape::length_for(const Lattice& lattice) const {
    auto get_measure_2D = [](const std::vector<Cartesian>& vec) -> std::vector<Cartesian> {
        float ab = vec[0].dot(vec[1]);
        float a2 = vec[0].squaredNorm();
        float b2 = vec[1].squaredNorm();
        float divisor = pow(ab, 2) - a2 * b2;

        Cartesian measure_a = (vec[1] * ab - vec[0] * b2) / divisor;
        Cartesian measure_b = (vec[0] * ab - vec[1] * a2) / divisor;

        return {measure_a, measure_b};
    };

    std::vector<Cartesian> measure;
    if (lattice.vectors.size() == 1)
        measure = {lattice.vectors[0]};
    else if (lattice.vectors.size() == 2)
        measure = get_measure_2D(lattice.vectors);
    else
        throw std::runtime_error{"3D systems have not been enabled yet."};

    Cartesian projection = Cartesian::Zero();
    for (std::size_t i = 0; i < lattice.vectors.size(); ++i) {
        // get a projection of each bounding vector onto the measure vectors
        for (const Cartesian& bound : bounding_vectors())
            projection[i] += std::abs(bound.dot(measure[i]));

        // we went around the whole shape, but we only need half
        projection[i] *= 0.5;
        projection[i] *= lattice.vectors[i].norm();
    }

    return projection;
}


void Primitive::contains(ArrayX<bool>& is_valid, CartesianArray const&) const {
    is_valid.setConstant(true);
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


void Circle::contains(ArrayX<bool>& is_valid, CartesianArray const& positions) const {
    for (auto i = 0; i < positions.size(); ++i) {
        is_valid[i] = (positions[i] - _center).norm() < radius;
    }
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

void Polygon::contains(ArrayX<bool>& is_valid, CartesianArray const& positions) const {
    // Raycasting algorithm checks if `positions` are inside this polygon
    is_valid.setConstant(false);

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
        auto negate = is_valid.select(
            ArrayX<bool>::Constant(is_valid.size(), false),
            ArrayX<bool>::Constant(is_valid.size(), true)
        );
        // Flip the states which intersect the side
        is_valid = (intersects_y && intersects_x).select(negate, is_valid);
    }
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
