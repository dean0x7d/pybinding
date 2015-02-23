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

bool Circle::contains(const Cartesian& point) const {
    Cartesian x = point - _center;
    return x.norm() < radius;
}

std::vector<Cartesian> Circle::bounding_vectors() const {
    std::vector<Cartesian> bounding_vectors;

    bounding_vectors.emplace_back(0, 2*radius, 0);
    bounding_vectors.emplace_back(2*radius, 0, 0);
    bounding_vectors.emplace_back(0, -2*radius, 0);
    bounding_vectors.emplace_back(-2*radius, 0 , 0);

    return bounding_vectors;
}

bool Polygon::contains(const Cartesian& point) const { 
    // raycasting algorithm that checks if the point is inside this polygon
    bool is_inside = false;
    
    // loop over all the sides (neighbouring points) of the polygon
    for (int i = 0, j = (int)x.size() - 1; i < x.size(); j = i++) { 
        // aliases for readability
        const auto& x1 = x[i]; const auto& x2 = x[j];
        const auto& y1 = y[i]; const auto& y2 = y[j];

        // we shoot the ray along the x direction
        if (y1 > point.y() == y2 > point.y())
            continue; // the ray does not intersect this side of the polygon
        
        // the slope of this side
        float k = (x2-x1) / (y2-y1);
        
        // we shoot the ray from left to right
        // if point.x() is bigger than the point on the side, we crossed the side
        if (point.x() > k*(point.y()-y1) + x1)
            is_inside = !is_inside;
    }

    // returns true for an odd number of side crossings
    return is_inside;
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
        bounding_vectors.emplace_back(x[i] - x[j], y[i] - y[j], 0);
    }

    return bounding_vectors;
}
