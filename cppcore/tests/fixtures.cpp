#include "fixtures.hpp"
using namespace tbm;

namespace graphene {

Lattice monolayer() {
    auto lattice = Lattice({a, 0, 0}, {a/2, a/2 * sqrt(3.0f), 0});

    auto const A = lattice.add_sublattice("A", {0, -a_cc/2, 0});
    auto const B = lattice.add_sublattice("B", {0,  a_cc/2, 0});
    auto const t0 = lattice.register_hopping_energy("t", t);

    lattice.add_registered_hopping({0,  0, 0}, A, B, t0);
    lattice.add_registered_hopping({1, -1, 0}, A, B, t0);
    lattice.add_registered_hopping({0, -1, 0}, A, B, t0);

    return lattice;
}

} // namespace graphene

namespace shape {

Shape rectangle(float x, float y) {
    auto const x0 = x / 2;
    auto const y0 = y / 2;
    return Polygon({{x0, y0, 0}, {x0, -y0, 0}, {-x0, -y0, 0}, {-x0, y0, 0}});
}

} // namespace shape

namespace field {

namespace {
    struct OnsiteEnergyOp {
        float value;

        template<class Array>
        void operator()(Array energy) {
            using scalar_t = typename Array::Scalar;
            energy.setConstant(static_cast<scalar_t>(value));
        }
    };
}

tbm::OnsiteModifier constant_potential(float value) {
    return {[value](ComplexArrayRef energy, CartesianArray const&, SubIdRef) {
        num::match<ArrayX>(energy, OnsiteEnergyOp{value});
    }};
}

} // namespace field
