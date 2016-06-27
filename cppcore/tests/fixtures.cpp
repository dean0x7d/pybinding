#include "fixtures.hpp"
#include "numeric/constant.hpp"
using namespace cpb;

namespace lattice {

cpb::Lattice square(float a, float t) {
    auto lattice = Lattice({a, 0, 0}, {0, a, 0});
    auto const subid = lattice.add_sublattice("A", {0, 0, 0}, 4 * t);
    auto const hopid = lattice.register_hopping_energy("-t", -t);
    lattice.add_registered_hopping({0, 1, 0}, subid, subid, hopid);
    lattice.add_registered_hopping({1, 0, 0}, subid, subid, hopid);
    return lattice;
}

} // namespace lattice

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
        void operator()(Array energy) const {
            using scalar_t = typename Array::Scalar;
            energy.setConstant(static_cast<scalar_t>(value));
        }
    };
}

cpb::OnsiteModifier constant_potential(float value) {
    return {[value](ComplexArrayRef energy, CartesianArray const&, SubIdRef) {
        num::match<ArrayX>(energy, OnsiteEnergyOp{value});
    }};
}

namespace {
    struct MagneticFieldOp {
        float magnitude;
        CartesianArray const& pos1;
        CartesianArray const& pos2;

        static constexpr auto scale = 1e-18f;

        template<class Array>
        void operator()(Array) const {}

        template<class real_t>
        void operator()(Map<ArrayX<std::complex<real_t>>> energy) const {
            using scalar_t = std::complex<real_t>;
            auto const k = static_cast<scalar_t>(scale * 2 * constant::pi / constant::phi0);
            auto const vp_x = 0.5f * magnitude * (pos1.y + pos2.y);
            auto const peierls = vp_x * (pos1.x - pos2.x);
            energy *= exp(scalar_t{constant::i1} * k * peierls.template cast<scalar_t>());
        }
    };
}

cpb::HoppingModifier constant_magnetic_field(float value) {
    return {[value](ComplexArrayRef energy, CartesianArray const& pos1,
                    CartesianArray const& pos2, HopIdRef) {
        num::match<ArrayX>(energy, MagneticFieldOp{value, pos1, pos2});
    }, /*is_complex*/true, /*is_double*/false};
}

namespace {
    struct LinearOnsite {
        float k;
        ArrayXf x;

        template<class Array>
        void operator()(Array energy) const {
            using scalar_t = typename Array::Scalar;
            energy = (k * x).template cast<scalar_t>();
        }
    };
}

cpb::OnsiteModifier linear_onsite(float k) {
    return {[k](ComplexArrayRef energy, CartesianArray const& pos, SubIdRef) {
        num::match<ArrayX>(energy, LinearOnsite{k, pos.x});
    }};
}

namespace {
    struct LinearHopping {
        float k;
        ArrayXf x;

        template<class Array>
        void operator()(Array energy) const {
            using scalar_t = typename Array::Scalar;
            energy = (k * x).template cast<scalar_t>();
        }
    };
}

cpb::HoppingModifier linear_hopping(float k) {
    return {[k](ComplexArrayRef energy, CartesianArray const& pos1,
                CartesianArray const& pos2, HopIdRef) {
        num::match<ArrayX>(energy, LinearHopping{k, 0.5f * (pos1.x + pos2.x)});
    }, /*is_complex*/false, /*is_double*/false};
}

cpb::HoppingModifier force_double_precision() {
    auto nop = [](ComplexArrayRef, CartesianArray const&, CartesianArray const&, HopIdRef) {};
    return cpb::HoppingModifier(nop, /*is_complex*/false, /*is_double*/true);
}

cpb::HoppingModifier force_complex_numbers() {
    auto nop = [](ComplexArrayRef, CartesianArray const&, CartesianArray const&, HopIdRef) {};
    return cpb::HoppingModifier(nop, /*is_complex*/true, /*is_double*/false);
}

} // namespace field
