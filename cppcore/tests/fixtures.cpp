#include "fixtures.hpp"
using namespace cpb;

namespace lattice {

Lattice square(float a, float t) {
    auto lattice = Lattice({a, 0, 0}, {0, a, 0});

    lattice.add_sublattice("A", {0, 0, 0}, 4 * t);

    lattice.register_hopping_energy("-t", -t);
    lattice.add_hopping({0, 1, 0}, "A", "A", "-t");
    lattice.add_hopping({1, 0, 0}, "A", "A", "-t");

    return lattice;
}

Lattice square_2atom(float a, float t1, float t2) {
    auto lattice = Lattice({a, 0, 0}, {0, a, 0});

    lattice.add_sublattice("A", {0, 0, 0});
    lattice.add_sublattice("B", {0.5f * a, 0.5f * a, 0});

    lattice.register_hopping_energy("t1", t1);
    lattice.register_hopping_energy("t2", t2);

    lattice.add_hopping({0, 0, 0}, "A", "B", "t1");
    lattice.add_hopping({1, 1, 0}, "A", "B", "t1");
    lattice.add_hopping({1, 0, 0}, "A", "A", "t2");

    return lattice;
}

Lattice square_multiorbital() {
    auto lattice = Lattice({1, 0, 0}, {0, 1, 0});

    lattice.add_sublattice("A", {0, 0, 0}, VectorXd::Constant(2, 0.0).eval());
    lattice.add_sublattice("B", {0, 0, 0}, VectorXd::Constant(1, 0.0).eval());
    lattice.add_sublattice("C", {0, 0, 0}, VectorXd::Constant(2, 0.0).eval());
    lattice.add_sublattice("D", {0, 0, 0}, VectorXd::Constant(3, 0.0).eval());

    lattice.register_hopping_energy("t22", MatrixXcd::Constant(2, 2, 1.0));
    lattice.register_hopping_energy("t12", MatrixXcd::Constant(1, 2, 1.0));
    lattice.register_hopping_energy("t13", MatrixXcd::Constant(1, 3, 1.0));
    lattice.register_hopping_energy("t23", MatrixXcd::Constant(2, 3, 1.0));
    lattice.register_hopping_energy("t32", MatrixXcd::Constant(3, 2, 1.0));

    lattice.add_hopping({0, 0, 0}, "A", "C", "t22");
    lattice.add_hopping({0, 0, 0}, "B", "A", "t12");
    lattice.add_hopping({1, 0, 0}, "B", "D", "t13");
    lattice.add_hopping({1, 0, 0}, "A", "A", "t22");
    lattice.add_hopping({0, 0, 0}, "C", "D", "t23");
    lattice.add_hopping({0, 1, 0}, "D", "A", "t32");

    return lattice;
}

Lattice checkerboard_multiorbital() {
    constexpr auto i1 = num::get_complex_t<double>{constant::i1};

    auto lattice = Lattice({1, 0, 0}, {0, 1, 0});
    // complex multi-orbital hopping and complex onsite energy
    auto hopping = MatrixXcd(2, 2);
    hopping << 2.0 + 2.0 * i1, 3.0 + 3.0 * i1, 4.0 + 4.0 * i1, 5.0 + 5.0 * i1;
    auto delta = MatrixXcd(2, 2);
    delta << 1.0, i1, -i1, 1.0;

    lattice.add_sublattice("A", {  0,   0, 0}, (-delta).eval());
    lattice.add_sublattice("B", {0.5, 0.5, 0}, delta);

    lattice.register_hopping_energy("t", hopping);

    lattice.add_hopping({ 0,  0, 0}, "A", "B", "t");
    lattice.add_hopping({ 0, -1, 0}, "A", "B", "t");
    lattice.add_hopping({-1,  0, 0}, "A", "B", "t");
    lattice.add_hopping({-1, -1, 0}, "A", "B", "t");

    return lattice;
}

Lattice hexagonal_complex() {
    constexpr auto i1 = num::get_complex_t<double>{constant::i1};
    // lattice vectors
    auto a1 = Cartesian{ 0.5f, 0.5f * sqrt(3.0f), 0};
    auto a2 = Cartesian{-0.5f, 0.5f * sqrt(3.0f), 0};
    // positions
    auto const pos_a = Cartesian{0,                       0, 0};
    auto const pos_b = Cartesian{0, -1.0f/3 * sqrt(3.0f), 0};

    auto lattice = Lattice(a1, a2);

    lattice.add_sublattice("A", pos_a);
    lattice.add_sublattice("B", pos_b);
    // complex hoppings
    lattice.register_hopping_energy("t1", -i1);
    lattice.register_hopping_energy("t2", 2.0 * i1);
    lattice.register_hopping_energy("t3", 3.0 * i1);

    lattice.add_hopping({0, 0, 0}, "A", "B", "t1");
    lattice.add_hopping({0, 1, 0}, "A", "B", "t2");
    lattice.add_hopping({1, 0, 0}, "A", "B", "t3");

    return lattice;
}

} // namespace lattice

namespace graphene {

Lattice monolayer() {
    auto lattice = Lattice({a, 0, 0}, {a/2, a/2 * sqrt(3.0f), 0});

    lattice.add_sublattice("A", {0, -a_cc/2, 0});
    lattice.add_sublattice("B", {0,  a_cc/2, 0});

    lattice.register_hopping_energy("t", t);
    lattice.add_hopping({0,  0, 0}, "A", "B", "t");
    lattice.add_hopping({1, -1, 0}, "A", "B", "t");
    lattice.add_hopping({0, -1, 0}, "A", "B", "t");

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
    return {[value](ComplexArrayRef energy, CartesianArrayConstRef, string_view) {
        num::match<ArrayX>(energy, OnsiteEnergyOp{value});
    }};
}

namespace {
    struct MagneticFieldOp {
        float magnitude;
        CartesianArrayConstRef pos1;
        CartesianArrayConstRef pos2;

        static constexpr auto scale = 1e-18f;

        template<class Array>
        void operator()(Array) const {}

        template<class real_t>
        void operator()(Map<ArrayX<std::complex<real_t>>> energy) const {
            using scalar_t = std::complex<real_t>;
            auto const k = static_cast<scalar_t>(scale * 2 * constant::pi / constant::phi0);
            auto const vp_x = 0.5f * magnitude * (pos1.y() + pos2.y());
            auto const peierls = vp_x * (pos1.x() - pos2.x());
            energy *= exp(scalar_t{constant::i1} * k * peierls.template cast<scalar_t>());
        }
    };
}

cpb::HoppingModifier constant_magnetic_field(float value) {
    return {[value](ComplexArrayRef energy, CartesianArrayConstRef pos1,
                    CartesianArrayConstRef pos2, string_view) {
        num::match<ArrayX>(energy, MagneticFieldOp{value, pos1, pos2});
    }, /*is_complex*/true, /*is_double*/false};
}

namespace {
    struct LinearOnsite {
        float k;
        Eigen::Ref<ArrayXf const> x;

        template<class Array>
        void operator()(Array energy) const {
            using scalar_t = typename Array::Scalar;
            energy = (k * x).template cast<scalar_t>();
        }
    };
}

cpb::OnsiteModifier linear_onsite(float k) {
    return {[k](ComplexArrayRef energy, CartesianArrayConstRef pos, string_view) {
        num::match<ArrayX>(energy, LinearOnsite{k, pos.x()});
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
    return {[k](ComplexArrayRef energy, CartesianArrayConstRef pos1,
                CartesianArrayConstRef pos2, string_view) {
        num::match<ArrayX>(energy, LinearHopping{k, 0.5f * (pos1.x() + pos2.x())});
    }, /*is_complex*/false, /*is_double*/false};
}

cpb::HoppingModifier force_double_precision() {
    auto nop = [](ComplexArrayRef, CartesianArrayConstRef, CartesianArrayConstRef, string_view) {};
    return cpb::HoppingModifier(nop, /*is_complex*/false, /*is_double*/true);
}

cpb::HoppingModifier force_complex_numbers() {
    auto nop = [](ComplexArrayRef, CartesianArrayConstRef, CartesianArrayConstRef, string_view) {};
    return cpb::HoppingModifier(nop, /*is_complex*/true, /*is_double*/false);
}

} // namespace field


namespace generator {

cpb::HoppingGenerator do_nothing_hopping(std::string const& name) {
    return {name, 0.0, [](System const&) {
        return HoppingGenerator::Result{ArrayXi{}, ArrayXi{}};
    }};
}

} // namespace generator
