#include <catch.hpp>

#include "fixtures.hpp"
using namespace cpb;

/// Return the data array of a Hamiltonian CSR matrix
template<class scalar_t = float>
ArrayXf matrix_data(Hamiltonian const& h) {
    auto const matrix = ham::get_reference<scalar_t>(h);
    return Eigen::Map<ArrayXf const>(matrix.valuePtr(), matrix.nonZeros());
}

TEST_CASE("Attach leads") {
    auto const width = 2.0f;
    auto const height = 3.0f;

    auto model = Model(lattice::square());
    model.set_shape(shape::rectangle(width, height));
    REQUIRE(model.system()->num_sites() == 6);

    model.attach_lead(-1, Line({0, -height/2, 0}, {0, height/2, 0}));
    model.attach_lead(+1, Line({0, -height/2, 0}, {0, height/2, 0}));
    REQUIRE(model.leads().size() == 2);

    auto const indices0 = std::vector<int>{0, 1, 2};
    REQUIRE(model.lead(0).indices() == indices0);
    auto const indices1 = std::vector<int>{3, 4, 5};
    REQUIRE(model.lead(1).indices() == indices1);

    SECTION("Hoppings grow from lead 0 to system") {
        model.add_hopping_modifier(field::linear_hopping());
        auto const h = matrix_data<>(model.hamiltonian());
        auto const h1 = matrix_data<>(model.lead(0).h1());
        REQUIRE(h1.minCoeff() < h.minCoeff());
    }

    SECTION("Onsite potential grows from system to lead 1") {
        model.add_onsite_modifier(field::linear_onsite());
        auto const h = matrix_data<>(model.hamiltonian());
        auto const h0 = matrix_data<>(model.lead(1).h0());
        REQUIRE(h.maxCoeff() < h0.maxCoeff());
    }
}
