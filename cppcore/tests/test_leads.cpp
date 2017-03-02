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

    auto model = Model(lattice::square(), shape::rectangle(width, height));
    REQUIRE(model.system()->num_sites() == 6);

    model.attach_lead(-1, Line({0, -height/2, 0}, {0, height/2, 0}));
    model.attach_lead(+1, Line({0, -height/2, 0}, {0, height/2, 0}));
    REQUIRE(model.leads().size() == 2);

    REQUIRE_THAT(model.lead(0).indices(), Catch::Equals(std::vector<int>{0, 2, 4}));
    REQUIRE_THAT(model.lead(1).indices(), Catch::Equals(std::vector<int>{1, 3, 5}));

    SECTION("Hoppings grow from lead 0 to system") {
        model.add(field::linear_hopping());
        auto const h = matrix_data<>(model.hamiltonian());
        auto const h1 = matrix_data<>(model.lead(0).h1());
        REQUIRE(h1.minCoeff() < h.minCoeff());
    }

    SECTION("Onsite potential grows from system to lead 1") {
        model.add(field::linear_onsite());
        auto const h = matrix_data<>(model.hamiltonian());
        auto const h0 = matrix_data<>(model.lead(1).h0());
        REQUIRE(h.maxCoeff() < h0.maxCoeff());
    }
}
