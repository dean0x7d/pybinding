#include "hamiltonian/HamiltonianModifiers.hpp"

namespace cpb {

bool HamiltonianModifiers::any_complex() const {
    const auto complex_potential = std::any_of(
        onsite.begin(), onsite.end(), [](OnsiteModifier const& o) { return o.is_complex; }
    );
    auto const complex_hoppings = std::any_of(
        hopping.begin(), hopping.end(), [](HoppingModifier const& h) { return h.is_complex; }
    );
    return complex_potential || complex_hoppings;
}

bool HamiltonianModifiers::any_double() const {
    auto const double_potential = std::any_of(
        onsite.begin(), onsite.end(), [](OnsiteModifier const& o) { return o.is_double; }
    );
    auto const double_hoppings = std::any_of(
        hopping.begin(), hopping.end(), [](HoppingModifier const& h) { return h.is_double; }
    );
    return double_potential || double_hoppings;
}

void HamiltonianModifiers::clear() {
    onsite.clear();
    hopping.clear();
}

} // namespace cpb
