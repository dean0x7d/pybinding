#pragma once
#include "Model.hpp"

namespace graphene {

static constexpr auto a = 0.24595f; // [nm] unit cell length
static constexpr auto a_cc = 0.142f; // [nm] carbon-carbon distance
static constexpr auto t = -2.8f; // [eV] nearest neighbor hopping

tbm::Lattice monolayer();

} // namespace graphene

namespace shape {

tbm::Shape rectangle(float x, float y);

} // namespace shape

namespace field {

tbm::OnsiteModifier constant_potential(float value);
tbm::HoppingModifier constant_magnetic_field(float value);

tbm::HoppingModifier force_double_precision();
tbm::HoppingModifier force_complex_numbers();

} // namespace field
