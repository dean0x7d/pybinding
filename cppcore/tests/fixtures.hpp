#pragma once
#include "Model.hpp"

namespace lattice {

cpb::Lattice square(float a = 1.f, float t = 1.f);

} // namespace lattice

namespace graphene {

static constexpr auto a = 0.24595f; // [nm] unit cell length
static constexpr auto a_cc = 0.142f; // [nm] carbon-carbon distance
static constexpr auto t = -2.8f; // [eV] nearest neighbor hopping

cpb::Lattice monolayer();

} // namespace graphene

namespace shape {

cpb::Shape rectangle(float x, float y);

} // namespace shape

namespace field {

cpb::OnsiteModifier constant_potential(float value);
cpb::HoppingModifier constant_magnetic_field(float value);

cpb::OnsiteModifier linear_onsite(float k = 1.f);
cpb::HoppingModifier linear_hopping(float k = 1.f);

cpb::HoppingModifier force_double_precision();
cpb::HoppingModifier force_complex_numbers();

} // namespace field
