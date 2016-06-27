#pragma once
#include "system/Lattice.hpp"

#include <functional>
#include <string>

namespace cpb {

/**
 Introduces a new hopping family (with new hop_id) via a list of index pairs

 This can be used to create new hoppings independent of the main Lattice definition.
 It's especially useful for creating additional local hoppings, e.g. to model defects.
 */
class HoppingGenerator {
public:
    /// Site index pairs which should form new hoppings
    struct Result {
        ArrayXi from;
        ArrayXi to;
    };
    using Function = std::function<Result(cpb::CartesianArray const&, SubIdRef)>;

    std::string name; ///< friendly hopping identifier - will be added to lattice registry
    std::complex<double> energy; ///< hopping energy - also added to lattice registry
    Function make; ///< function which will generate the new hopping index pairs

    HoppingGenerator(std::string const& name, std::complex<double> energy, Function const& make)
        : name(name), energy(energy), make(make) {}

    explicit operator bool() const { return static_cast<bool>(make); }
};

using HoppingGenerators = std::vector<HoppingGenerator>;

} // namespace cpb
