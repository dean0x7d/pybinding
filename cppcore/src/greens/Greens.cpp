#include "greens/Greens.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "support/physics.hpp"

using namespace tbm;
using physics::pi;


void Greens::set_model(Model const& new_model) {
    model = new_model;
    if (strategy) {
        // try to assign a new Hamiltonian to the existing Green's strategy
        bool success = strategy->set_hamiltonian(model.hamiltonian());
        if (!success) // fails if the they have incompatible scalar types
            strategy.reset();
    }

    // creates a Green's strategy with a scalar type suited to the Hamiltonian
    if (!strategy)
        strategy = create_strategy_for(model.hamiltonian());
}

ArrayXcf Greens::calc_greens(int i, int j, ArrayXf energy, float broadening) {
    auto const size = model.hamiltonian()->rows();
    if (i < 0 || i > size || j < 0 || j > size)
        throw std::logic_error{"KPM::calc_greens(i,j): invalid value for i or j."};

    calculation_timer.tic();
    auto greens_function = strategy->calculate(i, j, energy, broadening);
    calculation_timer.toc();

    return greens_function;
}

ArrayXf Greens::calc_ldos(ArrayXf energy, float broadening,
                          Cartesian position, sub_id sublattice)
{
    auto i = model.system()->find_nearest(position, sublattice);
    auto greens_function = calc_greens(i, i, energy, broadening);

    return -1/pi * greens_function.imag();
}

Deferred<ArrayXf> Greens::deferred_ldos(ArrayXf energy, float broadening,
                                        Cartesian position, sub_id sublattice)
{
    auto shared_strategy = std::shared_ptr<GreensStrategy>{
        create_strategy_for(model.hamiltonian())
    };
    auto& model = this->model;

    return {
        [shared_strategy, model, position, sublattice, energy, broadening](ArrayXf& ldos) {
            auto i = model.system()->find_nearest(position, sublattice);
            auto greens_function = shared_strategy->calculate(i, i, energy, broadening);
            ldos = -1/pi * greens_function.imag();
        },
        [shared_strategy] {
            return shared_strategy->report(true);
        }
    };
}
