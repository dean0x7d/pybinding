#include "greens/Greens.hpp"
#include "Model.hpp"
using namespace tbm;

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
