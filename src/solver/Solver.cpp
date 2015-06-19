#include "solver/Solver.hpp"
#include "Model.hpp"
using namespace tbm;

void Solver::set_model(const std::shared_ptr<const Model>& new_model) {
    if (!new_model)
        throw std::logic_error{"Solver::set_model(): trying to set nullptr."};

    if (model == new_model)
        return;

    model = new_model;
    if (strategy) {
        // try to assign a new Hamiltonian to the existing Solver strategy
        bool success = strategy->set_hamiltonian(model->hamiltonian());
        if (!success) // fails if the they have incompatible scalar types
            strategy.reset();
    }

    // creates a SolverStrategy with a scalar type suited to the Hamiltonian
    if (!strategy)
        strategy = create_strategy_for(model->hamiltonian());
}

void Solver::solve() {
    calculation_timer.tic();
    strategy->solve();
    calculation_timer.toc();
}

std::shared_ptr<const System> Solver::system() const {
    return model->system();
}

DenseURef Solver::eigenvalues() {
    solve();
    return strategy->eigenvalues();
}

DenseURef Solver::eigenvectors() {
    solve();
    return strategy->eigenvectors();
}
