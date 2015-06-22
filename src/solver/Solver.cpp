#include "solver/Solver.hpp"
#include "Model.hpp"
#include "support/physics.hpp"
using namespace tbm;

void Solver::set_model(const std::shared_ptr<const Model>& new_model) {
    if (!new_model)
        throw std::logic_error{"Solver::set_model(): trying to set nullptr."};

    if (model == new_model)
        return;

    is_solved = false;
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
    if (is_solved)
        return;

    calculation_timer.tic();
    strategy->solve();
    calculation_timer.toc();

    is_solved = true;
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

ArrayXd Solver::calc_dos(ArrayXd target_energies, double broadening) {
    ArrayXd dos(target_energies.size());

    // TODO: also handle <double>
    auto energies = uref_cast<ArrayXf>(eigenvalues());
    auto const inverted_broadening = 1 / (broadening*broadening);
    auto const sqrt_pi = sqrt(physics::pi);

    // calculate DOS(E) = 1/(sqrt(pi)*G) * sum(exp((En-E)^2 / G^2))
    transform(target_energies, dos, [&](const double E) {
        auto gaussian = exp(-(energies - E).square() * inverted_broadening);
        return 1 / (sqrt_pi * broadening) * sum(gaussian);
    });
}

ArrayXd Solver::calc_ldos(double target_energy, double broadening, int target_sublattice) {
    auto const& sys = *system();
    auto const system_size = sys.num_sites();
    ArrayXd ldos = ArrayXd::Zero(system_size);

    auto const inverted_broadening = 1 / (broadening * broadening);
    auto const sqrt_pi = sqrt(physics::pi);

    // TODO: also handle <double>
    auto energies = uref_cast<ArrayXf>(eigenvalues());
    for (int i = 0; i < system_size; i++) {
        // if a target_sublattice is set, only consider those sites
        if (target_sublattice >= 0 && sys.sublattice[i] != target_sublattice)
            continue;

        // TODO: also handle <double>
        auto probability_slice = [&]() -> ArrayXf {
            if (eigenvectors().type == ScalarType::f)
                return uref_cast<ArrayXXf>(eigenvectors()).row(i).abs2();
            else
                return uref_cast<ArrayXXcf>(eigenvectors()).row(i).abs2();
        }();

        auto gaussian = exp(-(energies - target_energy).square() * inverted_broadening);
        ldos[i] = 1 / (sqrt_pi * broadening) * sum(probability_slice * gaussian);
    }

    return ldos;
}
