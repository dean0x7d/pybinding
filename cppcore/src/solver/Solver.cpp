#include "solver/Solver.hpp"
#include "support/physics.hpp"
using namespace tbm;


BaseSolver::BaseSolver(Model const& model, MakeStrategy const& make_strategy)
    : model(model), make_strategy(make_strategy), strategy(make_strategy(model)) {}

void BaseSolver::set_model(Model const& new_model) {
    is_solved = false;
    model = new_model;

    if (strategy) {
        // try to assign a new Hamiltonian to the existing Solver strategy
        bool success = strategy->set_hamiltonian(model.hamiltonian());
        if (!success) // fails if the they have incompatible scalar types
            strategy.reset();
    }

    // creates a SolverStrategy with a scalar type suited to the Hamiltonian
    if (!strategy)
        strategy = make_strategy(model);
}

void BaseSolver::solve() {
    if (is_solved)
        return;

    calculation_timer.tic();
    strategy->solve();
    calculation_timer.toc();

    is_solved = true;
}

DenseURef BaseSolver::eigenvalues() {
    solve();
    return strategy->eigenvalues();
}

DenseURef BaseSolver::eigenvectors() {
    solve();
    return strategy->eigenvectors();
}

ArrayXd BaseSolver::calc_dos(ArrayXf target_energies, float broadening) {
    ArrayXd dos(target_energies.size());

    // TODO: also handle <double>
    auto En = uref_cast<ArrayXf>(eigenvalues());
    auto const scale = 1 / (broadening * sqrt(2 * physics::pi));
    auto const constant = -0.5f / pow(broadening, 2);

    // DOS(E) = 1 / (broadening * sqrt(2pi)) * sum(exp(-0.5 * (En-E)^2 / broadening^2))
    transform(target_energies, dos, [&](float E) {
        auto gaussian = exp((En - E).square() * constant);
        return scale * sum(gaussian);
    });

    return dos;
}

ArrayXd BaseSolver::calc_spatial_ldos(float target_energy, float broadening) {
    auto const& sys = *system();
    auto const system_size = sys.num_sites();
    ArrayXd ldos = ArrayXd::Zero(system_size);

    auto const scale = 1 / (broadening * sqrt(2 * physics::pi));
    auto const constant = -0.5f / pow(broadening, 2);

    // TODO: also handle <double>
    auto En = uref_cast<ArrayXf>(eigenvalues());

    // DOS(r) = 1 / (b * sqrt(2pi)) * sum(|psi(r)|^2 * exp(-0.5 * (En-E)^2 / b^2))
    for (int i = 0; i < system_size; i++) {
        // TODO: also handle <double>
        auto psi2 = [&]() -> ArrayXf {
            if (eigenvectors().type == ScalarType::f)
                return uref_cast<ArrayXXf>(eigenvectors()).row(i).abs2();
            else
                return uref_cast<ArrayXXcf>(eigenvectors()).row(i).abs2();
        }();

        auto gaussian = exp((En - target_energy).square() * constant);
        ldos[i] = scale * sum(psi2 * gaussian);
    }

    return ldos;
}

std::string BaseSolver::report(bool shortform) const {
    return strategy->report(shortform) + " " + calculation_timer.str();
}
