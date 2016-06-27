#include "solver/Solver.hpp"

namespace cpb { namespace compute {

struct CalcDOS {
    ArrayXf const& target_energies;
    float broadening;

    template<class Array>
    ArrayXd operator()(Array En) {
        auto const scale = 1 / (broadening * sqrt(2 * constant::pi));
        auto const constant = -0.5f / pow(broadening, 2);

        // DOS(E) = 1 / (broadening * sqrt(2pi)) * sum(exp(-0.5 * (En-E)^2 / broadening^2))
        ArrayXd dos(target_energies.size());
        transform(target_energies, dos, [&](float E) {
            auto gaussian = exp((En - E).square() * constant);
            return scale * sum(gaussian);
        });
        return dos;
    }
};

struct CalcSpatialLDOS {
    float target_energy;
    float broadening;

    template<class Array1D, class Array2D>
    ArrayXd operator()(Array1D En, Array2D psi) {
        using scalar_t = typename Array1D::Scalar;
        auto const scale = 1 / (broadening * sqrt(2 * constant::pi));
        auto const constant = -0.5f / pow(broadening, 2);

        // DOS(r) = 1 / (b * sqrt(2pi)) * sum(|psi(r)|^2 * exp(-0.5 * (En-E)^2 / b^2))
        ArrayXd ldos(psi.rows());
        for (auto i = 0; i < ldos.size(); ++i) {
            ArrayX<scalar_t> psi2 = psi.row(i).abs2();
            auto gaussian = exp((En - target_energy).square() * constant);
            ldos[i] = scale * sum(psi2 * gaussian);
        }
        return ldos;
    }
};

} // namespace compute

BaseSolver::BaseSolver(Model const& model, MakeStrategy const& make_strategy)
    : model(model), make_strategy(make_strategy), strategy(make_strategy(model.hamiltonian())) {}

void BaseSolver::set_model(Model const& new_model) {
    is_solved = false;
    model = new_model;

    if (strategy) {// try to assign a new Hamiltonian to the existing Solver strategy
        bool success = strategy->change_hamiltonian(model.hamiltonian());
        if (!success) { // fails if the they have incompatible scalar types
            strategy.reset();
        }
    }

    if (!strategy) { // creates a SolverStrategy with a scalar type suited to the Hamiltonian
        strategy = make_strategy(model.hamiltonian());
    }
}

void BaseSolver::solve() {
    if (is_solved)
        return;

    calculation_timer.tic();
    strategy->solve();
    calculation_timer.toc();

    is_solved = true;
}

RealArrayConstRef BaseSolver::eigenvalues() {
    solve();
    return strategy->eigenvalues();
}

ComplexArrayConstRef BaseSolver::eigenvectors() {
    solve();
    return strategy->eigenvectors();
}

ArrayXd BaseSolver::calc_dos(ArrayXf target_energies, float broadening) {
    return num::match<ArrayX>(eigenvalues(), compute::CalcDOS{target_energies, broadening});
}

ArrayXd BaseSolver::calc_spatial_ldos(float target_energy, float broadening) {
    return num::match2sp<ArrayX, ArrayXX>(
        eigenvalues(), eigenvectors(),
        compute::CalcSpatialLDOS{target_energy, broadening}
    );
}

std::string BaseSolver::report(bool shortform) const {
    return strategy->report(shortform) + " " + calculation_timer.str();
}

} // namespace cpb
