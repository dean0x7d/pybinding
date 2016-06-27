#include "greens/Greens.hpp"

namespace cpb {
using constant::pi;

BaseGreens::BaseGreens(Model const& model, MakeStrategy const& make_strategy)
    : model(model), make_strategy(make_strategy), strategy(make_strategy(model.hamiltonian())) {}

void BaseGreens::set_model(Model const& new_model) {
    model = new_model;

    if (strategy) { // try to assign a new Hamiltonian to the existing Green's strategy
        bool success = strategy->change_hamiltonian(model.hamiltonian());
        if (!success) { // fails if the they have incompatible scalar types
            strategy.reset();
        }
    }

    if (!strategy) { // create a Green's strategy with a scalar type suited to the Hamiltonian
        strategy = make_strategy(model.hamiltonian());
    }
}

ArrayXcd BaseGreens::calc_greens(int row, int col, ArrayXd const& energy,
                                 double broadening) const {
    auto const size = model.hamiltonian().rows();
    if (row < 0 || row > size || col < 0 || col > size) {
        throw std::logic_error("KPM::calc_greens(i,j): invalid value for i or j.");
    }

    calculation_timer.tic();
    auto greens_function = strategy->calc(row, col, energy, broadening);
    calculation_timer.toc();
    return greens_function;
}

std::vector<ArrayXcd> BaseGreens::calc_greens_vector(int row, std::vector<int> const& cols,
                                                     ArrayXd const& energy,
                                                     double broadening) const {
    auto const size = model.hamiltonian().rows();
    auto const row_error = row < 0 || row > size;
    auto const col_error = std::any_of(cols.begin(), cols.end(),
                                       [&](int col) { return col < 0 || col > size; });
    if (row_error || col_error) {
        throw std::logic_error("KPM::calc_greens(i,j): invalid value for i or j.");
    }

    calculation_timer.tic();
    auto greens_functions = strategy->calc_vector(row, cols, energy, broadening);
    calculation_timer.toc();
    return greens_functions;
}

ArrayXd BaseGreens::calc_ldos(ArrayXd const& energy, double broadening,
                              Cartesian position, sub_id sublattice) const {
    auto i = model.system()->find_nearest(position, sublattice);
    auto greens_function = calc_greens(i, i, energy, broadening);

    return -1/pi * greens_function.imag();
}

Deferred<ArrayXd> BaseGreens::deferred_ldos(ArrayXd const& energy, double broadening,
                                            Cartesian position, sub_id sublattice) const {
    auto shared_strategy = std::shared_ptr<GreensStrategy>(make_strategy(model.hamiltonian()));
    auto& model = this->model;

    return {
        [shared_strategy, model, position, sublattice, energy, broadening](ArrayXd& ldos) {
            auto i = model.system()->find_nearest(position, sublattice);
            auto greens_function = shared_strategy->calc(i, i, energy, broadening);
            ldos = -1/pi * greens_function.imag();
        },
        [shared_strategy] {
            return shared_strategy->report(true);
        }
    };
}

std::string BaseGreens::report(bool shortform) const {
    return strategy ? strategy->report(shortform) + " " + calculation_timer.str() : "";
}

} // namespace cpb
