#include "KPM.hpp"

namespace cpb {

KPM::KPM(Model const& model, MakeStrategy const& make_strategy)
    : model(model), make_strategy(make_strategy), strategy(make_strategy(model.hamiltonian())) {}

void KPM::set_model(Model const& new_model) {
    model = new_model;

    if (strategy) { // try to assign a new Hamiltonian to the existing strategy
        bool success = strategy->change_hamiltonian(model.hamiltonian());
        if (!success) { // fails if the they have incompatible scalar types
            strategy.reset();
        }
    }

    if (!strategy) { // create a new strategy with a scalar type suited to the Hamiltonian
        strategy = make_strategy(model.hamiltonian());
    }
}

ArrayXcd KPM::calc_greens(int row, int col, ArrayXd const& energy,
                          double broadening) const {
    auto const size = model.hamiltonian().rows();
    if (row < 0 || row > size || col < 0 || col > size) {
        throw std::logic_error("KPM::calc_greens(i,j): invalid value for i or j.");
    }

    calculation_timer.tic();
    auto greens_function = strategy->greens(row, col, energy, broadening);
    calculation_timer.toc();
    return greens_function;
}

std::vector<ArrayXcd> KPM::calc_greens_vector(int row, std::vector<int> const& cols,
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
    auto greens_functions = strategy->greens_vector(row, cols, energy, broadening);
    calculation_timer.toc();
    return greens_functions;
}

ArrayXd KPM::calc_ldos(ArrayXd const& energy, double broadening,
                       Cartesian position, std::string const& sublattice) const {
    auto const index = model.system()->find_nearest(position, sublattice);

    calculation_timer.tic();
    auto ldos = strategy->ldos(index, energy, broadening);
    calculation_timer.toc();
    return ldos;
}

std::string KPM::report(bool shortform) const {
    return strategy->report(shortform) + " " + calculation_timer.str();
}

kpm::Stats const& KPM::get_stats() const {
    return strategy->get_stats();
}

} // namespace cpb
