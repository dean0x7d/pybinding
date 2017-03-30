#include "KPM.hpp"

namespace cpb {

KPM::KPM(Model const& model, MakeStrategy const& make_strategy)
    : model(model.eval()), make_strategy(make_strategy),
      strategy(make_strategy(model.hamiltonian())) {}

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

std::vector<ArrayXcd> KPM::calc_greens_vector(idx_t row, std::vector<idx_t> const& cols,
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

ArrayXXdCM KPM::calc_ldos(ArrayXd const& energy, double broadening, Cartesian position,
                          string_view sublattice, bool reduce) const {
    auto const system_index = model.system()->find_nearest(position, sublattice);
    auto const ham_idx = model.system()->to_hamiltonian_indices(system_index);

    calculation_timer.tic();
    auto results = ArrayXXdCM(energy.size(), ham_idx.size());
    for (auto i = 0; i < ham_idx.size(); ++i) {
        results.col(i) = strategy->ldos(ham_idx[i], energy, broadening);
    }
    calculation_timer.toc();

    return (reduce && results.cols() > 1) ? results.rowwise().sum() : results;
}

ArrayXd KPM::calc_dos(ArrayXd const& energy, double broadening) const {
    calculation_timer.tic();
    auto dos = strategy->dos(energy, broadening);
    calculation_timer.toc();
    return dos;
}

std::string KPM::report(bool shortform) const {
    return strategy->report(shortform) + " " + calculation_timer.str();
}

kpm::Stats const& KPM::get_stats() const {
    return strategy->get_stats();
}

} // namespace cpb
