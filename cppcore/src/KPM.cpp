#include "KPM.hpp"

using namespace fmt::literals;

namespace cpb {

KPM::KPM(Model const& model, kpm::Compute const& compute, kpm::Config const& config)
    : model(model.eval()), core(kpm::Core(model.hamiltonian(), compute, config)) {}

void KPM::set_model(Model const& new_model) {
    model = new_model;
    core.set_hamiltonian(model.hamiltonian());
}

std::string KPM::report(bool shortform) const {
    return core.report(shortform) + " " + calculation_timer.str();
}

ArrayXcd KPM::moments(idx_t num_moments, VectorXcd const& alpha, VectorXcd const& beta,
                      SparseMatrixXcd const& op) const {
    auto const ham_size =  model.system()->hamiltonian_size();
    auto const check_size = std::unordered_map<char const*, bool>{
        {"alpha", alpha.size() == model.system()->hamiltonian_size()},
        {"beta", beta.size() == 0 || beta.size() == model.system()->hamiltonian_size()},
        {"operator", op.size() == 0 || (op.rows() == ham_size && op.cols() == ham_size)}
    };
    for (auto const& pair : check_size) {
        if (!pair.second) {
            throw std::runtime_error("Size mismatch between the model Hamiltonian and the given "
                                     "argument '{}'"_format(pair.first));
        }
    }

    if (!model.is_complex()) {
        auto const check_scalar_type = std::unordered_map<char const*, bool>{
            {"alpha", alpha.imag().isZero()},
            {"beta", beta.imag().isZero()},
            {"operator", Eigen::Map<ArrayXcd const>(op.valuePtr(), op.nonZeros()).imag().isZero()}
        };

        for (auto const& pair : check_scalar_type) {
            if (!pair.second) {
                throw std::runtime_error("The model Hamiltonian is real, but the given argument "
                                         "'{}' is complex"_format(pair.first));
            }
        }
    }

    calculation_timer.tic();
    auto moments = core.moments(num_moments, alpha, beta, op);
    calculation_timer.toc();
    return moments;
}

ArrayXXdCM KPM::calc_ldos(ArrayXd const& energy, double broadening, Cartesian position,
                          string_view sublattice, bool reduce) const {
    auto const system_index = model.system()->find_nearest(position, sublattice);
    auto const ham_idx = model.system()->to_hamiltonian_indices(system_index);

    calculation_timer.tic();
    auto results = core.ldos({begin(ham_idx), end(ham_idx)}, energy, broadening);
    calculation_timer.toc();

    return (reduce && results.cols() > 1) ? results.rowwise().sum() : results;
}

ArrayXXdCM KPM::calc_spatial_ldos(ArrayXd const& energy, double broadening, Shape const& shape,
                                  string_view sublattice) const {
    if (model.get_lattice().has_multiple_orbitals()) {
        throw std::runtime_error("This function doesn't currently support multi-orbital models");
    }

    auto const& system = *model.system();

    calculation_timer.tic();
    auto const indices = [&]{
        auto const contains = shape.contains(system.positions);
        auto const range = system.sublattice_range(sublattice);

        auto v = std::vector<idx_t>();
        v.reserve(std::count(contains.data() + range.start, contains.data() + range.end, true));

        for (auto i = range.start; i < range.end; ++i) {
            if (contains[i]) { v.push_back(i); }
        }

        return v;
    }();

    auto results = core.ldos(indices, energy, broadening);
    calculation_timer.toc();

    return results;
}

ArrayXd KPM::calc_dos(ArrayXd const& energy, double broadening, idx_t num_random) const {
    calculation_timer.tic();
    auto dos = core.dos(energy, broadening, num_random);
    calculation_timer.toc();
    return dos;
}

ArrayXcd KPM::calc_greens(idx_t row, idx_t col, ArrayXd const& energy, double broadening) const {
    auto const size = model.hamiltonian().rows();
    if (row < 0 || row > size || col < 0 || col > size) {
        throw std::logic_error("KPM::calc_greens(i,j): invalid value for i or j.");
    }

    calculation_timer.tic();
    auto greens_function = core.greens(row, col, energy, broadening);
    calculation_timer.toc();
    return greens_function;
}

std::vector<ArrayXcd> KPM::calc_greens_vector(idx_t row, std::vector<idx_t> const& cols,
                                              ArrayXd const& energy, double broadening) const {
    auto const size = model.hamiltonian().rows();
    auto const row_error = row < 0 || row > size;
    auto const col_error = std::any_of(cols.begin(), cols.end(),
                                       [&](idx_t col) { return col < 0 || col > size; });
    if (row_error || col_error) {
        throw std::logic_error("KPM::calc_greens(i,j): invalid value for i or j.");
    }

    calculation_timer.tic();
    auto greens_functions = core.greens_vector(row, cols, energy, broadening);
    calculation_timer.toc();
    return greens_functions;
}

ArrayXd KPM::calc_conductivity(ArrayXd const& chemical_potential, double broadening,
                               double temperature, string_view direction, idx_t num_random,
                               idx_t num_points) const {
    auto const xyz = std::string("xyz");
    if (direction.size() != 2 || xyz.find_first_of(direction) == std::string::npos) {
        throw std::logic_error("Invalid direction: must be 'xx', 'xy', 'zz', or similar.");
    }

    auto const is_multiorbital = model.get_lattice().has_multiple_orbitals();
    auto const& system = *model.system();
    auto const& p = is_multiorbital ? system.expanded_positions() : system.positions;
    auto map = std::unordered_map<char, ArrayXf const*>{{'x', &p.x}, {'y', &p.y}, {'z', &p.z}};

    calculation_timer.tic();
    auto result = core.conductivity(*map[direction[0]], *map[direction[1]], chemical_potential,
                                    broadening, temperature, num_random, num_points);
    calculation_timer.toc();
    return result.real();
}

} // namespace cpb
