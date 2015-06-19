#include "result/LDOSpoint.hpp"
#include "system/System.hpp"
#include "solver/Solver.hpp"
#include "greens/Greens.hpp"
#include "Model.hpp"
using namespace tbm;
using physics::pi;


LDOSpoint::LDOSpoint(ArrayXd energy, float broadening, Cartesian position,
                     short sublattice, std::vector<Cartesian> k_path)
    : energy{energy.cast<float>()},
      broadening{broadening},
      target_position{position},
      target_sublattice{sublattice},
      k_path{std::move(k_path)}
{}

ArrayXf LDOSpoint::calc_ldos(const SolverStrategy* solver)
{
    // find the lattice site closest to the desired coordinates
    int site_index = system->find_nearest(target_position, target_sublattice);
    
    // get a slice of the probability - contains values at a constant location but different energy
    // TODO: also handle <double>
    ArrayXf probability_slice;
    if (solver->eigenvectors().type == ScalarType::f)
        probability_slice = uref_cast<ArrayXXf>(solver->eigenvectors()).row(site_index).abs2();
    else
        probability_slice = uref_cast<ArrayXXcf>(solver->eigenvectors()).row(site_index).abs2();

    // TODO: also handle <double>
    auto eigenvalues = uref_cast<ArrayXf>(solver->eigenvalues());

    // at each value of the energy
    // calculate LDOS(E) = 1/(sqrt(pi)*G) * sum(psi^2 * exp((En-E)^2 / G^2))
    ArrayXf ldos(energy.size());
    transform(energy, ldos, [&](const float E) {
        const float inverted_broadening = 1 / (broadening*broadening);
        // Gaussian exponential
        auto gaussian = exp(-(eigenvalues - E).square() * inverted_broadening);
        // sum over the eigenvalues/probability slice
        return 1 / (sqrt(pi)*broadening) * sum(probability_slice * gaussian);
    });
    
    return ldos;
}

void LDOSpoint::visit(const SolverStrategy* solver)
{
    ldos = calc_ldos(solver);
}
