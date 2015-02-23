#include "result/LDOSenergy.hpp"
#include "system/System.hpp"
#include "solver/Solver.hpp"
#include "support/physics.hpp"
using namespace tbm;

LDOSenergy::LDOSenergy(float energy, float broadening, short sublattice)
    : target_energy{energy}, broadening{broadening}, target_sublattice{sublattice}
{}

void LDOSenergy::visit(const Solver* solver)
{
    using physics::pi;

    const int system_size = system->num_sites();
    const float inverted_broadening = 1 / (broadening*broadening);
    ldos.resize(system_size);
    // TODO: also handle <double>
    auto eigenvalues = uref_cast<ArrayXf>(solver->eigenvalues());

    for (int i = 0; i < system_size; i++)
    { // evaluate LDOS at each atom
        // if a filter is set, skip atoms in one sublattice
        if (target_sublattice >= 0 && system->sublattice[i] != target_sublattice)
            continue;

        // get a slice of the probability - constant location but different energy
        // TODO: also handle <double>
        ArrayXf probability_slice;
        if (solver->eigenvectors().type == ScalarType::f)
            probability_slice = uref_cast<ArrayXXf>(solver->eigenvectors()).row(i).abs2();
        else
            probability_slice = uref_cast<ArrayXXcf>(solver->eigenvectors()).row(i).abs2();

        // Gaussian exponential
        auto gaussian = exp(-(eigenvalues - target_energy).square() * inverted_broadening);
        // sum over the eigenvalues/probability slice
        ldos[i] = 1 / (sqrt(pi)*broadening) * sum(probability_slice * gaussian);
    }
}
