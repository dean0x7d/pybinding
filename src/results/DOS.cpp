#include "result/DOS.hpp"
#include "solver/Solver.hpp"
#include "support/physics.hpp"
using namespace tbm;
using physics::pi;

DOS::DOS(ArrayXd energy, float broadening)
    : energy{energy.cast<float>()}, broadening{broadening}
{}

void DOS::visit(const SolverStrategy* solver)
{
    const float inverted_broadening = 1 / (broadening*broadening);
    // TODO: also handle <double>
    auto eigenvalues = uref_cast<ArrayXf>(solver->eigenvalues());

    // at each value of the energy
    // calculate DOS(E) = 1/(sqrt(pi)*G) * sum(exp((En-E)^2 / G^2))
    dos.resize(energy.size());
    transform(energy, dos, [&](const float E) {
        // Gaussian exponential
        auto gaussian = exp(-(eigenvalues - E).square() * inverted_broadening);
        // sum over the eigenvalues
        return 1/(sqrt(pi)*broadening) * sum(gaussian);
    });
}
