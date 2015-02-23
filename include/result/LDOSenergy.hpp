#pragma once
#include "result/Result.hpp"
#include "support/dense.hpp"

namespace tbm {

class LDOSenergy : public Result {
public:
    LDOSenergy(float energy, float broadening, short sublattice = -1);

    virtual void visit(const Solver* solver) override;
    const ArrayX<float>& get_ldos() const { return ldos; }

protected:
    ArrayX<float> ldos;

    const float target_energy;
    const float broadening;
    const short target_sublattice;
};

} // namespace tbm
