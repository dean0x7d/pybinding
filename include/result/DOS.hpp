#pragma once
#include "result/Result.hpp"
#include "support/dense.hpp"

namespace tbm {

/**
 The total density of states.
 */
class DOS : public Result {
public:
    DOS(ArrayXd energy, float broadening);

    virtual void visit(const Solver* solver) override;

public:
    const ArrayX<float>& get_dos() const { return dos; }
    const ArrayX<float>& get_energy() const { return energy; }

protected:
    ArrayX<float> energy;
    ArrayX<float> dos;
    const float broadening;
};

} // namespace tbm
