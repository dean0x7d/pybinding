#pragma once
#include <memory>

namespace tbm {

class System;
class SolverStrategy;
class GreensStrategy;
class Model;

/**
 Result base class. Visitor pattern.
 */
class Result {
public:
    virtual void visit(const SolverStrategy*) {}
    virtual void visit(GreensStrategy*) {}

    std::shared_ptr<const System> system;
    Model* model;
};

} // namespace tbm
