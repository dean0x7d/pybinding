#pragma once
#include <memory>

namespace tbm {

class System;
class Solver;
class Greens;
class Model;

/**
 Result base class. Visitor pattern.
 */
class Result {
public:
    virtual void visit(const Solver*) {}
    virtual void visit(Greens*) {}

    std::shared_ptr<const System> system;
    Model* model;
};

} // namespace tbm
