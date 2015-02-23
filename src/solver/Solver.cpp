#include "solver/Solver.hpp"
#include "result/Result.hpp"
using namespace tbm;

void Solver::solve()
{
    if (is_solved)
        return;

    solve_timer.tic();
    v_solve();
    solve_timer.toc();
    is_solved = true;
}

std::string Solver::report(bool shortform) const
{
    auto report = v_report(shortform);
    report += " " + solve_timer.str();
    return report;
}

void Solver::accept(Result& result)
{
    solve();
    result.visit(this);
}
