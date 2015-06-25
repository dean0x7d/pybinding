import pytest

import pybinding as pb
from pybinding.repository import graphene

solvers = ['feast']
models = {
    'graphene-pristine': {'model': [graphene.lattice.monolayer(), pb.shape.rectangle(10)],
                          'feast': [(-0.1, 0.1), 28]},
    'graphene-magnetic_field': {'model': [graphene.lattice.monolayer(), pb.shape.rectangle(6),
                                          pb.magnetic.constant(10)],
                                'feast': [(-0.1, 0.1), 18]},
}


@pytest.fixture(scope='module', ids=list(models.keys()), params=models.values())
def model_ex(request):
    return pb.Model(*request.param['model']), request.param


@pytest.fixture(scope='module', params=solvers)
def solver(request, model_ex):
    model, solver_cfg = model_ex
    solver = pb.solver.make_feast(model, *solver_cfg[request.param])
    solver.solve()
    return solver


def test_pickle_round_trip(solver, tmpdir):
    file_name = str(tmpdir.join('file.npz'))
    solver.save(file_name)
    from_file = pb.solver.Solver.from_file(file_name)

    assert pytest.fuzzy_equal(solver, from_file)


def test_eigenvalues(solver, baseline):
    expected_eigenvalues = baseline(solver.eigenvalues)
    assert pytest.fuzzy_equal(solver.eigenvalues, expected_eigenvalues, 1.e-3, 1.e-6)
