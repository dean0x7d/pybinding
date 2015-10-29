import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


solvers = ['arpack']
if hasattr(pb._cpp, 'FEAST'):
    solvers.append('feast')

models = {
    'graphene-pristine': {'model': [graphene.lattice.monolayer(), pb.shape.rectangle(10)],
                          'arpack': [30],
                          'feast': [(-0.1, 0.1), 28]},
    'graphene-magnetic_field': {'model': [graphene.lattice.monolayer(), pb.shape.rectangle(6),
                                          pb.magnetic.constant(10)],
                                'arpack': [30],
                                'feast': [(-0.1, 0.1), 18]},
}


@pytest.fixture(scope='module', ids=list(models.keys()), params=models.values())
def model_ex(request):
    return pb.Model(*request.param['model']), request.param


@pytest.fixture(scope='module', params=solvers)
def solver(request, model_ex):
    model, solver_cfg = model_ex
    make_solver = getattr(pb.solver, 'make_' + request.param)
    solver = make_solver(model, *solver_cfg[request.param])
    solver.solve()
    return solver


def test_pickle_round_trip(solver, tmpdir):
    file_name = str(tmpdir.join('file.npz'))
    solver.save(file_name)
    from_file = pb.solver.Solver.from_file(file_name)

    assert pytest.fuzzy_equal(solver, from_file)


def test_eigenvalues(solver, baseline, plot):
    eig = solver.calc_eigenvalues(map_probability_at=(0, 0))
    expected = baseline(eig)
    plot(eig, expected, 'plot_heatmap')
    assert pytest.fuzzy_equal(eig, expected, 2.e-2, 1.e-6)


def test_dos(solver, baseline, plot):
    dos = solver.calc_dos(np.linspace(0, 0.1, 15), 0.01)
    expected = baseline(dos)
    plot(dos, expected, 'plot')
    assert pytest.fuzzy_equal(dos, expected)


def test_lapack(baseline, plot):
    model = pb.Model(graphene.lattice.monolayer(), pb.symmetry.translational())
    solver = pb.solver.make_lapack(model)
    assert pytest.fuzzy_equal(solver.eigenvalues, [-3*abs(graphene.t), 3*abs(graphene.t)])

    from math import pi, sqrt
    g = [0, 0]
    k1 = [-4*pi / (3*sqrt(3) * graphene.a_cc), 0]
    m = [0, 2*pi / (3 * graphene.a_cc)]
    k2 = [2*pi / (3*sqrt(3) * graphene.a_cc), 2*pi / (3 * graphene.a_cc)]

    bands = solver.calc_bands(k1, g, m, k2, step=1)
    expected = baseline(bands)
    plot(bands, expected, 'plot', names=['K', r'$\Gamma$', 'M', 'K'])
    assert pytest.fuzzy_equal(bands, expected, 2.e-2, 1.e-6)
