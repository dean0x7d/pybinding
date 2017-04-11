import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene

solvers = ['arpack']
if hasattr(pb._cpp, 'FEAST'):
    solvers.append('feast')

models = {
    'graphene-magnetic_field': {'model': [graphene.monolayer(), pb.rectangle(6),
                                          graphene.constant_magnetic_field(10)],
                                'arpack': [30],
                                'feast': [(-0.1, 0.1), 18]},
}


@pytest.fixture(scope='module', ids=list(models.keys()), params=models.values())
def model_ex(request):
    return pb.Model(*request.param['model']), request.param


@pytest.fixture(scope='module', params=solvers)
def solver(request, model_ex):
    model, solver_cfg = model_ex
    make_solver = getattr(pb.solver, request.param)
    solver = make_solver(model, *solver_cfg[request.param])
    solver.solve()
    return solver


def test_eigenvalues(solver, baseline, plot_if_fails):
    eig = solver.calc_eigenvalues(map_probability_at=(0, 0))
    expected = baseline(eig)
    plot_if_fails(eig, expected, 'plot_heatmap')
    assert pytest.fuzzy_equal(eig, expected, 2.e-2, 1.e-5)


def test_dos(solver, baseline, plot_if_fails):
    energy = np.linspace(0, 0.075, 15)
    result = solver.calc_dos(energy, 0.01)

    expected = result.with_data(baseline(result.data))
    plot_if_fails(result, expected, 'plot')

    assert pytest.fuzzy_equal(result, expected, rtol=2e-2, atol=1e-5)


def test_ldos(solver, plot_if_fails):
    """Compare an LDOS sum at every position with directly calculated DOS"""
    energy = np.linspace(0, 0.075, 15)
    broadening = 0.01
    expected = solver.calc_dos(energy, broadening)

    ldos = np.stack([solver.calc_ldos(energy, broadening, position).data
                     for position in zip(*solver.system.positions)])
    result = expected.with_data(np.sum(ldos, axis=0))

    plot_if_fails(result, expected, 'plot')
    assert pytest.fuzzy_equal(result, expected)


def test_spatial_ldos(solver, baseline, plot_if_fails):
    ldos_map = solver.calc_spatial_ldos(energy=0.05, broadening=0.01)

    x_max = solver.system.x.max()
    y_max = solver.system.y.max()
    ldos_map = ldos_map.cropped(x=(x_max - 1, x_max + 1), y=(y_max - 1, y_max + 1))

    expected = ldos_map.with_data(baseline(ldos_map.data))
    plot_if_fails(ldos_map, expected, "plot")

    assert pytest.fuzzy_equal(ldos_map, expected, rtol=1e-2, atol=1e-5)


def test_lapack(baseline, plot_if_fails):
    model = pb.Model(graphene.monolayer(), pb.translational_symmetry())
    solver = pb.solver.lapack(model)
    assert pytest.fuzzy_equal(solver.eigenvalues, [-3*abs(graphene.t), 3*abs(graphene.t)])

    from math import pi, sqrt
    g = [0, 0]
    k1 = [-4*pi / (3*sqrt(3) * graphene.a_cc), 0]
    m = [0, 2*pi / (3 * graphene.a_cc)]
    k2 = [2*pi / (3*sqrt(3) * graphene.a_cc), 2*pi / (3 * graphene.a_cc)]

    bands = solver.calc_bands(k1, g, m, k2, step=3)
    expected = baseline(bands)
    plot_if_fails(bands, expected, 'plot')
    assert pytest.fuzzy_equal(bands, expected, 2.e-2, 1.e-6)
