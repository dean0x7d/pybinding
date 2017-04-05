import pytest
import numpy as np

import pybinding as pb
from pybinding.repository import graphene

models = {
    'graphene-pristine': [graphene.monolayer(), pb.rectangle(15)],
    'graphene-pristine-oversized': [graphene.monolayer(), pb.rectangle(20)],
    'graphene-const_potential': [graphene.monolayer(), pb.rectangle(15),
                                 pb.constant_potential(0.5)],
    'graphene-magnetic_field': [graphene.monolayer(), pb.rectangle(15),
                                graphene.constant_magnetic_field(1e3)],
}

configurations = [
    {'matrix_format': "CSR", 'optimal_size': False, 'interleaved': False},
    {'matrix_format': "CSR", 'optimal_size': True,  'interleaved': False},
    {'matrix_format': "CSR", 'optimal_size': False, 'interleaved': True},
    {'matrix_format': "ELL", 'optimal_size': True,  'interleaved': True},
]


@pytest.fixture(scope='module', ids=list(models.keys()), params=models.values())
def model(request):
    return pb.Model(*request.param)


@pytest.fixture(scope='module')
def kpm(model):
    kernel = pb.chebyshev.lorentz_kernel()
    strategies = [pb.chebyshev.kpm(model, kernel=kernel, **c) for c in configurations]
    strategies += [pb.chebyshev.kpm_python(model, kernel=kernel)]
    if hasattr(pb._cpp, 'kpm_cuda'):
        strategies += [pb.chebyshev.kpm_cuda(model, kernel=kernel)]
    return strategies


def test_ldos(kpm, baseline, plot_if_fails):
    energy = np.linspace(0, 2, 25)
    results = [k.calc_ldos(energy, broadening=0.15, position=(0, 0), sublattice='B') for k in kpm]

    expected = pb.results.LDOS(energy, baseline(results[0].ldos.astype(np.float32)))
    for i in range(len(results)):
        plot_if_fails(results[i], expected, 'plot', label=i)

    for result in results:
        assert pytest.fuzzy_equal(result, expected, rtol=1e-3, atol=1e-6)


def test_kpm_multiple_indices(model):
    """KPM can take a vector of column indices and return the Green's function for all of them"""
    kpm = pb.chebyshev.kpm(model)

    num_sites = model.system.num_sites
    i, j = num_sites // 2, num_sites // 4
    energy = np.linspace(-0.3, 0.3, 10)
    broadening = 0.8

    cols = [j, j + 1, j + 2]
    gs = kpm.calc_greens(i, cols, energy, broadening)
    assert len(gs) == len(cols)

    g = kpm.calc_greens(j, i, energy, broadening)
    assert pytest.fuzzy_equal(gs[0], g)


def test_kpm_reuse():
    """KPM should return the same result when a single object is used for multiple calculations"""
    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(10))
    kpm = pb.chebyshev.kpm(model)
    energy = np.linspace(-5, 5, 50)
    broadening = 0.1

    for position in [0, 0], [6, 0]:
        actual = kpm.calc_ldos(energy, broadening, position)
        expected = pb.chebyshev.kpm(model).calc_ldos(energy, broadening, position)
        assert pytest.fuzzy_equal(actual, expected, rtol=1e-3, atol=1e-6)


def test_ldos_sublattice():
    """LDOS for A and B sublattices should be antisymmetric for graphene with a mass term"""
    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(10), graphene.mass_term(1))
    kpm = pb.chebyshev.kpm(model)

    a, b = (kpm.calc_ldos(np.linspace(-5, 5, 50), 0.1, [0, 0], sub) for sub in ('A', 'B'))
    assert pytest.fuzzy_equal(a.ldos, b.ldos[::-1], rtol=1e-3, atol=1e-6)


def test_optimized_hamiltonian():
    """Currently available only in internal interface"""
    from pybinding import _cpp
    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(10))
    h = model.hamiltonian
    oh = _cpp.OptimizedHamiltonian(model.raw_hamiltonian, 0)

    assert oh.matrix.shape == h.shape
    assert oh.sizes[-1] == h.shape[0]
    assert len(oh.indices) == h.shape[0]


dos_models = {
    'graphene-const_potential': [graphene.monolayer(), pb.rectangle(25),
                                 pb.constant_potential(0.5)],
    'graphene-magnetic_field': [graphene.monolayer(), pb.rectangle(25),
                                graphene.constant_magnetic_field(1e3)],
}


@pytest.mark.parametrize("params", dos_models.values(), ids=list(dos_models.keys()))
def test_dos(params, baseline, plot_if_fails):
    configurations = [
        {'matrix_format': "ELL", 'optimal_size': False, 'interleaved': False},
        {'matrix_format': "ELL", 'optimal_size': True, 'interleaved': True},
    ]
    model = pb.Model(*params)

    kernel = pb.chebyshev.lorentz_kernel()
    strategies = [pb.chebyshev.kpm(model, num_random=1, kernel=kernel, **c) for c in configurations]

    energy = np.linspace(0, 2, 25)
    results = [kpm.calc_dos(energy, broadening=0.15) for kpm in strategies]

    expected = pb.results.DOS(energy, baseline(results[0].dos.astype(np.float32)))
    for i in range(len(results)):
        plot_if_fails(results[i], expected, 'plot', label=i)

    for result in results:
        assert pytest.fuzzy_equal(result, expected, rtol=1e-3, atol=1e-6)
