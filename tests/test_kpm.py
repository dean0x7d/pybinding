import pytest
import numpy as np

import pybinding as pb
from pybinding.repository import graphene, group6_tmd

models = {
    'graphene-pristine': [graphene.monolayer(), pb.rectangle(15)],
    'graphene-pristine-oversized': [graphene.monolayer(), pb.rectangle(20)],
    'graphene-const_potential': [graphene.monolayer(), pb.rectangle(15),
                                 pb.constant_potential(0.5)],
    'graphene-magnetic_field': [graphene.monolayer(), pb.rectangle(15),
                                graphene.constant_magnetic_field(1e3)],
}


@pytest.fixture(scope='module', ids=list(models.keys()), params=models.values())
def model(request):
    return pb.Model(*request.param)


ldos_models = {**models, "mos2": [group6_tmd.monolayer_3band("MoS2"), pb.rectangle(6)]}


@pytest.mark.parametrize("params", ldos_models.values(), ids=list(ldos_models.keys()))
def test_ldos(params, baseline, plot_if_fails):
    configurations = [
        {'matrix_format': "CSR", 'optimal_size': False, 'interleaved': False},
        {'matrix_format': "CSR", 'optimal_size': True, 'interleaved': False},
        {'matrix_format': "CSR", 'optimal_size': False, 'interleaved': True},
        {'matrix_format': "ELL", 'optimal_size': True, 'interleaved': True},
    ]
    model = pb.Model(*params)

    kernel = pb.lorentz_kernel()
    strategies = [pb.kpm(model, kernel=kernel, silent=True, **c) for c in configurations]

    energy = np.linspace(0, 2, 25)
    results = [kpm.calc_ldos(energy, broadening=0.15, position=[0, 0.07], reduce=False)
               for kpm in strategies]

    expected = results[0].with_data(baseline(results[0].data.astype(np.float32)))
    for i in range(len(results)):
        plot_if_fails(results[i], expected, 'plot', label=i)

    for result in results:
        assert pytest.fuzzy_equal(result, expected, rtol=1e-3, atol=1e-6)


def test_moments(model, plot_if_fails):
    energy = np.linspace(0, 2, 25)
    broadening = 0.15
    position = dict(position=[0, 0], sublattice="A")

    kpm = pb.kpm(model, silent=True)
    expected_ldos = kpm.calc_ldos(energy, broadening, **position)

    def manual_ldos():
        idx = model.system.find_nearest(**position)
        alpha = np.zeros(model.hamiltonian.shape[0])
        alpha[idx] = 1

        a, b = kpm.scaling_factors
        num_moments = kpm.kernel.required_num_moments(broadening / a)
        moments = kpm.moments(num_moments, alpha)

        ns = np.arange(num_moments)
        scaled_energy = (energy - b) / a
        k = 2 / (a * np.pi * np.sqrt(1 - scaled_energy**2))
        chebyshev = np.cos(ns * np.arccos(scaled_energy[:, np.newaxis]))
        return k * np.sum(moments.real * chebyshev, axis=1)

    ldos = expected_ldos.with_data(manual_ldos())
    plot_if_fails(ldos, expected_ldos, "plot")
    assert pytest.fuzzy_equal(ldos, expected_ldos, rtol=1e-4, atol=1e-6)

    with pytest.raises(RuntimeError) as excinfo:
        kpm.moments(10, [1, 2, 3])
    assert "Size mismatch" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        kpm = pb.kpm(pb.Model(graphene.monolayer()))
        kpm.moments(10, [1j, 2j])
    assert "Hamiltonian is real, but the given argument 'alpha' is complex" in str(excinfo.value)


def test_kpm_multiple_indices(model):
    """KPM can take a vector of column indices and return the Green's function for all of them"""
    kpm = pb.kpm(model, silent=True)

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
    kpm = pb.kpm(model, silent=True)
    energy = np.linspace(-5, 5, 50)
    broadening = 0.1

    for position in [0, 0], [6, 0]:
        actual = kpm.calc_ldos(energy, broadening, position)
        expected = pb.kpm(model).calc_ldos(energy, broadening, position)
        assert pytest.fuzzy_equal(actual, expected, rtol=1e-3, atol=1e-6)


def test_ldos_sublattice():
    """LDOS for A and B sublattices should be antisymmetric for graphene with a mass term"""
    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(10), graphene.mass_term(1))
    kpm = pb.kpm(model, silent=True)

    a, b = (kpm.calc_ldos(np.linspace(-5, 5, 50), 0.1, [0, 0], sub) for sub in ('A', 'B'))
    assert pytest.fuzzy_equal(a.data, b.data[::-1], rtol=1e-3, atol=1e-6)


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

    kernel = pb.lorentz_kernel()
    strategies = [pb.kpm(model, kernel=kernel, silent=True, **c) for c in configurations]

    energy = np.linspace(0, 2, 25)
    results = [kpm.calc_dos(energy, broadening=0.15) for kpm in strategies]

    expected = results[0].with_data(baseline(results[0].data.astype(np.float32)))
    for i in range(len(results)):
        plot_if_fails(results[i], expected, 'plot', label=i)

    for result in results:
        assert pytest.fuzzy_equal(result, expected, rtol=1e-3, atol=1e-6)


cond_models = {
    'graphene-const_potential': [graphene.monolayer(), pb.rectangle(20),
                                 pb.constant_potential(0.5)],
    'graphene-magnetic_field': [graphene.monolayer(), pb.rectangle(20),
                                graphene.constant_magnetic_field(1e3)]
}


@pytest.mark.parametrize("params", cond_models.values(), ids=list(cond_models.keys()))
def test_conductivity(params, baseline, plot_if_fails):
    configurations = [
        {'matrix_format': "ELL", 'optimal_size': False, 'interleaved': False},
        {'matrix_format': "ELL", 'optimal_size': True, 'interleaved': True},
    ]
    model = pb.Model(*params)

    kernel = pb.lorentz_kernel()
    strategies = [pb.kpm(model, energy_range=[-9, 9], kernel=kernel, silent=True, **c)
                  for c in configurations]

    energy = np.linspace(-2, 2, 25)
    results = [kpm.calc_conductivity(energy, broadening=0.5, temperature=0, num_points=200)
               for kpm in strategies]

    expected = results[0].with_data(baseline(results[0].data.astype(np.float32)))
    for i in range(len(results)):
        plot_if_fails(results[i], expected, "plot", label=i)

    for result in results:
        assert pytest.fuzzy_equal(result, expected, rtol=1e-2, atol=1e-5)
