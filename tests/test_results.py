import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


@pytest.fixture(scope='module')
def model():
    return pb.Model(graphene.monolayer(), pb.rectangle(1))


def test_sweep():
    x0 = np.arange(3)
    y0 = np.arange(-1, 2)
    data0 = np.arange(9).reshape((3, 3))
    sweep = pb.results.Sweep(
        x0, y0, data0, tags=dict(b=1, c=2),
        labels=dict(title="test sweep", x="$\\alpha$", y=r"$\beta$ (eV)", data=r"$\gamma$")
    )

    assert sweep.plain_labels == dict(title="test sweep", x="alpha", y="beta (eV)", data="gamma")

    xgrid, ygrid = sweep.xy_grids()
    assert np.all(xgrid == [[v] * 3 for v in x0])
    assert np.all(ygrid == [y0] * 3)

    tmp = sweep.copy()
    tmp.crop(x=[0, 1], y=[0, 1])
    assert np.all(tmp.x == [0, 1]) and np.all(tmp.y == [0, 1])
    assert np.all(tmp.data == [[1, 2], [4, 5]])

    tmp = sweep.copy()
    tmp.mirror(axis='x')
    assert np.all(tmp.x == [-2, -1, 0, 1, 2])
    assert np.all(tmp.data == [[6, 7, 8], [3, 4, 5], [0, 1, 2], [3, 4, 5], [6, 7, 8]])

    s, x = sweep.slice_x(1.2)
    assert np.all(s == [3, 4, 5]) and x == 1
    s, y = sweep.slice_y(0.4)
    assert np.all(s == [1, 4, 7]) and y == 0


def test_spatial_map(model):
    system = model.system
    zeros = np.zeros_like(system.positions.x)

    spatial_map = pb.results.SpatialMap.from_system(zeros, system)

    assert system.positions.x.data == spatial_map.pos.x.data
    assert system.positions.y.data == spatial_map.pos.y.data
    assert system.positions.z.data == spatial_map.pos.z.data
    assert system.sublattices.data == spatial_map.sub.data

    tmp = spatial_map.copy()
    tmp.filter(tmp.sub == 0)
    assert len(spatial_map.pos.x) == 2 * len(tmp.pos.x)

    tmp = spatial_map.copy()
    tmp.crop(x=(-0.1, 0.1), y=(0, 0.1))
    assert len(tmp.pos.x) == 1


def test_structure_map(model):
    system = model.system
    zeros = np.zeros_like(system.positions.x)

    spatial_map = pb.results.SpatialMap.from_system(zeros, system)
    structure_map = pb.results.StructureMap.from_system(zeros, system)

    assert pytest.fuzzy_equal(spatial_map, structure_map.spatial_map)

    tmp = structure_map.copy()
    tmp.filter(tmp.pos.x < 0.05)
    assert structure_map.hoppings.nnz == 41
    assert tmp.hoppings.nnz == 21


def test_structure_map_plot(compare_figure):
    model = pb.Model(graphene.monolayer(), pb.rectangle(0.8))
    system = model.system
    data = np.arange(system.num_sites)
    structure_map = pb.results.StructureMap.from_system(data, system)

    with compare_figure() as chk:
        structure_map.plot_structure(site_radius=(0.03, 0.05), cbar_props=False)
    assert chk.passed
