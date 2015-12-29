import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


def test_freeform(baseline, plot_if_fails):
    def donut(inner_radius, outer_radius):
        def contains(x, y, _):
            r = np.sqrt(x**2 + y**2)
            return np.logical_and(inner_radius < r, r < outer_radius)

        return pb.FreeformShape(contains, width=[2 * outer_radius, 2 * outer_radius])

    assert pytest.fuzzy_equal(
        donut(0.5, 1).bbox_vertices,
        [[ 1, 1, 0], [ 1, 1, 0], [ 1, -1, 0], [ 1, -1, 0],
         [-1, 1, 0], [-1, 1, 0], [-1, -1, 0], [-1, -1, 0]]
    )

    model = pb.Model(graphene.lattice.monolayer(), donut(0.6, 1.1))
    expected = baseline(model.system)
    plot_if_fails(model.system, expected, 'plot')
    assert pytest.fuzzy_equal(model.system, expected)


def test_freeform_plot():
    def sphere(radius):
        def contains(x, y, z):
            r = np.sqrt(x**2, y**2, z**2)
            return r < radius

        return pb.FreeformShape(contains, width=[2 * radius] * 3)

    with pytest.raises(RuntimeError) as excinfo:
        sphere(1).plot()
    assert "only works for 2D shapes" in str(excinfo.value)
