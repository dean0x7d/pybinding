import pytest

import math
import numpy as np
import pybinding as pb
from pybinding.repository import graphene


polygons = {
    'triangle': pb.regular_polygon(3, radius=1.1),
    'triangle90': pb.regular_polygon(3, radius=1.1, angle=math.pi/2),
    'diamond': pb.regular_polygon(4, radius=1),
    'square': pb.regular_polygon(4, radius=1, angle=math.pi/4),
    'pentagon': pb.regular_polygon(5, radius=1),
}


@pytest.fixture(scope='module', ids=list(polygons.keys()), params=polygons.values())
def polygon(request):
    return request.param


def test_polygon_api():
    with pytest.raises(RuntimeError) as excinfo:
        pb.Polygon([0, 0], [0, 1])
    assert "at least 3 sides" in str(excinfo.value)


def test_polygon(polygon, baseline, plot_if_fails):
    model = pb.Model(graphene.monolayer(), polygon)
    expected = baseline(model.system)
    plot_if_fails(model.system, expected, 'plot')
    plot_if_fails(polygon, polygon, 'plot')
    assert pytest.fuzzy_equal(model.system, expected, 1.e-4, 1.e-6)


def test_freeform(baseline, plot_if_fails):
    def donut(inner_radius, outer_radius):
        def contains(x, y, _):
            r = np.sqrt(x**2 + y**2)
            return np.logical_and(inner_radius < r, r < outer_radius)

        return pb.FreeformShape(contains, width=[2 * outer_radius, 2 * outer_radius])

    assert pytest.fuzzy_equal(
        donut(0.5, 1).vertices,
        [[ 1, 1, 0], [ 1, 1, 0], [ 1, -1, 0], [ 1, -1, 0],
         [-1, 1, 0], [-1, 1, 0], [-1, -1, 0], [-1, -1, 0]]
    )

    shape = donut(0.6, 1.1)
    model = pb.Model(graphene.monolayer(), shape)
    expected = baseline(model.system)
    plot_if_fails(model.system, expected, 'plot')
    plot_if_fails(shape, shape, 'plot')
    assert pytest.fuzzy_equal(model.system, expected, 1.e-4, 1.e-6)


def test_freeform_plot():
    def sphere(radius):
        def contains(x, y, z):
            r = np.sqrt(x**2, y**2, z**2)
            return r < radius

        return pb.FreeformShape(contains, width=[2 * radius] * 3)

    with pytest.raises(RuntimeError) as excinfo:
        sphere(1).plot()
    assert "only works for 2D shapes" in str(excinfo.value)
