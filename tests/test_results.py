import pytest

import numpy as np
import pybinding as pb


def test_sweep():
    x0 = np.arange(3)
    y0 = np.arange(-1, 2)
    data0 = np.arange(9).reshape((3, 3))
    sweep = pb.results.Sweep(x0, y0, data0, title="test sweep", misc=dict(b=1, c=2),
                             labels=dict(x="$\\alpha$", y=r"$\beta$ (eV)", data=r"$\gamma$"))

    assert sweep.plain_labels == dict(x="alpha", y="beta (eV)", data="gamma")

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
