import pytest

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import pybinding as pb


def pytest_addoption(parser):
    parser.addoption("--alwaysplot", action="store_true",
                     help="Plot even for tests which pass.")
    parser.addoption("--savebaseline", action="store_true",
                     help="Save a new baseline for all tests.")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item):
    """Allows fixtures to access test reports

    Adds test reports `rep_setup`, `rep_call`, `rep_teardown` to `request.node`.
    Required by the `plot` fixture to determine if a test failed.
    """
    outcome = yield
    report = outcome.get_result()
    setattr(item, "rep_" + report.when, report)
    return report


def _make_file_path(request, directory: str, name: str='', ext: str=''):
    basedir = request.fspath.join('..').join(directory)
    if not basedir.exists():
        basedir.mkdir()

    module = request.module.__name__.split('.')[-1].replace('test_', '')
    subdir = basedir.join(module)
    if not subdir.exists():
        subdir.mkdir()

    if not name:
        name = request.node.name.replace('test_', '')

    return subdir.join(name + ext)


@pytest.fixture
def baseline(request):
    """Return baseline data for this result. If non exist create it."""
    def get_expected(result, group=''):
        name = request.node.name.replace('test_', '')
        if group:
            # replace 'some[thing]' with 'group[thing]'
            part = name.partition('[')
            name = group + part[1] + part[2]

        file = _make_file_path(request, 'baseline_data', name, '.pbz')
        if not request.config.getoption("--savebaseline") and file.exists():
            return pb.load(file)
        else:
            pb.save(result, file)
            return result

    return get_expected


@pytest.yield_fixture
def plot(request):
    class Gather:
        def __init__(self):
            self.data = []

        def __call__(self, result, expected, method, *args, **kwargs):
            self.data.append(locals().copy())

        def plot(self, what):
            for d in self.data:
                getattr(d[what], d['method'])(*d['args'], **d['kwargs'])

            plt.title(what)
            pb.pltutils.legend()

    gather = Gather()
    yield gather

    if request.config.getoption("--alwaysplot") or request.node.rep_call.failed:
        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        gather.plot('result')

        plt.subplot(122)
        gather.plot('expected')

        file_path = _make_file_path(request, 'plots', ext='.png')
        plt.savefig(str(file_path))
        plt.close()


def fuzzy_equal(data, expected, rtol=1e-05, atol=1e-08):
    """Recursively compares structures of ndarrays using np.allclose()"""
    tol = rtol, atol
    if isinstance(data, np.ndarray):
        return np.allclose(data, expected, *tol)
    if isinstance(data, csr_matrix):
        return all(fuzzy_equal(getattr(data, s), getattr(expected, s), *tol)
                   for s in ['shape', 'data', 'indices', 'indptr'])
    if isinstance(data, (tuple, list)):
        return (len(data) == len(expected) and
                all(fuzzy_equal(a, b, *tol) for a, b in zip(data, expected)))
    if isinstance(data, dict):
        return (len(data) == len(expected) and
                all(fuzzy_equal(data[k], expected[k], *tol) for k in data.keys()))
    else:
        specials = [s for s in ['__getstate__', '__getinitargs__'] if hasattr(data, s)]
        if specials:
            return all(fuzzy_equal(getattr(data, s)(), getattr(expected, s)(), *tol)
                       for s in specials)
        else:
            return data == expected


def pytest_namespace():
    return {'fuzzy_equal': fuzzy_equal}
