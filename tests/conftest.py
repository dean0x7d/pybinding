import pytest
import numpy as np
import matplotlib.pyplot as plt


def pytest_addoption(parser):
    parser.addoption("--alwaysplot", action="store_true",
                     help="Plot even for tests which pass.")
    parser.addoption("--savebaseline", action="store_true",
                     help="Save a new baseline for all tests.")


# noinspection PyUnusedLocal
@pytest.mark.tryfirst
def pytest_runtest_makereport(item, call, __multicall__):
    """This allows fixtures to access test reports"""
    # adds reports for 'setup', 'call', 'teardown' with 'rep_' prefix
    rep = __multicall__.execute()
    setattr(item, 'rep_' + rep.when, rep)
    return rep


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
    def get_expected(result, group=''):
        isarray = isinstance(result, np.ndarray)

        name = request.node.name.replace('test_', '')
        if group:
            part = name.partition('[')
            name = group + part[1] + part[2]

        ext = '.npz' if isarray else '.pbz'
        file_path = _make_file_path(request, 'baseline_data', name, ext)
        file_str = str(file_path)

        if not request.config.getoption("--savebaseline") and file_path.exists():
            if isarray:
                return np.load(file_str)['data']
            else:
                return result.__class__.from_file(file_str)
        else:
            if isarray:
                np.savez_compressed(file_str, data=result)
            else:
                result.save(file_str)
            return result

    return get_expected


@pytest.yield_fixture
def plot(request):
    class Gather:
        def __call__(self, result, expected, method, *args, **kwargs):
            self.__dict__.update(**locals())

        def plot(self, what):
            d = self.__dict__
            if what in d:
                getattr(d[what], d['method'])(*d['args'], **d['kwargs'])

    gather = Gather()
    yield gather

    if request.config.getoption("--alwaysplot") or request.node.rep_call.failed:
        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        gather.plot('result')
        plt.title("result")

        plt.subplot(122)
        gather.plot('expected')
        plt.title("expected")

        file_path = _make_file_path(request, 'plots', ext='.png')
        plt.savefig(str(file_path))
        plt.close()


def fuzzy_equal(data, expected, rtol=1e-05, atol=1e-08):
    """Recusively compares strutures of ndarrays using np.allclose()"""
    tol = rtol, atol
    if isinstance(data, np.ndarray):
        return np.allclose(data, expected, *tol)
    if isinstance(data, (tuple, list)):
        return (len(data) == len(expected) and
                all(fuzzy_equal(a, b, *tol) for a, b in zip(data, expected)))
    if isinstance(data, dict):
        return (len(data) == len(expected) and
                all(fuzzy_equal(a, b, *tol) for a, b in zip(data.values(), expected.values())))
    elif hasattr(data, '__getstate__'):
        return fuzzy_equal(data.__getstate__(), expected.__getstate__(), *tol)
    else:
        return data == expected


def pytest_namespace():
    return {'fuzzy_equal': fuzzy_equal}
