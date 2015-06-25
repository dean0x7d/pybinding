import pytest
import numpy as np
import matplotlib.pyplot as plt


def pytest_addoption(parser):
    parser.addoption("--alwaysplot", action="store_true", help="Plot even for tests which pass.")


# noinspection PyUnusedLocal
@pytest.mark.tryfirst
def pytest_runtest_makereport(item, call, __multicall__):
    """This allows fixtures to access test reports"""
    # adds reports for 'setup', 'call', 'teardown' with 'rep_' prefix
    rep = __multicall__.execute()
    setattr(item, 'rep_' + rep.when, rep)
    return rep


def _make_file_path(request, directory: str, extenstion: str):
    basedir = request.fspath.join('..').join(directory)
    if not basedir.exists():
        basedir.mkdir()

    module = request.module.__name__.split('.')[-1].replace('test_', '')
    subdir = basedir.join(module)
    if not subdir.exists():
        subdir.mkdir()

    test_name = request.node.name.replace('test_', '')
    file_path = subdir.join(test_name + extenstion)
    return file_path


@pytest.fixture
def baseline(request):
    file_path = _make_file_path(request, 'baseline_data', '.pbz')

    def func(generated):
        if file_path.exists():
            return generated.__class__.from_file(str(file_path))
        else:
            generated.save(str(file_path))
            return generated

    return func


@pytest.yield_fixture
def plot(request):
    class Gather:
        def __call__(self, plot_result, plot_expected, *args, **kwargs):
            self.__dict__.update(**locals())

    d = Gather()
    yield d

    if request.config.getoption("--alwaysplot") or request.node.rep_call.failed:
        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        # noinspection PyUnresolvedReferences
        d.plot_result(*d.args, **d.kwargs)
        plt.title("result")

        plt.subplot(122)
        # noinspection PyUnresolvedReferences
        d.plot_expected(*d.args, **d.kwargs)
        plt.title("expected")

        file_path = _make_file_path(request, 'plots', '.png')
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
