import pytest

from contextlib import suppress

import matplotlib as mpl
mpl.use('Agg')  # disable `plt.show()` popup window during testing
import matplotlib.pyplot as plt

import pybinding as pb

from .utils.path import path_from_fixture
from .utils.compare_figures import CompareFigure
from .utils.fuzzy_equal import FuzzyEqual, FuzzyReport


def pytest_namespace():
    return {'fuzzy_equal': FuzzyEqual}


def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, FuzzyReport) and right is not None and op == "==":
        return left.explanation


def pytest_addoption(parser):
    parser.addoption("--alwaysplot", action="store_true",
                     help="Plot even for tests which pass.")
    parser.addoption("--savebaseline", action="store_true",
                     help="Save a new baseline for all tests.")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item):
    """Allows fixtures to access test reports

    Adds test reports `rep_setup`, `rep_call`, `rep_teardown` to `request.node`.
    Required by the `plot_if_fails` fixture to determine if a test failed.
    """
    outcome = yield
    report = outcome.get_result()
    setattr(item, "rep_" + report.when, report)
    return report


@pytest.fixture
def baseline(request):
    """Return baseline data for this result. If non exist create it."""
    def get_expected(result, group=''):
        file = path_from_fixture(request, prefix='baseline_data', ext='.pbz',
                                 override_group=group)

        if not request.config.getoption("--savebaseline") and file.exists():
            return pb.load(file)
        else:
            pb.save(result, file)
            return result

    return get_expected


@pytest.fixture
def compare_figure(request):
    """Compare a figure to a baseline image"""
    return CompareFigure(request)


@pytest.yield_fixture
def plot_if_fails(request):
    """This fixture will plot the actual and expected data if the test fails"""
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

    figure_path = path_from_fixture(request, prefix='failed', ext='.png')
    if request.config.getoption("--alwaysplot") or request.node.rep_call.failed:
        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        gather.plot('result')

        plt.subplot(122)
        gather.plot('expected')

        plt.savefig(str(figure_path))
        plt.close()
    elif figure_path.exists():
        with suppress(OSError):
            figure_path.unlink()
