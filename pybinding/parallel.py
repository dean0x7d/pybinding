"""Multi-threaded functions for parameter sweeps"""
import sys
import inspect
import itertools
from copy import copy
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from pybinding.support.inspect import get_call_signature
from . import _cpp
from .utils import cpuinfo, progressbar, decorator_decorator
from .results import Sweep, NDSweep

__all__ = ['num_cores', 'parallel_for', 'parallelize', 'sweep', 'ndsweep']

num_cores = cpuinfo.physical_core_count()


def _sequential_for(sequence, produce, retire):
    """Simple single-threaded for loop"""
    for idx, var in enumerate(sequence):
        deferred = produce(var)
        deferred.compute()
        retire(deferred, idx)


def _parallel_for(sequence, produce, retire, num_threads=num_cores, queue_size=num_cores):
    """Multi-threaded for loop

    See the implementation of `_sequential_for` to get the basic idea. This parallel
    version is functionally identical but executes on multiple threads simultaniously
    thanks to C++. The `produce` function must return a Deferred compute object which
    has a `compute()` method and a `result` field.

    Everything is implemented in C++. This is just a wrapper which sets the default
    values of `num_threads` and `queue_size` to the number of physical cores.

    Parameters
    ----------
    sequence : array_like
        The for loop will iterate over this.
    produce : callable
        Takes a value from `sequence` and returns a `Deferred` compute object.
    retire : callable
        Takes the computed `Deferred` object and 'idx' which indicates the index
        of the value in `sequence` which was just computed.
    num_threads : int
        Number of thread that will run in parallel.
    queue_size : int
        Number of `Deferred` jobs to be queued up for consumption by the worker
        threads. The maximum number of jobs that will be kept in memory at any
        one time will be `queue_size` + `num_threads`.

    Examples
    --------
    ::

        def produce(var):
            model = pb.Model(...)  # something that depends on var
            greens = pb.greens.kpm(model)
            return greens.deferred_ldos(...)  # may also depend on var

        def retire(deferred, idx):
            print(deferred.result)

        _parallel_for(np.linspace(0, 1, 50), produce, retire)
    """
    _cpp.parallel_for(sequence, produce, retire, num_threads, queue_size)


class Hooks:
    """Holds functions which hook into `ParallelFor`

    Attributes
    ----------
    first : list of callable
        Called only once after the first `Deferred` is produced.
    status : list of callable
        Called every time a `Deferred` job is computed. As arguments it takes
        a `report` string, `idx` of the original value and `count` the number
        of job that have been computed so far.
    plot : list of callable
        Called once in a while with a `result` argument to be plotted.
    """
    def __init__(self):
        self.first = []
        self.status = []
        self.plot = []


class Config:
    """Configuration variables for `ParallelFor`

    Attributes
    ----------
    callsig : CallSignature
        Signature of the function call which made the parallel `Factory`.
        Used for automatic configuration.
    filename : str
        The name of the file (without an extension) for various files which will be
        produced. The computed data will be saved with the '.pbz' extension, plots
        with '.png', progress log with '.log', etc.
    num_threads, queue_size : int
        Forwarded to `_parallel_for`.
    save_every : float
        A 0 to 100 percentage points interval to save and plot the data.
    pbar_fd : {sys.stdout, sys.stderr, None}
        Output stream. The progress bar is always the last line of output.
    """
    def __init__(self, callsig, num_threads, queue_size):
        self.callsig = callsig
        self.num_threads = num_threads
        self.queue_size = queue_size

        self.filename = self.make_filename(callsig)
        self.save_every = 10.0
        self.pbar_fd = sys.stdout

    def make_save_set(self, total):
        save_at = {int(total * p) for p in np.arange(0, 1, self.save_every / 100)}
        save_at.remove(0)
        save_at.add(total)  # make sure progress is saved on the last iteration
        return save_at

    @staticmethod
    def make_filename(callsig):
        invalid_chars = " /.,"
        filename = "".join("{:.1s}{}".format(k, v) for k, v in callsig.named_args.items())
        if not filename:
            filename = "data"
        return "".join(c for c in filename if c not in invalid_chars)


class DefaultStatus:
    """Default status reporter"""
    def __init__(self, params, sequence):
        self.params = params
        self.sequence = sequence

        size = len(sequence)
        count_width = len(str(size))
        vars_width = max(len(self._vars(idx)) for idx in range(size))
        self.template = "{{count:{}}}| {{vars:{}}} | {{report}}".format(count_width, vars_width)

    def _vars(self, idx):
        return ", ".join("{} = {:.2g}".format(k, v)
                         for k, v in zip(self.params, self.sequence[idx]))

    def __call__(self, deferred, idx, count):
        report = deferred.solver.report(shortform=True)
        print(self.template.format(vars=self._vars(idx), **locals()))


class Factory:
    """Produces `Deferred` jobs for `ParallelFor`

    Attributes
    ----------
    variables : tuple of array_like
        Parameters which change while iterating.
    fixtures : dict
        Constant parameters.
    sequence : list
        Product of `variables`. The loop will iterate over its values.
    produce : callable
        Takes a value from `sequence` and returns a `Deferred` compute object.
    config : Config
    hooks : Hooks
    """
    def __init__(self, variables, fixtures, produce, config):
        self.variables = variables
        self.fixtures = fixtures
        self.produce = produce
        self.config = config

        self.sequence = list(itertools.product(*variables))

        self.hooks = Hooks()
        self.hooks.status.append(DefaultStatus(
            inspect.signature(self.produce).parameters, self.sequence
        ))


class ParallelFor:
    """Keep track of progress while running `_parallel_for`

    Parameters
    ----------
    factory : Factory
        Produces Deferred compute kernels.
    make_result : callable
        Creates the final result from raw data. See `_make_result` prototype.
    """
    def __init__(self, factory, make_result=None):
        self.factory = factory
        self.hooks = factory.hooks
        self.config = factory.config

        if make_result:
            self._make_result = make_result

        size = len(factory.sequence)
        self.save_at = self.config.make_save_set(size)

        logname = self.config.filename + ".log" if self.config.filename else ""
        self.pbar = progressbar.ProgressBar(size, stream=self.config.pbar_fd, filename=logname)

        if self.config.num_threads == 1:
            self.loop = _sequential_for
        else:
            self.loop = partial(_parallel_for, num_threads=self.config.num_threads,
                                queue_size=self.config.queue_size)

        self.called_first = False
        self.result = None
        self.data = [None] * size

    @staticmethod
    def _make_result(data):
        return data

    def _produce(self, var):
        deferred = self.factory.produce(*var, **self.factory.fixtures)

        if not self.called_first:
            self._first(deferred)
            self.called_first = True
            self.pbar.refresh()

        return deferred

    def _first(self, deferred):
        for f in self.hooks.first:
            f(deferred)

    def _retire(self, deferred, idx):
        self.data[idx] = copy(deferred.result)

        count = self.pbar.value + 1
        self._status(deferred, idx, count)
        self.pbar += 1  # also refreshes output stream

        if count in self.save_at:
            result = self._make_result(self.data)
            self.result = copy(result)  # _plot() may modify the local
            self._save(result)
            self._plot(result)

    def _status(self, deferred, idx, count):
        for f in self.hooks.status:
            f(deferred, idx, count)

    def _save(self, result):
        if not self.config.filename:
            return

        from .support.pickle import save
        save(result, self.config.filename)

    def _plot(self, result):
        if not self.config.filename:
            return

        try:
            if self.hooks.plot:
                for f in self.hooks.plot:
                    f(result)
                plt.savefig(self.config.filename + ".png")
                plt.close()
        except Exception as err:
            print(err)

    def __call__(self):
        self.called_first = False
        with self.pbar:
            self.loop(self.factory.sequence, self._produce, self._retire)
        return self.result


def parallel_for(factory, make_result=None):
    """Multi-threaded loop feed by the `factory` function

    Parameters
    ----------
    factory : :func:`Factory <parallelize>`
        Factory function created with the :func:`parallelize` decorator.
    make_result : callable, optional
        Creates the final result from raw data. This result is also the
        final return value of :func:`parallel_for`.

    Returns
    -------
    array_like
        A result for each loop iteration.

    Examples
    --------
    ::

        @parallelize(x=np.linspace(0, 1, 10))
        def factory(x):
            pb.Model(...)  # depends on `x`
            greens = pb.greens.kpm(model)
            return greens.deferred_ldos(...)  # may also depend on `x`

        results = parallel_for(factory)
    """
    return ParallelFor(factory, make_result)()


@decorator_decorator
def parallelize(num_threads=num_cores, queue_size=num_cores, **kwargs):
    """parallelize(num_threads=num_cores, queue_size=num_cores, **kwargs)

    A decorator which creates factory functions for :func:`parallel_for`

    The decorated function must return a `Deferred` compute kernel.

    Parameters
    ----------
    num_threads : int
        Number of threads that will run in parallel. Defaults to the number of
        cores in the current machine.
    queue_size : int
        Number of `Deferred` jobs to be queued up for consumption by the worker
        threads. The maximum number of jobs that will be kept in memory at any
        one time will be `queue_size` + `num_threads`.
    **kwargs
        Variables which will be iterated over in :func:`parallel_for`
        and passed to the decorated function. See example.

    Examples
    --------
    ::

        @parallelize(a=np.linspace(0, 1, 10), b=np.linspace(-2, 2, 10))
        def factory(a, b):
            pb.Model(...)  # depends on `a` and `b`
            greens = pb.greens.kpm(model)
            return greens.deferred_ldos(...)  # may also depend on `a` and `b`

        results = parallel_for(factory)
    """
    callsig = kwargs.pop('callsig', None)
    if not callsig:
        callsig = get_call_signature(up=2)

    def decorator(produce_func):
        params = inspect.signature(produce_func).parameters

        variables = tuple(kwargs[k] for k in params if k in kwargs)
        fixtures = {k: v.default for k, v in params.items() if k not in kwargs}

        return Factory(variables, fixtures, produce_func,
                       Config(callsig, num_threads, queue_size))

    return decorator


def sweep(factory, plot=lambda r: r.plot(), labels=None, tags=None, silent=False):
    """Do a multi-threaded parameter sweep

    Parameters
    ----------
    factory : :func:`Factory <parallelize>`
        Factory function created with the :func:`parallelize` decorator.
    plot : callable
        Plotting functions which takes a :class:`.Sweep` result as its only argument.
    labels, tags : dict
        Forwarded to :class:`.Sweep` object.
    silent : bool
        Don't print status messages.

    Returns
    -------
    :class:`~pybinding.Sweep`
    """
    x = factory.variables[0]
    energy = factory.fixtures['energy']
    zero = np.zeros_like(energy, np.float32)

    def make_result(data):
        sweep_data = np.vstack(v.squeeze() if v is not None else zero for v in data)
        return Sweep(x, energy, sweep_data, labels, tags)

    if silent:
        factory.hooks.status.clear()
    if plot:
        factory.hooks.plot.append(plot)

    return parallel_for(factory, make_result)


def ndsweep(factory, plot=None, labels=None, tags=None, silent=False):
    """Do a multi-threaded n-dimensional parameter sweep

    Parameters
    ----------
    factory : :func:`Factory <parallelize>`
        Factory function created with the :func:`parallelize` decorator.
    plot : callable
        Plotting functions which takes a :class:`.NDSweep` result as its only argument.
    labels, tags : dict
        Forwarded to :class:`.NDSweep` object.
    silent : bool
        Don't print status messages.

    Returns
    -------
    :class:`~pybinding.NDSweep`
    """
    energy = factory.fixtures['energy']
    variables = factory.variables + (energy,)
    zero = np.zeros_like(energy, np.float32)

    def make_result(data):
        sweep_data = np.vstack(v.squeeze() if v is not None else zero for v in data)
        return NDSweep(variables, sweep_data, labels, tags)

    if silent:
        factory.hooks.status.clear()
    if plot:
        factory.hooks.plot.append(plot)

    return parallel_for(factory, make_result)
