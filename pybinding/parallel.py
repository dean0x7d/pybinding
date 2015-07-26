import sys

import numpy as np
import matplotlib.pyplot as plt

import _pybinding
from .utils import cpuinfo, progressbar
from .results import Sweep

__all__ = ['num_cores', 'sweep']

num_cores = cpuinfo.physical_core_count()


def _plain_sweep(variables, produce, report, num_threads=num_cores, queue_size=num_cores):
    """Execute a multi-threaded parameter sweep.

    Runs over a list of 'variables' and gives each one to a 'produce' function which returns
    a Deferred compute object. The computations are executed in parallel. The result is given
    back to the 'report' function.

    Implementation: Everything is in C++. This is just a wrapper which sets num_threads and
                    and queue_size default values to the number of physical cores.

    Parameters
    ----------
    variables : array_like
        Main parameter which is being swept.
    produce : callable(var)
        Function which takes a values from 'variables' and returns a Deferred compute job.
    report : callable(deferred, index)
        Function which processes the computed Deferred job from 'produce'. The 'index' is the
        original location of this result in 'variables'.
    num_thread : int
        Number of thread that will run in parallel.
    queue_size : int
        Number of Deferred jobs to be queued up for consumption by the worker threads.
        The total number of jobs in memory will be queue_size + num_threads.

    Example
    -------
    def produce(var):
        model = pb.Model(...)  # something that depends on var
        greens = pb.greens.kpm(model)
        return greens.deferred_ldos(...)  # may also depend on var

    def report(deferred, index):
        print(deferred.result)

    _plain_sweep(np.linspace(0, 1, 50), produce, report)
    """
    _pybinding.sweep(variables, produce, report, num_threads, queue_size)


def _progressbar_sweep(variables, produce, report, pbar_fd=sys.stdout, log_file="", **kwargs):
    """Just like '_pain_sweep' but with a nifty progress bar.

    Parameters
    ----------
    fd : {sys.stdout, sys.stderr, None}
        Output stream. The progress bar is always the last line of output.
    log_file : str
        Also write all output to a file. The progressbar is always the first line of the file.
    """
    pbar = progressbar.Range(len(variables), fd=pbar_fd, file_name=log_file)

    def _produce(var):
        deferred = produce(var)
        pbar.update()
        return deferred

    def _report(deferred, job_id):
        try:
            report(deferred, job_id)
        except Exception as err:
            print(err)
        pbar.update(pbar.currval + 1)

    pbar.start()
    _plain_sweep(variables, _produce, _report, **kwargs)
    pbar.finish()


def sweep(variables, produce, report=None, first=None, file="", save_every=10,
          plot=lambda r: r.plot(), title="", labels: dict=None, tags: dict=None, **kwargs):
    """Do a multi-threaded parameter sweep and return a 'Sweep' result.

    Runs over a list of 'variables' and gives each one to a 'produce' function which returns
    a Deferred compute object. The computations are executed in parallel. A 'Sweep' result is
    returned. A preview of the computed data may be plotted during execution.

    Parameters
    ----------
    variables : array_like
        Main parameter which is being swept.
    produce : callable(var)
        Function which takes a values from 'variables' and returns a Deferred compute job.
    report : callable(deferred, index)
        Function which processes the computed Deferred job from 'produce'. The 'index' is the
        original location of this result in 'variables'.
    first : callable(deferred)
        Called only once after the first job is produced. May be used for printing information.
    file : str
        The name of the file (without an extension) where the computed data will be saved ('.pbz')
        and plotted ('.png') in regular intervals during execution.
    save_every : float
        A 0 to 100 percentage points interval to save and plot the data.
    plot : callable(result)
        Custom plot function.
    labels, tags : dict
        Passed to Sweep object.

    Example
    -------
    def produce(var):
        model = pb.Model(...)  # something that depends on var
        greens = pb.greens.kpm(model)
        return greens.deferred_ldos(...)  # may also depend on var

    result = sweep(np.linspace(0, 1, 50), produce)
    """
    result = Sweep(variables, 0, 0, title, labels, tags)

    last = len(variables) - 1
    save_at = {(last * p) // 100 for p in np.arange(save_every, 100, save_every)}  # skip zero
    save_at |= {last}  # make sure progress is saved on the last iteration
    silent = kwargs.pop('silent', False)

    def _produce(var):
        deferred = produce(var)
        if result.data.shape == (1, 1):
            result.y = deferred.y
            result.data = np.zeros((len(variables), len(deferred.y)), np.float32)
            if first:
                first(deferred)
        return deferred

    def save_progress():
        if file:
            result.save(file + ".pbz")

        if file and plot:
            plot(result)
            plt.savefig(file + ".png")
            plt.close()

    class Report:
        count = 0

        def __call__(self, deferred, job_id):
            self.count += 1
            result.data[job_id, :] = deferred.result

            if not silent:
                print("{step:3}| {name} = {value:.2f}, {message}".format(
                    step=self.count,
                    name=result.plain_labels['x'],
                    value=variables[job_id],
                    message=deferred.report
                ))

            if report:
                report(deferred, job_id)
            if self.count - 1 in save_at:
                save_progress()

    _progressbar_sweep(variables, _produce, Report(), **kwargs)
    return result
