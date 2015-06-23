import numpy as _np
import matplotlib.pyplot as _plt

import pybinding as _pb
from pybinding.utils import progressbar as _progressbar
from .data3d import Data3D
from _pybinding import KPMldos


class Execute(type):
    def __init__(cls, class_name, bases, namespace):
        super().__init__(class_name, bases, namespace)
        if len(bases) == 0:
            return  # only run in derived classes
        if 'init' not in namespace:
            return

        # get the "init" function arguments
        import inspect
        spec = inspect.getfullargspec(namespace['init'])
        spec.args.pop(0)  # remove the 'self' argument

        import os
        import atexit
        if "SUBMITTED_0X7D" in os.environ:  # previously submitted job
            arguments = Execute.extract_arg_values_from_environment(spec)
            atexit.register(lambda: cls(**arguments).run())
        else:
            arguments = Execute.extract_arg_values_from_defaults(spec)
            if 'deploy' not in namespace or namespace['deploy'] is None:  # run locally
                atexit.register(lambda: cls(**arguments).run())
            else:  # deploy the main script file to remote
                import sys
                namespace['deploy'](sys.argv[0], arguments)

    @staticmethod
    def extract_arg_values_from_environment(spec):
        """Get argument values from environment variables."""
        import os
        arguments = dict()
        for name in spec.args:
            # convert strings to proper arguments (based on annotated type)
            convert = spec.annotations.get(name, lambda x: x)
            arguments[name] = convert(os.environ[name])

        return arguments

    @staticmethod
    def extract_arg_values_from_defaults(spec):
        """Get argument values from method defaults."""
        missing_defaults = 0 if not spec.defaults else len(spec.args) - len(spec.defaults)
        if missing_defaults:
            raise Exception("Missing default value(s): " + ", ".join(spec.args[:missing_defaults]))

        arguments = {k: v for k, v in zip(spec.args, spec.defaults)}

        # check that all arguments have type annotations
        missing_types = [arg for arg in spec.args if arg not in spec.annotations]
        if missing_types:
            raise TypeError("Missing type annotation(s): " + ", ".join(missing_types))

        # the annotations must be valid callable objects
        invalid_types = [arg for arg, t in spec.annotations.items() if not callable(t)]
        if invalid_types:
            raise TypeError("Bad type annotation(s) for variables: " + ", ".join(invalid_types))

        return arguments


class Sweep(metaclass=Execute):
    def __init__(self, *args, **kwargs):
        self.constant = []
        self.variable = lambda var: []
        self.result = lambda var: None

        self.name = ""

        self.pbar = None
        self.progress = dict(
            show_bar=True,
            save_step=10,
            plot_step=10,
            file=None,
        )

        from pybinding.utils import cpuinfo
        self.num_threads = cpuinfo.physical_core_count()

        self.data = Data3D()
        self.range = _np.array([])

        self.init(*args, **kwargs)

        self.range = self.range.astype(_np.float32)
        if self.data.file_name != 'None':
            self.data.file_name = "{}.npz".format(self.name)
        else:
            self.data.file_name = None
        self.progress['file'] = "{}.log".format(self.name)

        def progress_set(step):
            from math import floor
            s = set(floor(p / 100 * self.range.size)
                    for p in _np.arange(0, 100, step))
            s.remove(0)  # don't waste time saving progress at zero
            s.add(self.range.size-1)  # make sure progress is saved on the last iteration
            return s

        self.progress['save_at'] = progress_set(self.progress['save_step'])
        self.progress['plot_at'] = progress_set(self.progress['plot_step'])

    def init(self, *args, **kwargs):
        pass

    def prepare_data(self):
        result = self.make_result(self.make_model(0), self.range[0])
        self.data.x = self.range
        self.data.y = result.energy.copy()
        self.data.z = _np.zeros((self.data.x.size, self.data.y.size), _np.float32)

    def report(self, result, job_id):
        self.data.z[job_id, :] = result.ldos

        report = result.report()
        var_name = self.data.plain_labels()['x']
        print("{:3}| {} = {:.2f}, {}".format(
            self.pbar.currval + 1, var_name, self.range[job_id], report))

        self.save_progress(self.pbar.currval)
        self.pbar += 1

    def save_progress(self, iteration):
        if iteration in self.progress['save_at']:
            self.data.save()
            self.progress['save_at'].remove(iteration)

        if iteration in self.progress['plot_at']:
            data = self.data.copy()
            data = self.modify_plot_data(data)
            try:
                self.plot(data)
            except Exception as err:
                print("Plot Error: {}".format(err))
                self.pbar.update()
            _plt.close()
            self.progress['plot_at'].remove(iteration)

    @staticmethod
    def modify_plot_data(data: Data3D):
        data.interpolate(multiply=(2, 1))
        return data

    def plot(self, data: Data3D):
        data.plot()
        _plt.savefig('{}.png'.format(self.name))

    def make_model(self, job_id):
        from pybinding.utils import to_tuple
        var = self.range[job_id]
        params = to_tuple(self.constant) + to_tuple(self.variable(var))
        return _pb.Model(*params)

    def make_result(self, model, job_id):
        return self.result(model, self.range[job_id])

    def run(self):
        self.prepare_data()

        import sys
        self.pbar = _progressbar.Range(
            self.range.size, file_name=self.progress['file'],
            output=(sys.stdout if self.progress['show_bar'] else None)
        ).start()

        from pybinding.utils import cpuinfo
        print('\n', cpuinfo.name(), '\n', cpuinfo.threads(), '\n', sep='')
        print(self.make_model(self.range[0]).report(), '\n')
        self.pbar.update()

        from _pybinding import parallel_sweep
        parallel_sweep(len(self.range), self.num_threads, self.num_threads,
                       self.make_model, self.make_result, self.report)
        self.pbar.finish()
