import inspect
from collections import OrderedDict

__all__ = ['CallSignature', 'get_call_signature']


class CallSignature:
    """Holds a function and the arguments it was called with"""
    def __init__(self, function: callable, positional: OrderedDict,
                 args: tuple, keyword_only: OrderedDict, kwargs: dict):
        self.positional = positional
        self.args = args
        self.keyword_only = keyword_only
        self.kwargs = kwargs
        self.function = function

    @property
    def named_args(self):
        """All arguments except *args"""
        ordered = OrderedDict()
        for x in self.positional, self.keyword_only, self.kwargs:
            ordered.update(x)
        return ordered

    def _format_args(self, func):
        """Apply `func` to each argument value and return the formatted string"""
        if self.args:
            positional = [func(v) for k, v in self.positional.items()]
        else:
            positional = ["{}={}".format(k, func(v)) for k, v in self.positional.items()]

        args = [func(v) for v in self.args]
        keywords_only = ["{}={}".format(k, func(v)) for k, v in self.keyword_only.items()]
        kwargs = ["{}={}".format(k, func(v)) for k, v in self.kwargs.items()]

        return ", ".join(positional + args + keywords_only + kwargs)

    def __str__(self):
        return "{}({})".format(self.function.__name__, self._format_args(str))

    def __repr__(self):
        return "{}({})".format(self.function.__qualname__, self._format_args(repr))


def _find_callable(name, frame):
    """Find the callable which matches the name and frame code"""
    func = frame.f_globals.get(name)
    if func and inspect.isfunction(func) and func.__code__ is frame.f_code:
        return func

    if frame.f_back:
        func = frame.f_back.f_locals.get(name)
        if func and inspect.isfunction(func) and func.__code__ is frame.f_code:
            return func

    import gc  # try brute force
    for obj in gc.get_objects():
        if inspect.isfunction(obj) and obj.__code__ is frame.f_code:
            return obj

    raise RuntimeError("Can't find callable '{}()'".format(name))


def get_call_signature(up=0):
    """Return a CallSignature of the function currently being executed or `up` a few frames

    Parameters
    ----------
    up : int, optional
        Selects which call frame to inspect: number of frames up from the current one.

    Examples
    --------
    >>> # noinspection PyUnusedLocal
    ... def hello(a, b=8, *args, target=1, **kwargs):
    ...    # noinspection PyUnusedLocal
    ...    def world(x=1):
    ...        return get_call_signature(target)
    ...    return world()
    >>> hello(7)
    hello(a=7, b=8, target=1)
    >>> hello(7, 8, d=2)
    hello(a=7, b=8, target=1, d=2)
    >>> hello(7, 8, 9, d=2)
    hello(7, 8, 9, target=1, d=2)
    >>> hello(7, target=0)
    hello.<locals>.world(x=1)
    >>> str(hello(7, target=0))
    'world(x=1)'

    >>> def outer(x):
    ...    # noinspection PyUnusedLocal
    ...    def inner(y=x):
    ...        return get_call_signature(0)
    ...    return inner
    >>> outer(2)()
    outer.<locals>.inner(y=2)
    """
    stack = inspect.stack()
    try:
        frame, _, _, func_name, *_ = stack[up + 1]
    except:
        raise IndexError("Stack frame out of range")

    if func_name == '<module>':
        raise IndexError("Can't inspect a module")

    _, args_name, kwargs_name, frame_locals = inspect.getargvalues(frame)
    function = _find_callable(func_name, frame)
    params = inspect.signature(function).parameters

    positional = OrderedDict([(name, frame_locals[name])
                              for name, param in params.items()
                              if param.kind == param.POSITIONAL_OR_KEYWORD])
    args = frame_locals.get(args_name, ())

    keyword_only = OrderedDict([(name, frame_locals[name])
                                for name, param in params.items()
                                if param.kind == param.KEYWORD_ONLY])
    kwargs = frame_locals.get(kwargs_name, {})

    return CallSignature(function, positional, args, keyword_only, kwargs)
