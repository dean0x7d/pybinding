import os
from functools import wraps
from contextlib import contextmanager

import numpy as np

from ..support.inspect import get_call_signature


def to_tuple(o):
    try:
        return tuple(o)
    except TypeError:
        return (o,) if o is not None else ()


def to_list(o):
    try:
        return list(o)
    except TypeError:
        return [o] if o is not None else []


def with_defaults(options: dict, defaults_dict: dict=None, **defaults_kwargs):
    """Return a dict where missing keys are filled in by defaults

    >>> options = dict(hello=0)
    >>> with_defaults(options, hello=4, world=5) == dict(hello=0, world=5)
    True
    >>> defaults = dict(hello=4, world=5)
    >>> with_defaults(options, defaults) == dict(hello=0, world=5)
    True
    >>> with_defaults(options, defaults, world=7, yes=3) == dict(hello=0, world=5, yes=3)
    True
    """
    options = options if options else {}
    if defaults_dict:
        options = dict(defaults_dict, **options)
    return dict(defaults_kwargs, **options)


def x_pi(value):
    """Return str of value in 'multiples of pi' latex representation

    >>> x_pi(6.28) == r"2$\pi$"
    True
    >>> x_pi(3) == r"0.95$\pi$"
    True
    >>> x_pi(-np.pi) == r"-$\pi$"
    True
    >>> x_pi(0) == "0"
    True
    """
    n = value / np.pi
    if np.isclose(n, 0):
        return "0"
    elif np.isclose(abs(n), 1):
        return r"$\pi$" if n > 0 else r"-$\pi$"
    else:
        return r"{:.2g}$\pi$".format(n)


def decorator_decorator(decorator_wrapper):
    """A decorator decorator which allows it to be used with or without arguments

    Parameters
    ----------
    decorator_wrapper : Callable[[Any], Callable]

    Examples
    --------
    >>> @decorator_decorator
    ... def decorator_wrapper(optional="default"):
    ...     def actual_decorator(func):
    ...         return lambda x: func(x, optional)
    ...     return actual_decorator

    >>> @decorator_wrapper("hello")
    ... def foo(x, y):
    ...     print(x, y)
    >>> foo(1)
    1 hello

    >>> @decorator_wrapper
    ... def bar(x, y):
    ...     print(x, y)
    >>> bar(2)
    2 default
    """
    @wraps(decorator_wrapper)
    def new_wrapper(*args, **kwargs):
        try:
            callsig = get_call_signature(up=1)
        except IndexError:
            callsig = None

        if len(args) == 1 and not kwargs and (isinstance(args[0], type) or callable(args[0])):
            args[0].callsig = callsig
            return decorator_wrapper()(args[0])
        else:
            def deferred(cls_or_func):
                cls_or_func.callsig = callsig
                return decorator_wrapper(*args, **kwargs)(cls_or_func)
            return deferred

    return new_wrapper


@contextmanager
def cd(directory):
    """Change directory within this context

    Parameters
    ----------
    directory : str or path
    """
    previous_dir = os.getcwd()
    os.chdir(os.path.expanduser(str(directory)))
    try:
        yield
    finally:
        os.chdir(previous_dir)


def with_suffix(number, base=1000):
    """Convert number to string with SI suffix, e.g.: 14226 -> 14.2k, 5395984 -> 5.39M"""
    mapping = {base**3: "G", base**2: "M", base: "k"}
    for bucket, suffix in mapping.items():
        if number > 0.999 * bucket:
            return "{:.3g}{}".format(number / bucket, suffix)
    return "{:.3g}".format(number)


def rotate_axes(position, axes):
    """Rotate axes in position

    Examples
    --------
    >>> rotate_axes([1, 2, 3], "xy")
    (1, 2, 3)
    >>> rotate_axes([1, 2, 3], "yz")
    (2, 3, 1)
    """
    missing_axes = set('xyz') - set(axes)
    for a in missing_axes:
        axes += a
    assert len(axes) == 3

    mapping = dict(x=0, y=1, z=2)
    return tuple(position[mapping[a]] for a in axes)
