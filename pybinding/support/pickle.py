import gzip
import os
import pathlib
import pickle
from collections import namedtuple
from itertools import zip_longest

from ..utils import decorator_decorator

__all__ = ['pickleable', 'save', 'load']


def _normalize(file):
    """Convenience function to support path objects."""
    if 'Path' in type(file).__name__:
        return str(file)
    else:
        return file


def _add_extension(file):
    if not isinstance(file, str):
        return file

    path = pathlib.Path(file)
    if not path.suffix:
        return str(path.with_suffix('.pbz'))
    else:
        return file


def save(obj, file, add_pbz_extension=True):
    """Pickle an object and save it to a compressed file

    Essentially, this is just a wrapper for :func:`pickle.dump()` with a few conveniences,
    like default pickle protocol 4 and gzip compression.

    Parameters
    ----------
    obj : Any
        Object to be saved.
    file : Union[str, pathlib.Path]
        May be a str, a pathlib object or a file object created with open().
    add_pbz_extension : bool
        The '.pbz' extension is added if file has none.
    """
    file = _normalize(file)
    if add_pbz_extension:
        file = _add_extension(file)

    with gzip.open(file, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load(file):
    """Load a pickled object from a compressed file

    Wraps :func:`pickle.load()` with the same conveniences as :func:`save()`.

    Parameters
    ----------
    file : Union[str, pathlib.Path]
        May be a str, a pathlib object or a file object created with open().
    """
    file = _normalize(file)
    file_ext = _add_extension(file)
    if isinstance(file, str) and not os.path.exists(file) and os.path.exists(file_ext):
        file = file_ext

    with gzip.open(file, 'rb') as f:
        return pickle.load(f)


def _check_version(self, data, version):
    if data.get('version', version) != version:
        msg = "Can't create class {} v{} from incompatible data v{}".format(
            self.__class__.__name__, version, data['version']
        )
        raise RuntimeError(msg)


def _override_methods(cls, **kwargs):
    for name, method in kwargs.items():
        setattr(cls, name, method)
    return cls


def _find_boost_python_attr(obj, attr, default=None):
    if any("Boost.Python.instance" in str(c) for c in obj.__class__.mro()):
        return getattr(super(obj.__class__, obj), attr, default)
    else:
        return default


@decorator_decorator
def pickleable(props='', impl='', version: int=0):
    props = props.split()

    tokens = {
        '.': lambda x: x.impl if x else None,
        '[]': lambda x: [v.impl for v in x] if x else []
    }

    impl_names, conversions = [], []
    for name in impl.split():
        for token, call in tokens.items():
            if token in name:
                impl_names.append(name.strip(token))
                conversions.append(call)
                break
        else:
            impl_names.append(name)
            conversions.append(lambda x: x)

    mock_impl = namedtuple('T', impl_names)

    def getstate(self):
        state = dict(version=version, dict=self.__dict__.copy())

        bp_getstate = _find_boost_python_attr(self, '__getstate__')
        if bp_getstate:
            state['boost_python'] = bp_getstate()

        if props:
            state['props'] = [getattr(self, n) for n in props]

        if impl_names:
            state['dict'].pop('impl')
            state['impl'] = [getattr(self, n) for n in impl_names]

        return state

    def setstate(self, state):
        _check_version(self, state, version)
        self.__dict__.update(state['dict'])

        bp_setstate = _find_boost_python_attr(self, '__setstate__')
        if bp_setstate:
            bp_setstate(state['boost_python'])

        if props:
            for prop, value in zip(props, state['props']):
                setattr(self, prop, value)

        if impl_names:
            impl_state = (convert(v) for convert, v in zip_longest(conversions, state['impl']))
            self.impl = mock_impl(*impl_state)

    def decorator(cls):
        if not hasattr(cls, '__getstate__'):
            cls.__getstate_manages_dict__ = True  # enables boost_python pickling

        return _override_methods(cls, __getstate__=getstate, __setstate__=setstate)

    return decorator
