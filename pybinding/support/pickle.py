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
    """Pickle an object and save it to a compressed file.

    This functions wraps pickle.dump() with a few conveniences:
     - file may be a str, a pathlib object or a file object created with open()
     - pickle protocol 4 is used and the data is compressed with gzip
     - the '.pbz' extension is added if file has none
    """
    file = _normalize(file)
    if add_pbz_extension:
        file = _add_extension(file)

    with gzip.open(file, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load(file):
    """Load a pickled object from a compressed file.

    This functions wraps pickle.load() with the same conveniences as pb.save().
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
        if not hasattr(cls, name):
            setattr(cls, name, method)
    return cls


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
        __dict__ = self.__dict__.copy()
        if impl:
            __dict__.pop('impl')

        return dict(version=version, dict=__dict__,
                    props=[getattr(self, n) for n in props],
                    impl=[getattr(self, n) for n in impl_names])

    def setstate(self, data):
        _check_version(self, data, version)

        self.__dict__.update(data['dict'])

        for prop, value in zip(props, data['props']):
            setattr(self, prop, value)

        if impl_names:
            impl_state = (convert(v) for convert, v in zip_longest(conversions, data['impl']))
            self.impl = mock_impl(*impl_state)

    def decorator(cls):
        if not hasattr(cls, '__getstate__'):
            cls.__getstate_manages_dict__ = True  # enables boost_python pickling

        return _override_methods(cls, __getstate__=getstate, __setstate__=setstate)

    return decorator
