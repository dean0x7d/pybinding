import pickle
import gzip
from functools import wraps
from collections import namedtuple

__all__ = ['pickleable']


def _save(self, file_name):
    with gzip.open(file_name, 'wb') as file:
        pickle.dump(self, file, protocol=4)


def _from_file(_, file_name):
    with gzip.open(file_name, 'rb') as file:
        return pickle.load(file)


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


def _decorator(dec):
    """A decorator decorator which enables use with or without arguments"""
    @wraps(dec)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], type):
            return dec()(args[0])
        else:
            return lambda cls: dec(*args, **kwargs)(cls)

    return new_dec


@_decorator
def pickleable(props='', impl='', version: int=0):
    props = props.split()

    tokens = {
        '.': lambda x: x.impl,
        '[]': lambda x: [v.impl for v in x]
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
            impl_state = (convert(v) for convert, v in zip(conversions, data['impl']))
            self.impl = mock_impl(*impl_state)

    def decorator(cls):
        if not hasattr(cls, '__getstate__'):
            cls.__getstate_manages_dict__ = True  # enables boost_python pickling

        return _override_methods(cls, __getstate__=getstate, __setstate__=setstate,
                                 save=_save, from_file=classmethod(_from_file))

    return decorator
