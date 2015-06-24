import pickle
import gzip
from collections import namedtuple

__all__ = ['pickleable', 'pickleable_impl']


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


def pickleable(version_or_cls=0):
    version, cls = (0, version_or_cls) if isinstance(version_or_cls, type) else (version_or_cls, 0)

    def getstate(self):
        return dict(version=version, state=self.__dict__.copy())

    def setstate(self, data):
        _check_version(self, data, version)
        self.__dict__.update(data['state'])

    def decorator(_cls):
        return _override_methods(_cls, __getstate__=getstate, __setstate__=setstate,
                                 save=_save, from_file=classmethod(_from_file))

    if cls:
        return decorator(cls)
    else:
        return decorator


def pickleable_impl(field_names: str, version: int=0):
    tokens = {
        '.': lambda x: x.impl,
        '[]': lambda x: [v.impl for v in x]
    }

    names, conversions = [], []
    for name in field_names.split():
        for token, call in tokens.items():
            if token in name:
                names.append(name.strip(token))
                conversions.append(call)
                break
        else:
            names.append(name)
            conversions.append(lambda x: x)

    mock_impl = namedtuple('T', names)

    def getstate(self):
        return dict(version=version, state=list(getattr(self, n) for n in names))

    def setstate(self, data):
        _check_version(self, data, version)
        state = (convert(v) for convert, v in zip(conversions, data['state']))
        self.impl = mock_impl(*state)

    return lambda cls: _override_methods(cls, __getstate__=getstate, __setstate__=setstate,
                                         save=_save, from_file=classmethod(_from_file))
