import os
from .. import _cpp

_cached_info = None


def cpu_info():
    """Forwarded from `cpuinfo.get_cpu_info()`"""
    global _cached_info
    if not _cached_info:
        try:
            from cpuinfo import get_cpu_info
        except ImportError:
            def get_cpu_info():
                return {}
        _cached_info = get_cpu_info()
    return _cached_info


def physical_core_count():
    """Try to return the number of physical cores

    An accurate number of physical cores will only be returned if the extension
    module was compiled with MKL. Otherwise, this will return the same number
    as `virtual_core_count()`.

    Examples
    --------
    >>> physical_core_count() <= virtual_core_count()
    True
    """
    try:
        # noinspection PyUnresolvedReferences
        return _cpp.physical_core_count
    except AttributeError:
        return os.cpu_count()


def virtual_core_count():
    """Return the number of threads the CPU can process simultaniously"""
    return os.cpu_count()


def summary():
    """Return a short description of the host CPU

    The returned SIMD instruction set is the one that the extension module was
    compiled with, not the highest one supported by the CPU.
    """
    info = cpu_info()
    if not info:
        return "py-cpuinfo is not installed"

    info = info.copy()
    hz_raw, scale = info['hz_advertised_raw']
    info['ghz'] = hz_raw * 10**(scale - 9)
    info['physical'] = physical_core_count()
    info['virtual'] = virtual_core_count()
    info['simd'] = _cpp.simd_info()
    return "{brand}\n{physical}/{virtual} cores @ {ghz:.2g} GHz with {simd}".format_map(info)


if __name__ == '__main__':
    print(summary())
