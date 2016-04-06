from cpuinfo import cpuinfo
from .. import _cpp

_info = cpuinfo.get_cpu_info()
try:
    physical_thread_count = _cpp.get_max_threads()
except AttributeError:
    physical_thread_count = _info['count']


def name():
    return _info['brand']


def physical_core_count():
    return physical_thread_count


def virtual_core_count():
    return _info['count']


def threads():
    return "Threads {}/{} @ {:.3} GHz with {}".format(
        physical_core_count(), virtual_core_count(), _info['hz_advertised'], _cpp.simd_info()
    )
