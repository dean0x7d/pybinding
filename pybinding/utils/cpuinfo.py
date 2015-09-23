from cpuinfo import cpuinfo
_info = cpuinfo.get_cpu_info()
try:
    from .. import _cpp
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
    return "Threads {}/{} @ {:.3} GHz".format(
        physical_core_count(), virtual_core_count(), _info['hz_advertised']
    )
