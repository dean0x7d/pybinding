from .. import _cpp

_cached_info = None


def get_cpu_info():
    global _cached_info
    if not _cached_info:
        import cpuinfo
        _cached_info = cpuinfo.get_cpu_info()
    return _cached_info


def physical_core_count():
    try:
        # noinspection PyUnresolvedReferences
        return _cpp.get_max_threads()
    except AttributeError:
        return get_cpu_info()['count']


def virtual_core_count():
    return get_cpu_info()['count']


def summary():
    info = get_cpu_info().copy()
    hz_raw, scale = info['hz_advertised_raw']
    info['ghz'] = hz_raw * 10**(scale - 9)
    info['physical'] = physical_core_count()
    info['virtual'] = virtual_core_count()
    info['simd'] = _cpp.simd_info()
    return "{brand}\n{physical}/{virtual} cores @ {ghz:.2g} GHz with {simd}".format_map(info)


if __name__ == '__main__':
    print(summary())
