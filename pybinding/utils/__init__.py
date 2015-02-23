from . import cpuinfo, progressbar

_tic_times = []


def tic(msg=""):
    if msg != "":
        print(msg)
    global _tic_times
    import time
    _tic_times.append(time.time())


def toc(msg=""):
    if len(_tic_times) == 0:
        return

    if msg != "":
        msg += " "

    import time
    import datetime
    time_str = str(datetime.timedelta(seconds=time.time() - _tic_times.pop()))
    print(msg + time_str)


def cm2inch(value):
    """ Convert from centimeter to inch """
    return value / 2.54


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    import itertools
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def to_tuple(o):
    if isinstance(o, (tuple, list)):
        return tuple(o)
    else:
        return o,


def unpack_limits(args):
    args = to_tuple(args)
    if len(args) == 1:
        lim = abs(args[0])
        x = y = -lim, lim
    elif len(args) == 2:
        x = y = args
    elif len(args) == 4:
        x = args[:1]
        y = args[2:]
    else:
        raise Exception('Invalid number of arguments')

    return x, y
