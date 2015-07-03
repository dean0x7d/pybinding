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
        x = args[:2]
        y = args[2:]
    else:
        raise Exception('Invalid number of arguments')

    return x, y


def with_defaults(options: dict, defaults_dict: dict=None, **defaults_kwargs):
    """ Return a dict where missing keys are filled in by defaults. """
    options = options if options else {}
    if defaults_dict:
        options = dict(defaults_dict, **options)
    options = dict(defaults_kwargs, **options)
    return options


def pretty_duration(seconds: float):
    """Return a pretty duration string

    >>> pretty_duration(2.1e-6)
    '0.00ms'
    >>> pretty_duration(2.1e-5)
    '0.02ms'
    >>> pretty_duration(2.1e-4)
    '0.21ms'
    >>> pretty_duration(2.1e-3)
    '2.1ms'
    >>> pretty_duration(2.1e-2)
    '21ms'
    >>> pretty_duration(2.1e-1)
    '0.21s'
    >>> pretty_duration(2.1)
    '2.10s'
    >>> pretty_duration(12.1)
    '12.1s'
    >>> pretty_duration(22.1)
    '22s'
    >>> pretty_duration(62.1)
    '1:02'
    >>> pretty_duration(621.1)
    '10:21'
    >>> pretty_duration(6217.1)
    '1:43:37'
    """
    miliseconds = seconds * 1000
    if miliseconds < 1:
        return "{:.2f}ms".format(miliseconds)
    elif miliseconds < 10:
        return "{:.1f}ms".format(miliseconds)
    elif miliseconds < 100:
        return "{:.0f}ms".format(miliseconds)
    elif seconds < 10:
        return "{:.2f}s".format(seconds)
    elif seconds < 20:
        return "{:.1f}s".format(seconds)
    elif seconds < 60:
        return "{:.0f}s".format(seconds)
    else:
        minutes = seconds // 60
        seconds = int(seconds - minutes * 60)

        if minutes < 60:
            return "{minutes:.0f}:{seconds:02}".format(**locals())
        else:
            hours = minutes // 60
            minutes = int(minutes - hours * 60)
            return "{hours:.0f}:{minutes:02}:{seconds:02}".format(**locals())
