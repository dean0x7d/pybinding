import time
import contextlib

_tic_times = []


def tic(message=""):
    if message:
        print(message)

    global _tic_times
    _tic_times.append(time.time())


def toc(message=""):
    if not _tic_times:
        return

    if message:
        print(message, end=" ")
    print(pretty_duration(time.time() - _tic_times.pop()))


@contextlib.contextmanager
def timed(end_msg="", start_msg=""):
    tic(start_msg)
    yield
    toc(end_msg)


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
