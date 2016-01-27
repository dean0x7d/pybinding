import time

__all__ = ['tic', 'toc', 'timed', 'pretty_duration']

_tic_times = []


def tic():
    """Set a start time"""
    global _tic_times
    _tic_times.append(time.time())


def toc(message=""):
    """Print the elapsed time from the last :func:`.tic`

    Parameters
    ----------
    message : str
        Print this in front of the elapsed time.
    """
    if not _tic_times:
        raise RuntimeError("Called toc() without a tic()")

    if message:
        print(message, end=" ")
    print(pretty_duration(time.time() - _tic_times.pop()))


class _Timed:
    def __init__(self, message=""):
        self.message = message

    def __enter__(self):
        self._enter_time = time.time()
        return self

    def __exit__(self, *_):
        self.elapsed = time.time() - self._enter_time
        if self.message:
            print(self.message, self)

    def __str__(self):
        return pretty_duration(self.elapsed)


def timed(message=""):
    """Context manager which times its code block

    Parameters
    ----------
    message : str
        Message to print on block exit, followed by the elapsed time.
    """
    return _Timed(message)


def pretty_duration(seconds):
    """Return a pretty duration string

    Parameters
    ----------
    seconds : float
        Duration in seconds

    Examples
    --------
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
