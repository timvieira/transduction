import resource
import signal
from functools import partial
from contextlib import contextmanager


#_______________________________________________________________________________
# Integerizer (formerly from arsenal)

class Integerizer:
    """Maintain a perfect hash (bijection) between keys and contiguous ints."""

    def __init__(self, data=None):
        self._map = {}
        self._list = []
        self._frozen = False
        if data: self.add(data)

    def _add(self, k):
        try:
            return self._map[k]
        except KeyError as exc:
            if self._frozen:
                raise ValueError(f'Alphabet is frozen. Key "{k}" not found.') from exc
            x = self._map[k] = len(self._list)
            self._list.append(k)
            return x

    def __contains__(self, k):
        return k in self._map

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)

    def __eq__(self, other):
        return self._list == other._list

    def __repr__(self):
        return f'Integerizer(size={len(self)}, frozen={self._frozen})'

    def __getitem__(self, i):
        if isinstance(i, list):
            return [self._list[ii] for ii in i]
        return self._list[i]

    def __call__(self, k):
        if isinstance(k, list):
            return [self._add(kk) for kk in k]
        return self._add(k)

    encode = __call__
    decode = __getitem__
    lookup = __getitem__
    add = __call__

    def freeze(self):
        self._frozen = True
        return self

    def keys(self):
        return self._list

    def items(self):
        return self._map.items()


#_______________________________________________________________________________
# colors (formerly from arsenal)

class _AnsiStyle:
    """ANSI escape code wrapper supporting ``style % string`` formatting."""
    def __init__(self, code):
        self._code = code
    def __mod__(self, text):
        return f'\033[{self._code}m{text}\033[0m'

class _LightColors:
    red = _AnsiStyle('1;31')
    green = _AnsiStyle('1;32')

class _DarkColors:
    white = _AnsiStyle('2;37')

class colors:
    light = _LightColors()
    dark = _DarkColors()
    @staticmethod
    def mark(ok):
        return '\033[0;32m\u2714\033[0m' if ok else '\033[2;31m\u2718\033[0m'


#_______________________________________________________________________________
# memoize (formerly from arsenal.cache)

class memoize:
    """Cache a function's return value to avoid recalculation."""
    def __init__(self, func):
        self.func = func
        self.cache = {}
        try:
            self.__name__ = func.__name__
            self.__doc__ = func.__doc__
        except AttributeError:
            pass

    def __get__(self, obj, objtype=None):
        if obj is None: return self.func
        return partial(self, obj)

    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        return f'<memoize({self.func!r})>'


#_______________________________________________________________________________
# timelimit (formerly from arsenal)

class Timeout(Exception):
    pass

@contextmanager
def timelimit(seconds, sig=signal.SIGALRM):
    """Context manager that raises Timeout after `seconds`.

    Saves and restores the previous signal handler. Clears the timer
    on both normal exit and exception paths.
    """
    if seconds is None:
        yield
        return
    def signal_handler(signum, frame):
        raise Timeout(f'Call took longer than {seconds} seconds.')
    prev = signal.signal(sig, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(sig, prev)


@contextmanager
def timeit(name, fmt='{name} ({htime})', header=None):
    """Context manager that prints elapsed wall-clock time."""
    import sys
    from time import time
    if header is not None:
        print(header)
    b4 = time()
    yield
    sec = time() - b4
    ht = f'{sec:.4f} sec' if sec < 60 else f'{sec/60:.1f} min'
    print(fmt.format(name=name, htime=ht, sec=sec), file=sys.stderr)


#_______________________________________________________________________________
# memory limits

def set_memory_limit(gb):
    """Set process virtual address space limit in gigabytes (RLIMIT_AS).

    No-op if gb is None or non-positive.
    """
    if gb is not None and gb > 0:
        limit = int(gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


@contextmanager
def memory_limit(gb):
    """Context manager that sets a memory limit and restores the original on exit.

    Saves the original RLIMIT_AS soft/hard limits, sets a new soft limit
    (preserving the hard limit), and restores on exit.
    """
    if gb is None or gb <= 0:
        yield
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = int(gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


#_______________________________________________________________________________
# sample (formerly from arsenal.maths)

def sample(w, size=None, u=None):
    """Draw samples from an (unnormalized) discrete distribution via inverse CDF."""
    import numpy as np
    c = np.cumsum(w)
    if u is None:
        u = np.random.uniform(0, 1, size=size)
    return c.searchsorted(u * c[-1])


