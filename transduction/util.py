"""Shared utilities: Integerizer, log-space types, caching, memory/time limits.

Key public classes:

- :class:`Integerizer` — bijection between objects and contiguous ints.
- :class:`LogVector` — mutable sparse log-space accumulator (``logaddexp``).
- :class:`LogDistr` — immutable normalized log-probability distribution.
- :class:`memoize` — simple function-result cache (decorator).

Key public functions:

- :func:`logsumexp` — numerically stable log-sum-exp.
- :func:`set_memory_limit` / :func:`memory_limit` — RLIMIT_AS guards.
- :func:`timelimit` — wall-clock timeout via SIGALRM.
"""

from __future__ import annotations

import html as _html
import resource
import signal
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from functools import partial
from types import FrameType
from typing import Any, Generic, TypeVar, overload

import numpy as np
import numpy.typing as npt


#_______________________________________________________________________________
# Integerizer (formerly from arsenal)

K = TypeVar('K', bound=Hashable)


class Integerizer(Generic[K]):
    """Maintain a perfect hash (bijection) between keys and contiguous ints."""

    def __init__(self, data: list[K] | None = None) -> None:
        """Create an Integerizer, optionally adding initial keys from *data*."""
        self._map: dict[K, int] = {}
        self._list: list[K] = []
        self._frozen = False
        if data: self.add(data)

    def _add(self, k: K) -> int:
        """Return the integer for *k*, assigning a new one if unseen."""
        try:
            return self._map[k]
        except KeyError as exc:
            if self._frozen:
                raise ValueError(f'Alphabet is frozen. Key "{k}" not found.') from exc
            x = self._map[k] = len(self._list)
            self._list.append(k)
            return x

    def __contains__(self, k: object) -> bool:
        """Return True if *k* has been assigned an integer."""
        return k in self._map

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys in insertion order."""
        return iter(self._list)

    def __len__(self) -> int:
        """Return the number of registered keys."""
        return len(self._list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Integerizer):
            return NotImplemented
        return self._list == other._list  # pyright: ignore[reportUnknownMemberType]

    def __repr__(self) -> str:
        return f'Integerizer(size={len(self)}, frozen={self._frozen})'

    @overload
    def __getitem__(self, i: list[int]) -> list[K]: ...
    @overload
    def __getitem__(self, i: int) -> K: ...
    def __getitem__(self, i: int | list[int]) -> K | list[K]:
        """Decode: integer(s) -> key(s). Accepts an int or list of ints."""
        if isinstance(i, list):
            return [self._list[ii] for ii in i]
        return self._list[i]

    @overload
    def __call__(self, k: list[K]) -> list[int]: ...
    @overload
    def __call__(self, k: K) -> int: ...
    def __call__(self, k: K | list[K]) -> int | list[int]:
        """Encode: key(s) -> integer(s). Accepts a key or list of keys."""
        if isinstance(k, list):
            return [self._add(kk) for kk in k]
        return self._add(k)

    encode = __call__
    decode = __getitem__
    lookup = __getitem__
    add = __call__

    def freeze(self) -> Integerizer[K]:
        """Freeze the mapping; future unseen keys raise ValueError."""
        self._frozen = True
        return self

    def keys(self) -> list[K]:
        """Return the list of keys in insertion order."""
        return self._list

    def items(self) -> Any:
        """Return (key, int) pairs."""
        return self._map.items()

    def _repr_html_(self) -> str:
        """Rich HTML table for Jupyter notebooks."""
        MAX = 20
        n = len(self)
        frozen = ' frozen' if self._frozen else ''
        header = (f'<b>Integerizer</b> '
                  f'<span style="color:#888">(n={n}{frozen})</span>')
        if n == 0:
            return header + ' <i>empty</i>'
        rows = []
        for idx, key in enumerate(self._list[:MAX]):
            rows.append(
                f'<tr><td style="text-align:right;color:#888;'
                f'padding:1px 6px">{idx}</td>'
                f'<td style="padding:1px 6px">'
                f'{_html.escape(repr(key))}</td></tr>'
            )
        if n > MAX:
            rows.append(
                f'<tr><td colspan="2" style="text-align:center;'
                f'color:#888;padding:1px 6px">'
                f'… {n - MAX} more</td></tr>'
            )
        return (
            header
            + '<table style="border-collapse:collapse;margin-top:4px;'
              'font:12px monospace">'
            + '<thead><tr>'
              '<th style="text-align:right;padding:1px 6px;'
              'border-bottom:1px solid #ccc">idx</th>'
              '<th style="text-align:left;padding:1px 6px;'
              'border-bottom:1px solid #ccc">key</th>'
              '</tr></thead><tbody>'
            + ''.join(rows)
            + '</tbody></table>'
        )


#_______________________________________________________________________________
# colors (formerly from arsenal)

class _AnsiStyle:
    """ANSI escape code wrapper supporting ``style % string`` formatting."""
    def __init__(self, code: str) -> None:
        self._code = code
    def __mod__(self, text: object) -> str:
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
    def mark(ok: bool) -> str:
        return '\033[0;32m\u2714\033[0m' if ok else '\033[2;31m\u2718\033[0m'


#_______________________________________________________________________________
# memoize (formerly from arsenal.cache)

class memoize:
    """Cache a function's return value to avoid recalculation."""
    def __init__(self, func: Any) -> None:
        self.func = func
        self.cache: dict[Any, Any] = {}
        try:
            self.__name__: str = func.__name__
            self.__doc__ = func.__doc__
        except AttributeError:
            pass

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None: return self.func
        return partial(self, obj)

    def __call__(self, *args: Any) -> Any:
        try:
            return self.cache[args]
        except KeyError:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self) -> str:
        return f'<memoize({self.func!r})>'


#_______________________________________________________________________________
# timelimit (formerly from arsenal)

class Timeout(Exception):
    pass

@contextmanager
def timelimit(seconds: float | None, sig: signal.Signals = signal.SIGALRM) -> Iterator[None]:
    """Context manager that raises Timeout after `seconds`.

    Saves and restores the previous signal handler. Clears the timer
    on both normal exit and exception paths.
    """
    if seconds is None:
        yield
        return
    def signal_handler(signum: int, frame: FrameType | None) -> None:
        raise Timeout(f'Call took longer than {seconds} seconds.')
    prev = signal.signal(sig, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(sig, prev)


@contextmanager
def timeit(name: str, fmt: str = '{name} ({htime})', header: str | None = None) -> Iterator[None]:
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

def set_memory_limit(gb: float | None) -> None:
    """Set process virtual address space limit in gigabytes (RLIMIT_AS).

    No-op if gb is None or non-positive.
    """
    if gb is not None and gb > 0:
        limit = int(gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


@contextmanager
def memory_limit(gb: float | None) -> Iterator[None]:
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
# logsumexp

def logsumexp(arr: npt.ArrayLike) -> float:
    """Numerically stable log-sum-exp over an array of log values."""
    a = np.array(arr, dtype=np.float64)
    a = a[a > -np.inf]
    if len(a) == 0:
        return float('-inf')
    vmax = a.max()
    a -= vmax
    np.exp(a, out=a)
    out = np.log(a.sum())
    out += vmax
    return float(out)


#_______________________________________________________________________________
# Sparse log-space mappings: LogVector (mutable) and LogDistr (immutable)

_NEG_INF = float('-inf')

V = TypeVar('V', bound=Hashable)


class _SparseLogMap(dict[V, float]):
    """Dict subclass for sparse mappings in log-space (keys -> log-values).

    Missing keys return ``-inf`` (via ``__missing__``).
    """

    def __missing__(self, key: V) -> float:
        return _NEG_INF

    def materialize(self, top: int | None = None) -> dict[V, float]:
        """Return a dict of ``{key: value}`` sorted by descending value.

        If ``top`` is given, return only the top-k entries.
        """
        items = sorted(self.items(), key=lambda kv: kv[1], reverse=True)
        if top is not None:
            items = items[:int(top)]
        return dict(items)

    def top(self, K: int) -> dict[V, float]:
        """Return a dict of the top-K entries by value."""
        return self.materialize(top=K)

    def argmax(self) -> V:
        """Return the key with the highest value."""
        return max(self, key=self.__getitem__)

    def __repr__(self) -> str:
        name = type(self).__name__
        n = len(self)
        if n <= 5:
            inner = ', '.join(f'{k!r}: {v:.4f}' for k, v in self.items())
            return f'{name}({{{inner}}})'
        return f'{name}(n={n})'

    def _repr_html_(self) -> str:
        """Rich HTML table for Jupyter notebooks."""
        MAX = 20
        name = type(self).__name__
        n = len(self)
        header = (f'<b>{_html.escape(name)}</b> '
                  f'<span style="color:#888">(n={n})</span>')
        if n == 0:
            return header + ' <i>empty</i>'
        items = sorted(self.items(), key=lambda kv: kv[1], reverse=True)
        rows = []
        for key, logv in items[:MAX]:
            p = float(np.exp(logv))
            rows.append(
                f'<tr>'
                f'<td style="padding:1px 6px">'
                f'{_html.escape(repr(key))}</td>'
                f'<td style="text-align:right;padding:1px 6px">'
                f'{logv:.4f}</td>'
                f'<td style="text-align:right;padding:1px 6px">'
                f'{p:.4f}</td>'
                f'</tr>'
            )
        if n > MAX:
            rows.append(
                f'<tr><td colspan="3" style="text-align:center;'
                f'color:#888;padding:1px 6px">'
                f'… {n - MAX} more</td></tr>'
            )
        return (
            header
            + '<table style="border-collapse:collapse;margin-top:4px;'
              'font:12px monospace">'
            + '<thead><tr>'
              '<th style="text-align:left;padding:1px 6px;'
              'border-bottom:1px solid #ccc">key</th>'
              '<th style="text-align:right;padding:1px 6px;'
              'border-bottom:1px solid #ccc">logp</th>'
              '<th style="text-align:right;padding:1px 6px;'
              'border-bottom:1px solid #ccc">p</th>'
              '</tr></thead><tbody>'
            + ''.join(rows)
            + '</tbody></table>'
        )


class LogVector(_SparseLogMap[V]):
    """Mutable accumulator for sparse log-nonneg-real vectors.

    Replaces the ``defaultdict(lambda: -np.inf)`` + ``logaddexp`` pattern.
    """

    def logaddexp(self, key: V, value: float) -> None:
        """Accumulate: ``self[key] = logaddexp(self[key], value)``."""
        prev = self.get(key)
        if prev is None:
            self[key] = value
        else:
            self[key] = float(np.logaddexp(prev, value))

    def normalize(self) -> LogDistr[V]:
        """Return a ``LogDistr`` by subtracting ``logsumexp`` from each entry."""
        Z = logsumexp(list(self.values()))
        return LogDistr({k: float(v - Z) for k, v in self.items()})


class LogDistr(_SparseLogMap[V]):
    """Immutable normalized distribution in log-space.

    Supports ``sample()`` but not mutation.
    """

    def sample(self) -> V:
        """Sample a key proportional to ``exp(value)``."""
        toks = list(self.keys())
        logps = np.array(list(self.values()), dtype=np.float64)
        logps -= logps.max()
        probs = np.exp(logps)
        probs /= probs.sum()
        return toks[np.random.choice(len(toks), p=probs)]


#_______________________________________________________________________________
# sample (formerly from arsenal.maths)

def sample(w: npt.ArrayLike, size: int | None = None, u: npt.ArrayLike | None = None) -> Any:
    """Draw samples from an (unnormalized) discrete distribution via inverse CDF."""
    c = np.cumsum(w)
    if u is None:
        u = np.random.uniform(0, 1, size=size)
    return c.searchsorted(u * c[-1])  # pyright: ignore[reportUnknownVariableType]
