"""Abstract base classes for autoregressive language models.

Defines :class:`LM` (model factory) and :class:`LMState` (decode position).
Together they follow the Iterable/Iterator pattern: an LM produces LMState
objects via ``initial()``, and each state supports ``state >> token`` to
advance and ``state.logp_next`` to query next-token log-probabilities.

Concrete implementations: :mod:`transduction.lm.ngram`,
:mod:`transduction.lm.huggingface_lm`, :mod:`transduction.lm.transduced`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from functools import cached_property
from typing import Generic, TypeVar

from transduction.util import LogDistr

Token = TypeVar('Token', bound=Hashable)


class HistoryPool:
    """Hash-consing pool for LM state histories.

    Assigns a unique integer ID to each ``(parent_id, token)`` pair,
    making ``context_key`` O(1) to compute and compare.
    """
    __slots__ = ('_next_id', '_table')

    def __init__(self) -> None:
        self._next_id = 1       # 0 is reserved for the empty history
        self._table: dict[tuple, int] = {}

    def intern(self, parent_id: int, token) -> int:
        """Return a unique ID for (parent_id, token), creating one if needed."""
        key = (parent_id, token)
        hid = self._table.get(key)
        if hid is None:
            hid = self._next_id
            self._next_id += 1
            self._table[key] = hid
        return hid


class LM(ABC, Generic[Token]):
    """Abstract base for language models; delegates to self.initial().

    The LM / LMState design mirrors Python's Iterable / Iterator pattern.
    An LM holds model parameters and acts as a factory for LMState objects
    (via ``initial()``), just as a list holds data and produces iterators
    (via ``__iter__()``).  LMStates are lightweight, immutable decode
    positions — like iterators, many can exist for a single LM.

    This mixin provides convenience methods so the LM can be used directly
    where an LMState would be expected::

        lm('abc')          # same as lm.initial()('abc')
        lm >> 'a'          # same as lm.initial() >> 'a'
        lm.logp_next['a']  # same as lm.initial().logp_next['a']

    Subclasses must implement ``initial()`` returning an ``LMState``.
    """

    eos: Token

    @cached_property
    def _history_pool(self) -> HistoryPool:
        return HistoryPool()

    @abstractmethod
    def initial(self) -> LMState:
        """Return the initial LMState (conditioned on the empty context)."""
        ...

    def __call__(self, xs: Iterable[Token]) -> LMState:
        """Shorthand for ``self.initial()(xs)``."""
        return self.initial()(xs)

    def __rshift__(self, token: Token) -> LMState:
        """Shorthand for ``self.initial() >> token``."""
        return self.initial() >> token

    @property
    def logp_next(self) -> LogDistr[Token]:
        """Shorthand for ``self.initial().logp_next``."""
        return self.initial().logp_next

    def greedy_decode(self, max_len: int = 100) -> list[Token]:
        """Shorthand for ``self.initial().greedy_decode(max_len)``."""
        return self.initial().greedy_decode(max_len)

    def prefetch(self, states: list[LMState]) -> None:
        """Hint: pre-compute logp_next for multiple states in a batch.

        Default: no-op. HuggingFaceLM overrides to batch sibling forward passes.
        Calling prefetch is always optional — states still compute lazily.
        """
        pass

    def sample(self) -> LMState:
        """Shorthand for ``self.initial().sample()``."""
        return self.initial().sample()

    def sample_decode(self, max_len: int = 100) -> list[Token]:
        """Shorthand for ``self.initial().sample_decode(max_len)``."""
        return self.initial().sample_decode(max_len)


class LMState(ABC, Generic[Token]):
    """An immutable position in an autoregressive decode — a conditional
    distribution P(next_token | tokens_so_far).

    Analogous to an Iterator: lightweight, many can coexist for one LM,
    and ``state >> token`` produces a new state (like ``next()`` advances
    an iterator).  The parent LM (analogous to Iterable) holds the shared
    model parameters; see :class:`LM`.

    Subclasses must provide:
        logp_next  — dict-like; includes EOS token with its log-probability
        eos        — EOS sentinel
        __rshift__ — advance state by one token, returns new state
        __call__   — advance state by a sequence of tokens, returns final state
    """

    eos: Token
    logprefix: float

    @property
    def context_key(self):
        """Hashable sufficient statistic for the LM state's distribution.

        Two states with the same context_key produce the same logp_next.
        Default: the hash-consed history ID (O(1) integer).  Subclasses may
        override with a coarser key (e.g., n-gram context tuple) when a suffix
        of the path is a sufficient statistic.

        All concrete LM states should set ``_history_id`` via their LM's
        ``_history_pool.intern()`` in ``__rshift__``.
        """
        return self._history_id

    @property
    @abstractmethod
    def logp_next(self) -> LogDistr[Token]:
        """Log-probability distribution over next tokens.

        Returns a dict-like object supporting ``logp_next[token]`` -> float
        and ``logp_next.items()`` -> iterable of (token, logp) pairs.
        Missing tokens should return ``-inf``.
        """
        ...

    @abstractmethod
    def __rshift__(self, token: Token) -> LMState:
        """Advance by one token, returning a new LMState conditioned on the extended context."""
        ...

    @cached_property
    def logprob(self):
        """Log probability that the LM generates exactly this string."""
        return self.logprefix + self.logp_next[self.eos]

    def greedy_decode(self, max_len: int = 100) -> list[Token]:
        """Greedy decode until EOS or max_len. Returns list of tokens."""
        state = self
        tokens: list[Token] = []
        for _ in range(max_len):
            if not state.logp_next:
                break
            best_tok = state.logp_next.argmax()
            if best_tok == state.eos:
                break
            tokens.append(best_tok)
            state = state >> best_tok
        return tokens

    def __call__(self, xs: Iterable[Token]) -> LMState:
        """Advance state by a sequence of tokens. Returns the final state."""
        s = self
        for x in xs:
            s = s >> x
        return s

    def sample(self) -> LMState:
        """Sample one token and advance. Returns new state (or self if EOS)."""
        tok = self.logp_next.sample()
        if tok == self.eos:
            return self
        return self >> tok

    def sample_decode(self, max_len: int = 100) -> list[Token]:
        """Sample autoregressively until EOS or max_len. Returns list of tokens."""
        state = self
        tokens: list[Token] = []
        for _ in range(max_len):
            tok = state.logp_next.sample()
            if tok == state.eos:
                break
            tokens.append(tok)
            state = state >> tok
        return tokens
