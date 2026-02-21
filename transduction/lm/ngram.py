"""
N-gram language models compatible with the StateLM interface.

Provides two variants:
- ByteNgramLM: byte-level (alphabet = 0..255), uses numpy arrays
- CharNgramLM: character/symbol-level (arbitrary alphabet), uses dicts

Usage:
    # Byte-level (alphabet = int 0-255)
    lm = ByteNgramLM.train(b"Hello world.", n=3)
    state = lm(b'H')              # advance by one byte
    print(state.logp_next[ord('e')])  # log P(e | H)

    # Character-level
    lm = CharNgramLM.train("abcabc", n=2)
    state = lm >> 'a'
    print(state.logp_next['b'])    # log P(b | a)
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from functools import cached_property
from typing import Any

from transduction.lm.base import LM, LMState, Token
from transduction.util import LogDistr, Str


# ===========================================================================
# Byte-level n-gram LM
# ===========================================================================

class NgramState(LMState[int]):
    """Immutable n-gram LM state, compatible with StateLM interface.

    Supports:
        state >> token     -> new state  (token is int 0-255)
        state.logp_next[x] -> log P(x | context)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token (int 0)
    """

    def __init__(self, lm: ByteNgramLM, context: Str[int], logp: float,
                 history: tuple[Any, ...] = ()) -> None:
        self.lm = lm
        self.eos = lm.eos
        self._context = context      # last (n-1) bytes as tuple of ints
        self.logp = logp
        self.history = history        # full path as nested tuple (like StateLM.context)

    def __rshift__(self, token: int) -> NgramState:
        if not isinstance(token, int):
            raise TypeError(f"Expected int (byte value 0-255), got {type(token).__name__}: {token!r}")
        if token == self.eos:
            raise ValueError(f"Cannot advance past EOS (byte {self.eos})")
        lp = self.logp_next[token]
        n = self.lm.n
        new_ctx = (self._context + (token,))[-(n - 1):] if n > 1 else ()
        return NgramState(self.lm, new_ctx, self.logp + lp,
                          history=(self.history, token))

    @cached_property
    def logp_next(self) -> LogDistr[int]:
        return self.lm._logp_next(self._context)

    @property
    def p_next(self) -> LogDistr[int]:
        return LogDistr({k: np.exp(v) for k, v in self.logp_next.items()})

    def path(self) -> list[int]:
        """Recover the full sequence of tokens from the history."""
        tokens: list[int] = []
        h = self.history
        while h:
            h, token = h
            tokens.append(token)
        tokens.reverse()
        return tokens

    def path_bytes(self) -> bytes:
        """Recover the full input as a bytes object."""
        return bytes(self.path())

    def __lt__(self, other: NgramState) -> bool:
        # Higher logp → smaller → explored first in min-heap
        return self.logp > other.logp

    def __repr__(self) -> str:
        ctx_bytes = bytes(self._context)
        return f'NgramState({ctx_bytes!r})'


class ByteNgramLM(LM[int]):
    """Byte-level n-gram language model with Laplace smoothing.

    Compatible with the StateLM/TokenizedLLM interface used by
    transduction.enumeration.
    """

    def __init__(self, counts: dict[Str[int], Counter[int]],
                 n: int, alpha: float = 1.0) -> None:
        """
        Args:
            counts: dict mapping (context_tuple,) -> Counter({byte_val: count})
            n: n-gram order (e.g., 3 for trigram)
            alpha: Laplace smoothing parameter
        """
        self.n = n
        self.alpha = alpha
        self.eos: int = 0  # use null byte as EOS

        # Precompute log-probability tables for each context
        self._tables: dict[Str[int], np.ndarray] = {}
        for ctx, counter in counts.items():
            total = sum(counter.values()) + 256 * alpha
            log_probs = np.full(256, np.log(alpha / total))
            for byte_val, count in counter.items():
                log_probs[byte_val] = np.log((count + alpha) / total)
            self._tables[ctx] = log_probs

        # Fallback: uniform distribution for unseen contexts
        self._uniform: np.ndarray = np.full(256, np.log(1.0 / 256))

    def _logp_next(self, context: Str[int]) -> LogDistr[int]:
        """Return LogDistr for a given context tuple."""
        # Try full context, then back off to shorter contexts
        for start in range(len(context) + 1):
            ctx = context[start:]
            if ctx in self._tables:
                lp = self._tables[ctx]
                return LogDistr({i: float(lp[i]) for i in range(256)})
        return LogDistr({i: float(self._uniform[i]) for i in range(256)})

    def initial(self) -> NgramState:
        return NgramState(self, (), 0.0)

    @classmethod
    def train(cls, data: bytes | bytearray | str | list[bytes | str],
              n: int = 3, alpha: float = 1.0) -> ByteNgramLM:
        """Train an n-gram model from byte data.

        Args:
            data: bytes, bytearray, str, or list of (bytes/str) instances.
                If data is a list, each element is treated as a separate
                training instance with EOS (\\x00) appended.  Contexts do
                not cross instance boundaries.
            n: n-gram order (1=unigram, 2=bigram, 3=trigram, ...)
            alpha: Laplace smoothing parameter

        Returns:
            ByteNgramLM instance
        """
        EOS_BYTE = 0  # b'\x00'

        # Convert to list of bytes instances
        if isinstance(data, (list, tuple)):
            instances: list[bytes] = []
            for d in data:
                if isinstance(d, str):
                    instances.append(d.encode('utf-8'))
                else:
                    instances.append(bytes(d))
        elif isinstance(data, str):
            instances = [data.encode('utf-8')]
        else:
            instances = [bytes(data)]

        counts: dict[Str[int], Counter[int]] = {}
        for inst in instances:
            seq = inst + bytes([EOS_BYTE])
            for order in range(1, n + 1):
                for i in range(len(seq) - order + 1):
                    ctx = tuple(seq[i:i + order - 1])
                    byte_val = seq[i + order - 1]
                    if ctx not in counts:
                        counts[ctx] = Counter()
                    counts[ctx][byte_val] += 1

        return cls(counts, n, alpha)

    @classmethod
    def from_file(cls, path: str, n: int = 3, alpha: float = 1.0) -> ByteNgramLM:
        """Train from a file."""
        with open(path, 'rb') as f:
            return cls.train(f.read(), n=n, alpha=alpha)

    def __repr__(self) -> str:
        return f'ByteNgramLM(n={self.n}, contexts={len(self._tables)})'


# ===========================================================================
# Character-level n-gram LM
# ===========================================================================

class CharNgramState(LMState[Token]):
    """Immutable char-level n-gram LM state, compatible with StateLM interface.

    Supports:
        state >> token     -> new state
        state.logp_next[x] -> log P(x | context)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token
    """

    def __init__(self, lm: CharNgramLM, context: Str[Token], logp: float,
                 history: tuple[Any, ...] = ()) -> None:
        self.lm = lm
        self.eos = lm.eos
        self._context = context      # last (n-1) symbols as tuple
        self.logp = logp
        self.history = history

    def __rshift__(self, token: Token) -> CharNgramState:
        if token not in self.logp_next or token == self.eos:
            raise ValueError(f"Out of vocabulary: {token!r}")
        lp = self.logp_next[token]
        n = self.lm.n
        new_ctx = (self._context + (token,))[-(n - 1):] if n > 1 else ()
        return CharNgramState(self.lm, new_ctx, self.logp + lp,
                              history=(self.history, token))

    @cached_property
    def logp_next(self) -> LogDistr[Token]:
        return self.lm._logp_next(self._context)

    def path(self) -> list[Token]:
        """Recover the full sequence of tokens from the history."""
        tokens: list[Token] = []
        h = self.history
        while h:
            h, token = h
            tokens.append(token)
        tokens.reverse()
        return tokens

    def __lt__(self, other: CharNgramState) -> bool:
        # Higher logp → smaller → explored first in min-heap
        return self.logp > other.logp

    def __repr__(self) -> str:
        return f'CharNgramState({self._context!r})'


class CharNgramLM(LM[Token]):
    """Character-level n-gram language model with Laplace smoothing.

    Works with arbitrary symbol alphabets (strings, characters, etc.).
    Compatible with the StateLM interface used by transduction.enumeration
    and transduction.lm.transduced.

    Usage:
        lm = CharNgramLM.train("abcabc", n=2)
        state = lm >> 'a'
        print(state.logp_next['b'])
    """

    def __init__(self, counts: dict[Str[Token], Counter[Token]],
                 n: int, alpha: float, alphabet: set[Token]) -> None:
        """
        Args:
            counts: dict mapping context_tuple -> Counter({symbol: count})
            n: n-gram order (e.g., 2 for bigram)
            alpha: Laplace smoothing parameter
            alphabet: set of symbols (including EOS)
        """
        self.n = n
        self.alpha = alpha
        self.eos: str = '<EOS>'
        self.alphabet = alphabet
        V = len(self.alphabet)

        self._tables: dict[Str[Token], dict[Token, float]] = {}
        for ctx, counter in counts.items():
            total = sum(counter.values()) + V * alpha
            self._tables[ctx] = {
                s: np.log((counter.get(s, 0) + alpha) / total)
                for s in self.alphabet
            }
        self._uniform: dict[Token, float] = {
            s: np.log(1.0 / V) for s in self.alphabet
        }

    def _logp_next(self, context: Str[Token]) -> LogDistr[Token]:
        """Return LogDistr for a given context tuple."""
        # Try full context, then back off to shorter contexts
        for start in range(len(context) + 1):
            ctx = context[start:]
            if ctx in self._tables:
                return LogDistr(self._tables[ctx])
        return LogDistr(self._uniform)

    def initial(self) -> CharNgramState:
        return CharNgramState(self, (), 0.0)

    @classmethod
    def train(cls, data: str | list[Any] | tuple[Any, ...],
              n: int = 2, alpha: float = 0.5,
              alphabet: set[Token] | None = None) -> CharNgramLM:
        """Train from a string, iterable of symbols, or list of instances.

        Args:
            data: string, iterable of symbols, or list of sequences.
                If data is a list of sequences (list of lists/tuples), each
                sequence is treated as a separate training instance with EOS
                appended.  Contexts do not cross instance boundaries.
            n: n-gram order (1=unigram, 2=bigram, ...)
            alpha: Laplace smoothing parameter
            alphabet: optional set of extra symbols to include in the vocabulary

        Returns:
            CharNgramLM instance
        """
        eos = '<EOS>'

        # Detect list-of-instances: data is a list/tuple whose first element
        # is itself a list/tuple (not a scalar symbol like str/int).
        if (isinstance(data, (list, tuple)) and len(data) > 0
                and isinstance(data[0], (list, tuple))):
            instances = data
        else:
            instances = [data]

        obs_alphabet: set[Token] = set()
        counts: dict[Str[Token], Counter[Token]] = {}
        for inst in instances:
            seq = list(inst) + [eos]
            obs_alphabet.update(inst)
            for order in range(1, n + 1):
                for i in range(len(seq) - order + 1):
                    ctx = tuple(seq[i:i + order - 1])
                    sym = seq[i + order - 1]
                    if ctx not in counts:
                        counts[ctx] = Counter()
                    counts[ctx][sym] += 1

        full_alphabet = obs_alphabet | {eos}
        if alphabet is not None:
            full_alphabet |= set(alphabet)
        return cls(counts, n, alpha, full_alphabet)

    def __repr__(self) -> str:
        return f'CharNgramLM(n={self.n}, alphabet_size={len(self.alphabet)})'
