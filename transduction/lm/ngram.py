"""
N-gram language models compatible with the StateLM interface.

Provides two variants:
- ByteNgramLM: byte-level (alphabet = 0..255), uses numpy arrays
- CharNgramLM: character/symbol-level (arbitrary alphabet), uses dicts

Usage:
    # Byte-level
    lm = ByteNgramLM.train(b"Hello world.", n=3)
    state = lm.initial()
    state = state >> b'H'
    print(state.logp_next[b'e'])   # log P(e | H)

    # Character-level
    lm = CharNgramLM.train("abcabc", n=2)
    state = lm.initial()
    state = state >> 'a'
    print(state.logp_next['b'])    # log P(b | a)
"""

import numpy as np
from collections import Counter
from functools import cached_property

from transduction.lm.base import LMState, LogpNext


# ===========================================================================
# Byte-level n-gram LM
# ===========================================================================

def _to_byte(token):
    """Coerce token to a byte value (int 0-255)."""
    if isinstance(token, (bytes, bytearray)):
        return token[0] if len(token) == 1 else None
    if isinstance(token, int):
        return token
    return None


class NgramState(LMState):
    """Immutable n-gram LM state, compatible with StateLM interface.

    Supports:
        state >> token     -> new state
        state.logp_next[x] -> log P(x | context)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token
    """

    def __init__(self, lm, context, logp, history=()):
        self.lm = lm
        self.eos = lm.eos
        self._context = context      # last (n-1) bytes as tuple
        self.logp = logp
        self.history = history        # full path as nested tuple (like StateLM.context)

    def __rshift__(self, token):
        byte_val = _to_byte(token)
        if byte_val is None or bytes([byte_val]) == self.eos:
            raise ValueError(f"Out of vocabulary: {token!r}")
        lp = self.logp_next[token]
        n = self.lm.n
        new_ctx = (self._context + (byte_val,))[-(n - 1):] if n > 1 else ()
        return NgramState(self.lm, new_ctx, self.logp + lp,
                          history=(self.history, token))

    @cached_property
    def logp_next(self):
        return self.lm._logp_next(self._context)

    @property
    def p_next(self):
        return LogpNext({k: np.exp(v) for k, v in self.logp_next.items()})

    def path(self):
        """Recover the full sequence of tokens from the history."""
        tokens = []
        h = self.history
        while h:
            h, token = h
            tokens.append(token)
        tokens.reverse()
        return tokens

    def path_bytes(self):
        """Recover the full input as a bytes object."""
        return bytes(_to_byte(t) for t in self.path())

    def __lt__(self, other):
        # Higher logp → smaller → explored first in min-heap
        return self.logp > other.logp

    def __repr__(self):
        ctx_bytes = bytes(self._context)
        return f'NgramState({ctx_bytes!r})'


class ByteNgramLM:
    """Byte-level n-gram language model with Laplace smoothing.

    Compatible with the StateLM/TokenizedLLM interface used by
    transduction.enumeration.
    """

    def __init__(self, counts, n, alpha=1.0):
        """
        Args:
            counts: dict mapping (context_tuple,) -> Counter({byte_val: count})
            n: n-gram order (e.g., 3 for trigram)
            alpha: Laplace smoothing parameter
        """
        self.n = n
        self.alpha = alpha
        self.eos = b'\x00'  # use null byte as EOS

        # Precompute log-probability tables for each context
        self._tables = {}
        for ctx, counter in counts.items():
            total = sum(counter.values()) + 256 * alpha
            log_probs = np.full(256, np.log(alpha / total))
            for byte_val, count in counter.items():
                log_probs[byte_val] = np.log((count + alpha) / total)
            self._tables[ctx] = log_probs

        # Fallback: uniform distribution for unseen contexts
        self._uniform = np.full(256, np.log(1.0 / 256))

    def _logp_next(self, context):
        """Return LogpNext for a given context tuple."""
        # Try full context, then back off to shorter contexts
        for start in range(len(context) + 1):
            ctx = context[start:]
            if ctx in self._tables:
                lp = self._tables[ctx]
                return LogpNext({bytes([i]): float(lp[i]) for i in range(256)})
        return LogpNext({bytes([i]): float(self._uniform[i]) for i in range(256)})

    def initial(self):
        return NgramState(self, (), 0.0)

    @classmethod
    def train(cls, data, n=3, alpha=1.0):
        """Train an n-gram model from byte data.

        Args:
            data: bytes, bytearray, or iterable of bytes objects
            n: n-gram order (1=unigram, 2=bigram, 3=trigram, ...)
            alpha: Laplace smoothing parameter

        Returns:
            ByteNgramLM instance
        """
        if isinstance(data, (list, tuple)):
            data = b''.join(d if isinstance(d, bytes) else d.encode() for d in data)
        elif isinstance(data, str):
            data = data.encode('utf-8')

        counts = {}
        for order in range(1, n + 1):
            for i in range(len(data) - order + 1):
                ctx = tuple(data[i:i + order - 1])
                byte_val = data[i + order - 1]
                if ctx not in counts:
                    counts[ctx] = Counter()
                counts[ctx][byte_val] += 1

        return cls(counts, n, alpha)

    @classmethod
    def from_file(cls, path, n=3, alpha=1.0):
        """Train from a file."""
        with open(path, 'rb') as f:
            return cls.train(f.read(), n=n, alpha=alpha)

    def __repr__(self):
        return f'ByteNgramLM(n={self.n}, contexts={len(self._tables)})'


# ===========================================================================
# Character-level n-gram LM
# ===========================================================================

class CharNgramState(LMState):
    """Immutable char-level n-gram LM state, compatible with StateLM interface.

    Supports:
        state >> token     -> new state
        state.logp_next[x] -> log P(x | context)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token
    """

    def __init__(self, lm, context, logp, history=()):
        self.lm = lm
        self.eos = lm.eos
        self._context = context      # last (n-1) symbols as tuple
        self.logp = logp
        self.history = history

    def __rshift__(self, token):
        if token not in self.logp_next or token == self.eos:
            raise ValueError(f"Out of vocabulary: {token!r}")
        lp = self.logp_next[token]
        n = self.lm.n
        new_ctx = (self._context + (token,))[-(n - 1):] if n > 1 else ()
        return CharNgramState(self.lm, new_ctx, self.logp + lp,
                              history=(self.history, token))

    @cached_property
    def logp_next(self):
        return self.lm._logp_next(self._context)

    def path(self):
        """Recover the full sequence of tokens from the history."""
        tokens = []
        h = self.history
        while h:
            h, token = h
            tokens.append(token)
        tokens.reverse()
        return tokens

    def __lt__(self, other):
        # Higher logp → smaller → explored first in min-heap
        return self.logp > other.logp

    def __repr__(self):
        return f'CharNgramState({self._context!r})'


class CharNgramLM:
    """Character-level n-gram language model with Laplace smoothing.

    Works with arbitrary symbol alphabets (strings, characters, etc.).
    Compatible with the StateLM interface used by transduction.enumeration
    and transduction.lm.transduced.

    Usage:
        lm = CharNgramLM.train("abcabc", n=2)
        state = lm.initial()
        state = state >> 'a'
        print(state.logp_next['b'])
    """

    def __init__(self, counts, n, alpha, alphabet):
        """
        Args:
            counts: dict mapping context_tuple -> Counter({symbol: count})
            n: n-gram order (e.g., 2 for bigram)
            alpha: Laplace smoothing parameter
            alphabet: set of symbols (including EOS)
        """
        self.n = n
        self.alpha = alpha
        self.eos = '<EOS>'
        self.alphabet = sorted(alphabet)
        V = len(self.alphabet)

        self._tables = {}
        for ctx, counter in counts.items():
            total = sum(counter.values()) + V * alpha
            self._tables[ctx] = {
                s: np.log((counter.get(s, 0) + alpha) / total)
                for s in self.alphabet
            }
        self._uniform = {s: np.log(1.0 / V) for s in self.alphabet}

    def _logp_next(self, context):
        """Return LogpNext for a given context tuple."""
        # Try full context, then back off to shorter contexts
        for start in range(len(context) + 1):
            ctx = context[start:]
            if ctx in self._tables:
                return LogpNext(self._tables[ctx])
        return LogpNext(self._uniform)

    def initial(self):
        return CharNgramState(self, (), 0.0)

    @classmethod
    def train(cls, data, n=2, alpha=0.5):
        """Train from a string or iterable of symbols.

        Args:
            data: string or iterable of symbols
            n: n-gram order (1=unigram, 2=bigram, ...)
            alpha: Laplace smoothing parameter

        Returns:
            CharNgramLM instance
        """
        alphabet = set(data)
        counts = {}
        for order in range(1, n + 1):
            for i in range(len(data) - order + 1):
                ctx = tuple(data[i:i + order - 1])
                sym = data[i + order - 1]
                if ctx not in counts:
                    counts[ctx] = Counter()
                counts[ctx][sym] += 1
        return cls(counts, n, alpha, alphabet | {'<EOS>'})

    def __repr__(self):
        return f'CharNgramLM(n={self.n}, alphabet_size={len(self.alphabet)})'
