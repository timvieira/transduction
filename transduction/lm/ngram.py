"""
Byte-level n-gram language model compatible with the StateLM interface.

Usage:
    lm = ByteNgramLM.train(b"Hello world. This is training data.", n=3)
    state = lm.initial()
    state = state << b'H'
    state = state << b'e'
    print(state.logp_next[b'l'])   # log P(l | He)
    print(state.logp)              # cumulative log prob
"""

import numpy as np
from collections import Counter


class NgramLogpNext:
    """Lazy log-probability lookup for next byte, matching LazyProb interface."""

    def __init__(self, log_probs):
        # log_probs: np.array of shape (256,) indexed by byte value
        self._log_probs = log_probs

    def _to_byte(self, token):
        """Coerce token to a byte value (int 0-255)."""
        if isinstance(token, (bytes, bytearray)):
            return token[0] if len(token) == 1 else None
        if isinstance(token, int):
            return token
        if isinstance(token, str):
            # FST arc symbols are strings like '116' for byte 116
            try:
                return int(token)
            except ValueError:
                return None
        return None

    def __getitem__(self, token):
        b = self._to_byte(token)
        if b is not None:
            return float(self._log_probs[b])
        return 0.0

    def keys(self):
        return [bytes([i]) for i in range(256)]

    def items(self):
        return [(bytes([i]), float(self._log_probs[i])) for i in range(256)]

    def materialize(self, top=None):
        idx = self._log_probs.argsort()
        if top is not None:
            idx = idx[-int(top):]
        return {bytes([i]): float(self._log_probs[i]) for i in reversed(idx)}


class NgramState:
    """Immutable n-gram LM state, compatible with StateLM interface.

    Supports:
        state << token     -> new state
        state.logp_next[x] -> log P(x | context)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token
    """

    def __init__(self, lm, context, logp, history=()):
        self.lm = lm
        self._context = context      # last (n-1) bytes as tuple
        self.logp = logp
        self.history = history        # full path as nested tuple (like StateLM.context)

    @property
    def eos(self):
        return self.lm.eos

    def __lshift__(self, token):
        byte_val = self.logp_next._to_byte(token)
        if byte_val is None:
            raise ValueError(f"Cannot convert {token!r} to byte")
        lp = float(self.logp_next._log_probs[byte_val])
        n = self.lm.n
        new_ctx = (self._context + (byte_val,))[-(n - 1):] if n > 1 else ()
        return NgramState(self.lm, new_ctx, self.logp + lp,
                          history=(self.history, token))

    @property
    def logp_next(self):
        return self.lm._logp_next(self._context)

    @property
    def p_next(self):
        lp = self.logp_next
        return NgramLogpNext(np.exp(lp._log_probs))

    def advance(self, xs):
        s = self
        for x in xs:
            s = s << x
        return s

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
        return bytes(self.logp_next._to_byte(t) for t in self.path())

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
        """Return NgramLogpNext for a given context tuple."""
        # Try full context, then back off to shorter contexts
        for start in range(len(context) + 1):
            ctx = context[start:]
            if ctx in self._tables:
                return NgramLogpNext(self._tables[ctx])
        return NgramLogpNext(self._uniform)

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
