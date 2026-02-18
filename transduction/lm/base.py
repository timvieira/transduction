from abc import ABC, abstractmethod
from transduction.util import LogDistr


class LM(ABC):
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

    @abstractmethod
    def initial(self):
        """Return the initial LMState (conditioned on the empty context)."""
        ...

    def __call__(self, xs):
        """Shorthand for ``self.initial()(xs)``."""
        return self.initial()(xs)

    def __rshift__(self, token):
        """Shorthand for ``self.initial() >> token``."""
        return self.initial() >> token

    @property
    def logp_next(self):
        """Shorthand for ``self.initial().logp_next``."""
        return self.initial().logp_next

    def greedy_decode(self, max_len=100):
        """Shorthand for ``self.initial().greedy_decode(max_len)``."""
        return self.initial().greedy_decode(max_len)

    def sample(self):
        """Shorthand for ``self.initial().sample()``."""
        return self.initial().sample()

    def sample_decode(self, max_len=100):
        """Shorthand for ``self.initial().sample_decode(max_len)``."""
        return self.initial().sample_decode(max_len)


class LMState(ABC):
    """An immutable position in an autoregressive decode — a conditional
    distribution P(next_token | tokens_so_far).

    Analogous to an Iterator: lightweight, many can coexist for one LM,
    and ``state >> token`` produces a new state (like ``next()`` advances
    an iterator).  The parent LM (analogous to Iterable) holds the shared
    model parameters; see :class:`LM`.

    Subclasses must provide:
        logp_next  — dict-like with items() → (token, logp) pairs
                     and __getitem__(token) → logp
        eos        — EOS sentinel token (attribute or property)
        __rshift__ — advance state by one token, returns new state
        __call__   — advance state by a sequence of tokens, returns final state
    """

    @property
    @abstractmethod
    def logp_next(self):
        ...

    @abstractmethod
    def __rshift__(self, token):
        ...

    def greedy_decode(self, max_len=100):
        """Greedy decode until EOS or max_len. Returns list of tokens."""
        state = self
        tokens = []
        for _ in range(max_len):
            best_tok = state.logp_next.argmax()
            if best_tok == state.eos:
                break
            tokens.append(best_tok)
            state = state >> best_tok
        return tokens

    def __call__(self, xs):
        """Advance state by a sequence of tokens. Returns the final state."""
        s = self
        for x in xs:
            s = s >> x
        return s

    def sample(self):
        """Sample one token and advance. Returns new state (or self if EOS)."""
        tok = self.logp_next.sample()
        if tok == self.eos:
            return self
        return self >> tok

    def sample_decode(self, max_len=100):
        """Sample autoregressively until EOS or max_len. Returns list of tokens."""
        state = self
        tokens = []
        for _ in range(max_len):
            tok = state.logp_next.sample()
            if tok == state.eos:
                break
            tokens.append(tok)
            state = state >> tok
        return tokens
