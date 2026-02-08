import numpy as np
from abc import ABC, abstractmethod


class LogpNext:
    """Dict-backed log-probability lookup with top-k support.

    Convention: items() includes ALL tokens in the distribution,
    including EOS. __getitem__ returns -inf for unknown tokens.
    """

    def __init__(self, scores):
        self._scores = scores

    def __getitem__(self, token):
        v = self._scores.get(token)
        return v if v is not None else -np.inf

    def __contains__(self, token):
        return token in self._scores

    def keys(self):
        return self._scores.keys()

    def items(self):
        return self._scores.items()

    def materialize(self, top=None):
        items = sorted(self._scores.items(), key=lambda kv: kv[1], reverse=True)
        if top is not None:
            items = items[:int(top)]
        return dict(items)

    def top(self, K):
        return self.materialize(top=K)

    def argmax(self):
        """Return the token with the highest log-probability."""
        return max(self._scores, key=self._scores.__getitem__)

    def sample(self):
        """Sample a token from the distribution."""
        toks = list(self._scores.keys())
        logps = np.array(list(self._scores.values()), dtype=np.float64)
        logps -= logps.max()
        probs = np.exp(logps)
        probs /= probs.sum()
        return toks[np.random.choice(len(toks), p=probs)]


class LMState(ABC):
    """Abstract base class for LM state objects.

    Subclasses must provide:
        logp_next  — dict-like with items() → (token, logp) pairs
                     and __getitem__(token) → logp
        eos        — EOS sentinel token (attribute or property)
        __lshift__ — advance state by one token, returns new state
        __call__   — advance state by a sequence of tokens, returns final state
    """

    @property
    @abstractmethod
    def logp_next(self):
        ...

    @abstractmethod
    def __lshift__(self, token):
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
            state = state << best_tok
        return tokens

    def __call__(self, xs):
        """Advance state by a sequence of tokens. Returns the final state."""
        s = self
        for x in xs:
            s = s << x
        return s

    def sample(self):
        """Sample one token and advance. Returns new state (or self if EOS)."""
        tok = self.logp_next.sample()
        if tok == self.eos:
            return self
        return self << tok

    def sample_decode(self, max_len=100):
        """Sample autoregressively until EOS or max_len. Returns list of tokens."""
        state = self
        tokens = []
        for _ in range(max_len):
            tok = state.logp_next.sample()
            if tok == state.eos:
                break
            tokens.append(tok)
            state = state << tok
        return tokens
