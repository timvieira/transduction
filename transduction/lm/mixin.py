import numpy as np
from abc import ABC, abstractmethod


class LMState(ABC):
    """Abstract base class for LM state objects.

    Subclasses must implement:
        logp_next  — dict-like with items() → (token, logp) pairs
                     and __getitem__(token) → logp
        eos        — EOS sentinel token
        __lshift__ — advance state by one token, returns new state
    """

    @property
    @abstractmethod
    def logp_next(self):
        ...

    @property
    @abstractmethod
    def eos(self):
        ...

    @abstractmethod
    def __lshift__(self, token):
        ...

    def greedy_decode(self, max_len=100):
        """Greedy decode until EOS or max_len. Returns list of tokens."""
        state = self
        tokens = []
        for _ in range(max_len):
            best_tok, best_lp = state.eos, state.logp_next[state.eos]
            for tok, lp in state.logp_next.items():
                if lp > best_lp:
                    best_tok, best_lp = tok, lp
            if best_tok == state.eos:
                break
            tokens.append(best_tok)
            state = state << best_tok
        return tokens

    def advance(self, xs):
        """Advance state by a sequence of tokens. Returns the final state."""
        s = self
        for x in xs:
            s = s << x
        return s

    def sample_next_token(self):
        """Sample a single next token from logp_next (including EOS)."""
        toks = []
        logps = []
        for tok, lp in self.logp_next.items():
            toks.append(tok)
            logps.append(lp)
        toks.append(self.eos)
        logps.append(self.logp_next[self.eos])
        logps = np.array(logps, dtype=np.float64)
        logps -= logps.max()
        probs = np.exp(logps)
        probs /= probs.sum()
        return toks[np.random.choice(len(toks), p=probs)]

    def sample(self):
        """Sample one token and advance. Returns new state (or self if EOS)."""
        tok = self.sample_next_token()
        if tok == self.eos:
            return self
        return self << tok

    def sample_decode(self, max_len=100, temperature=1.0):
        """Sample autoregressively until EOS or max_len. Returns list of tokens."""
        state = self
        tokens = []
        for _ in range(max_len):
            toks = []
            logps = []
            for tok, lp in state.logp_next.items():
                toks.append(tok)
                logps.append(lp)
            toks.append(state.eos)
            logps.append(state.logp_next[state.eos])
            logps = np.array(logps, dtype=np.float64)
            logps /= temperature
            logps -= logps.max()
            probs = np.exp(logps)
            probs /= probs.sum()
            idx = np.random.choice(len(toks), p=probs)
            if toks[idx] == state.eos:
                break
            tokens.append(toks[idx])
            state = state << toks[idx]
        return tokens


# Backward-compatible alias
LMStateMixin = LMState
