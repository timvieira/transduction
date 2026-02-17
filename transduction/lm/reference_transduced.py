"""
Reference transduced language model using Precover decomposition.

Computes exact next-token probabilities by enumerating the Q/R languages
from the Precover decomposition at each step.  This is a ground-truth
implementation for validating approximate algorithms like TransducedLM.

Only terminates when Q and R are finite (i.e., finite-relation FSTs).

Usage:
    from transduction.lm.ngram import CharNgramLM
    from transduction.lm.reference_transduced import ReferenceTransducedLM

    inner = CharNgramLM.train("hello world", n=2)
    fst = ...
    tlm = ReferenceTransducedLM(inner, fst)

    state = tlm.initial()
    print(state.logp_next['a'])
    state = state >> 'a'
"""

import numpy as np
from functools import cached_property

from transduction.fsa import EPSILON
from transduction.lm.base import LM, LMState, LogpNext
from transduction.precover import Precover


class ReferenceTransducedLM(LM):
    """Ground-truth transduced LM using Precover decomposition.

    Computes exact next-token probabilities by enumerating Q/R languages.
    Only terminates when Q and R are finite.
    """

    def __init__(self, inner_lm, fst, eos='<EOS>'):
        self.inner_lm = inner_lm
        self.fst = fst
        self.eos = eos
        self._decomp = Precover.factory(fst)
        self._target_alphabet = sorted(fst.B - {EPSILON})

    def initial(self):
        return ReferenceTransducedState(self, (), 0.0)


class ReferenceTransducedState(LMState):
    """Immutable state for the ReferenceTransducedLM.

    Supports:
        state >> y         -> new state (advance by target symbol y)
        state.logp_next[y] -> log P(y | target_so_far)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token
    """

    def __init__(self, tlm, target, logp):
        self.tlm = tlm
        self.eos = tlm.eos
        self._target = target
        self.logp = logp

    # Note: this is independent of the state
    def _score(self, prefix):
        """Compute log P(output starts with prefix).

        Uses the Precover decomposition:
        - Q strings contribute prefix probability (marginalized over continuations).
        - R strings contribute exact string probability (with EOS).
        """
        # TODO: these strings could be structured into a trie to reduce the number 
        # of inner LM state updates
        result = self.tlm._decomp(prefix)
        inner_eos = self.tlm.inner_lm.eos
        parts = []
        for src in result.quotient.language():
            state = self.tlm.inner_lm(src)
            parts.append(state.logp)
        for src in result.remainder.language():
            state = self.tlm.inner_lm(src)
            parts.append(state.logp + state.logp_next[inner_eos])
        return float(np.logaddexp.reduce(parts)) if parts else -np.inf

    @cached_property
    def logp_next(self):
        Z = self._score(self._target)
        scores = {}
        for y in self.tlm._target_alphabet:
            s = self._score(self._target + (y,))
            if s > -np.inf:
                scores[y] = s - Z
        # EOS = residual: P(EOS) = 1 - sum_y P(y)
        if scores:
            p_non_eos = sum(np.exp(lp) for lp in scores.values())
            scores[self.eos] = float(np.log1p(-p_non_eos)) if p_non_eos < 1 else -np.inf
        else:
            scores[self.eos] = 0.0
        return LogpNext(scores)

    def __rshift__(self, y):
        if y == self.eos:
            raise ValueError("Cannot advance past EOS")
        lp = self.logp_next[y]
        if lp == -np.inf:
            raise ValueError(f"Symbol {y!r} has zero probability")
        return ReferenceTransducedState(self.tlm, self._target + (y,), self.logp + lp)

    def path(self):
        return list(self._target)

    def _repr_html_(self):
        from transduction.viz import render_logp_next_html
        return render_logp_next_html(
            'ReferenceTransducedState', self._target, self.logp, self.logp_next,
        )

    def __repr__(self):
        return f'ReferenceTransducedState(target={self._target!r})'
