"""
Reference transduced language model using Precover decomposition.

Computes exact next-token probabilities by enumerating the Q/R languages
from the Precover decomposition at each step.  Terminates whenever the
inner LM assigns zero probability to all sufficiently long source strings
(e.g., any finite-support LM), even when Q and R are infinite.

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

from __future__ import annotations

import heapq
from functools import cached_property
from typing import Any

from transduction.fsa import EPSILON
from transduction.fst import FST
from transduction.lm.base import LM, LMState, Token
from transduction.util import LogDistr, Str, log1mexp, logsumexp
from transduction.precover import Precover


class ReferenceTransducedLM(LM[Token]):
    """Ground-truth transduced LM using Precover decomposition.

    Computes exact next-token probabilities by enumerating Q/R languages
    via LM-pruned heap search over the Precover DFA.  Terminates for any
    LM with finite support, even when Q/R are infinite.
    """

    def __init__(self, inner_lm: LM, fst: FST[Any, Any],
                 eos: Token = '<EOS>') -> None:  # type: ignore[assignment]
        self.inner_lm = inner_lm
        self.fst = fst
        self.eos = eos
        self._decomp = Precover.factory(fst)
        self._target_alphabet = fst.B - {EPSILON}
        self._source_alphabet = fst.A - {EPSILON}

    def initial(self) -> ReferenceTransducedState:
        return ReferenceTransducedState(self, (), 0.0)


class ReferenceTransducedState(LMState[Token]):
    """Immutable state for the ReferenceTransducedLM.

    Supports:
        state >> y         -> new state (advance by target symbol y)
        state.logp_next[y] -> log P(y | target_so_far)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token
    """

    def __init__(self, tlm: ReferenceTransducedLM, target: Str[Token],
                 logp: float) -> None:
        self.tlm = tlm
        self.eos = tlm.eos
        self._target = target
        self.logp = logp

    def _score(self, prefix: Str[Token]) -> float:
        """Compute log P(output starts with prefix).

        Heap-ordered enumeration over the Precover DFA (Q and R share arcs
        and start states; only stop states differ).  Processes source strings
        in order of decreasing LM prefix probability.

        - Q-accepting (universal) states: add prefix probability, prune subtree.
        - R-accepting (non-universal final) states: add string probability.
        - Prune when LM prefix probability is -inf.
        """
        result = self.tlm._decomp(prefix)
        Q, R = result.quotient, result.remainder
        # Q and R share the same DFA arcs and start states.
        inner_lm = self.tlm.inner_lm
        inner_eos = inner_lm.eos
        parts: list[float] = []

        # Max-heap by LM prefix probability: (-logp, tie_breaker, dfa_state, lm_state)
        heap: list[tuple[float, int, Any, Any]] = []
        counter = 0
        for s in Q.start:
            lm0 = inner_lm.initial()
            heapq.heappush(heap, (-lm0.logp, counter, s, lm0))
            counter += 1

        while heap:
            (_, _, dfa_s, lm_s) = heapq.heappop(heap)

            if dfa_s in Q.stop:
                parts.append(lm_s.logp)
                continue  # prefix probability covers all extensions

            if dfa_s in R.stop:
                parts.append(lm_s.logp + lm_s.logp_next[inner_eos])

            for x, dfa_t in Q.arcs(dfa_s):
                next_lm = lm_s >> x
                if next_lm.logp == float('-inf'):
                    continue
                heapq.heappush(heap, (-next_lm.logp, counter, dfa_t, next_lm))
                counter += 1

        return logsumexp(parts)

    @cached_property
    def logp_next(self) -> LogDistr[Token]:
        Z = self._score(self._target)
        scores: dict[Token, float] = {}
        for y in self.tlm._target_alphabet:
            s = self._score(self._target + (y,))
            if s > float('-inf'):
                scores[y] = s - Z
        # EOS as residual: log P(EOS) = log(1 - Σ_y P(y))
        total = logsumexp(list(scores.values()))
        scores[self.eos] = log1mexp(total)
        return LogDistr(scores)

    def __rshift__(self, y: Token) -> ReferenceTransducedState:
        if y == self.eos:
            raise ValueError("Cannot advance past EOS")
        lp = self.logp_next[y]
        if lp == float('-inf'):
            raise ValueError(f"Symbol {y!r} has zero probability")
        return ReferenceTransducedState(self.tlm, self._target + (y,), self.logp + lp)

    def path(self) -> list[Token]:
        return list(self._target)

    def _repr_html_(self) -> str:
        from transduction.viz import render_logp_next_html
        return render_logp_next_html(
            'ReferenceTransducedState', self._target, self.logp, self.logp_next,
        )

    def __repr__(self) -> str:
        return f'ReferenceTransducedState(target={self._target!r})'
