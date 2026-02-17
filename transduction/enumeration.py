"""
These are baseline, nonincremental methods for sampling/enumerating strings 
and prefixes in the precover, i.e., estimating prefix probabilities for a 
fixed target string.  They have been included in the repo primarily for 
pedagogical reasons.

Unlike TransducedLM and FusedTransducedLM, which compute the transduced
distribution incrementally (one output symbol at a time), these methods
take a fixed target string and estimate its prefix probability under the
inner LM by sampling/enumerating source strings in the precover.

Provides: prioritized_enumeration (best-first search), importance_sampling,
and crude_importance_sampling (without decomposition).

"""

from transduction import Precover, LazyPrecoverNFA

import heapq
import numpy as np
from dataclasses import dataclass

from transduction.util import colors, sample, logsumexp


@dataclass(frozen=False, eq=True, unsafe_hash=True)
class Item:
    "Items used in search queue"
    weight: float
    state: object
    source: object
    def __lt__(self, other):
        return self.weight > other.weight  # higher weight = higher priority (max-heap via heapq)
    def __repr__(self):
        return f'Item({self.weight:.3f}, {self.state}, {repr(self.source)})'


class prioritized_enumeration:
    def __init__(self, lm, fst, target, max_steps, decompose=None):
        """
        Args:
            decompose: A callable ``(fst, target) -> DecompositionResult`` returning
                an object with ``.quotient`` and ``.remainder`` FSAs.  Defaults
                to ``Precover``.  Examples: ``NonrecursiveDFADecomp``,
                ``RustDecomp``.
        """

        if decompose is None:
            decompose = Precover
        precover = decompose(fst, target)
        dfa = precover.quotient
        Q = precover.quotient.stop
        R = precover.remainder.stop
        dfa.stop |= R
        self.dfa = dfa.trim()

        oov = set(target) - fst.B
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        self.remainder_terms = []
        self.quotient_terms = []
        self.queue = []

        for q in self.dfa.start:
            heapq.heappush(self.queue, Item(weight = 0, state = q, source = lm))

        self.Q = Q
        self.R = R
        self.precover = precover
        self.lm = lm

        self.run(max_steps)

    def run(self, max_steps, verbosity=0):
        lm = self.lm
        EOS = lm.eos

        t = 0
        while self.queue:
            t += 1
            if t > max_steps:
                print(colors.light.red % 'stopped early')
                break
            item = heapq.heappop(self.queue)
            if verbosity > 0: print('pop:', item)
            lm_logp_next = item.source.logp_next
            if item.state in self.Q:
                self.quotient_terms.append(item)
                continue
            if item.state in self.R:
                # add the eos probability here
                self.remainder_terms.append(Item(
                    weight = item.weight + lm_logp_next[EOS],
                    state = item.state,
                    source = item.source,   # << EOS?
                ))
            for x, next_state in self.dfa.arcs(item.state):
                next_weight = float(item.weight + lm_logp_next[x])   # use LM state here
                if next_weight == -np.inf:
                    continue
                next_item = Item(
                    weight = next_weight,
                    state = next_state,
                    source = item.source >> x,
                )
                #print('push:', next_item)
                heapq.heappush(self.queue, next_item)


def _sample_step(lm_logp_next, arcs, is_eos_eligible, EOS):
    """One step of importance-sampling: build proposal, sample, return choice.

    Returns (chosen_symbol, transitions_dict, log_normalizer).
    If chosen_symbol is EOS, transitions_dict may not contain it.
    """
    q = {}
    T = {}
    if is_eos_eligible:
        q[EOS] = lm_logp_next[EOS]
    for x, next_state in arcs:
        q[x] = lm_logp_next[x]
        T[x] = next_state
    keys = list(q.keys())
    vals = np.array(list(q.values()))
    Z = logsumexp(vals)
    if np.isfinite(Z):
        vals = np.exp(vals - Z)
    else:
        vals = np.ones(len(vals))
    x_t = keys[sample(vals)]
    return x_t, T, Z


class importance_sampling:

    def __init__(self, lm, fst, target, decompose=None):
        """
        Args:
            decompose: A callable ``(fst, target) -> DecompositionResult`` returning
                an object with ``.quotient`` and ``.remainder`` FSAs.  Defaults
                to ``Precover``.  Examples: ``NonrecursiveDFADecomp``,
                ``RustDecomp``.
        """

        if decompose is None:
            decompose = Precover
        precover = decompose(fst, target)
        dfa = precover.quotient
        Q = precover.quotient.stop
        R = precover.remainder.stop
        dfa.stop |= R
        self.dfa = dfa.trim()

        oov = set(target) - fst.B
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        self.Q = Q
        self.R = R
        self.precover = precover
        self.lm = lm

    def sample(self, max_length=np.inf):
        EOS = self.lm.eos

        t = 0
        for i in self.dfa.lazy().start():
            item = Item(weight = 0, state = i, source = self.lm)
        while True:
            t += 1
            if t > max_length:
                print(colors.light.red % 'stopped early')
                break

            lm_logp_next = item.source.logp_next
            if item.state in self.Q:
                return item

            x_t, T, Z = _sample_step(lm_logp_next, self.dfa.arcs(item.state),
                                      item.state in self.R, EOS)

            if x_t == EOS:
                return item

            item = Item(
                weight = item.weight + Z,
                state = T[x_t],
                source = item.source >> x_t,
            )


class crude_importance_sampling:
    """
    This class is similar to importance_sampling except that it does not
    leverage from decomposition
    """

    def __init__(self, lm, fst, target):

        oov = set(target) - fst.B
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        self.dfa = LazyPrecoverNFA(fst, target).det()
        self.lm = lm

    def sample(self, max_length=np.inf):
        EOS = self.lm.eos

        t = 0
        for i in self.dfa.start():
            item = Item(weight = 0, state = i, source = self.lm)

        while True:
            t += 1
            if t > max_length:
                print(colors.light.red % 'stopped early')
                break

            lm_logp_next = item.source.logp_next

            x_t, T, Z = _sample_step(lm_logp_next, self.dfa.arcs(item.state),
                                      self.dfa.is_final(item.state), EOS)

            if x_t == EOS:
                return item

            item = Item(
                weight = item.weight + Z,
                state = T[x_t],
                source = item.source >> x_t,
            )
