from transduction.base import DecompositionResult
from transduction.precover import Precover
from transduction.lazy import Lazy
from transduction.fsa import FSA
from transduction.fst import EPSILON
from transduction.precover_nfa import PrecoverNFA, PeekabooFixedNFA as PeekabooPrecover

from collections import deque
from functools import cached_property


class PeekabooStrings:

    def __init__(self, fst, max_steps=float('inf'), **kwargs):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.max_steps = max_steps

    def __call__(self, target):
        """Return Q/R for a single target string.

        For the empty target, delegates to the reference Precover
        implementation — decompose_next computes results for target+y
        (all next symbols) so it cannot produce the result for the
        empty target itself.

        For non-empty targets, calls decompose_next on the prefix and
        looks up the last symbol.
        """
        target = tuple(target)
        if not target:
            return Precover(self.fst, target)
        return self.decompose_next(target[:-1]).get(
            target[-1], DecompositionResult(set(), set())
        )

    def decompose_next(self, target):
        target = tuple(target)
        precover = {y: DecompositionResult(set(), set()) for y in self.target_alphabet}
        worklist = deque()

        dfa = PeekabooPrecover(self.fst, target).det()
        for state in dfa.start():
            worklist.append(('', state))

        N = len(target)
        t = 0
        while worklist:
            (xs, state) = worklist.popleft()
            t += 1
            if t > self.max_steps:
                break

            relevant_symbols = {ys[N] for _, ys in state if len(ys) > N}

            # Shortcut: At most one of the `relevant_symbols` can be
            # continuous. If we find one, we can stop expanding.
            #
            # NOTE: This assumes the FST is functional.  A productive
            # input-epsilon cycle (eps-input arcs that produce non-epsilon
            # output) makes an FST non-functional, since the cycle can be
            # traversed any number of times yielding distinct outputs for
            # the same input.  Non-functional FSTs may violate the
            # uniqueness invariant below.
            #
            # Proof (functional FSTs): Suppose state S is universal for
            # both y and z (y != z).  FilteredDFA(target+y) recognises
            # precover(target+y).  Universality from S means
            # Reach(S)·Σ* ⊆ precover(target+y), and likewise for z.
            # For a functional FST each input has a unique output, so
            # precover(target+y) ∩ precover(target+z) = ∅.  Therefore
            # Reach(S)·Σ* ⊆ ∅, but Reach(S) is non-empty (S is on the
            # worklist), giving a contradiction.
            continuous = set()
            for y in relevant_symbols:
                dfa_filtered = FilteredDFA(dfa=dfa, fst=self.fst, target=target + (y,))
                if dfa_filtered.accepts_universal(state, self.source_alphabet):
                    precover[y].quotient.add(xs)
                    continuous.add(y)
                elif dfa_filtered.is_final(state):
                    precover[y].remainder.add(xs)
            if len(continuous) > 1:
                raise ValueError(
                    f"State is universal for multiple symbols {continuous} — "
                    f"FST is likely non-functional (see FST.is_functional())"
                )
            if continuous:
                continue    # we have found a quotient and can skip

            for x, next_state in dfa.arcs(state):
                worklist.append((xs + x, next_state))

        return precover


class Peekaboo(DecompositionResult):

    def __init__(self, fst, target=(), *, _parent=None, _symbol=None):
        self.fst = fst
        target = tuple(target)
        self.target = target
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")
        self._parent = _parent
        self._symbol = _symbol

    @cached_property
    def _results(self):
        """Run the BFS for self.target, returning {y: (quotient_FSA, remainder_FSA)}."""
        precover = {y: DecompositionResult(set(), set()) for y in self.target_alphabet}
        worklist = deque()
        states = set()

        dfa = PeekabooPrecover(self.fst, self.target).det()
        for state in dfa.start():
            worklist.append(state)
            states.add(state)

        arcs = []

        N = len(self.target)
        while worklist:
            state = worklist.popleft()

            relevant_symbols = {ys[N] for _, ys in state if len(ys) > N}

            continuous = set()
            for y in relevant_symbols:
                dfa_filtered = FilteredDFA(dfa=dfa, fst=self.fst, target=self.target + (y,))
                if dfa_filtered.accepts_universal(state, self.source_alphabet):
                    precover[y].quotient.add(state)
                    continuous.add(y)
                elif dfa_filtered.is_final(state):
                    precover[y].remainder.add(state)
            if len(continuous) > 1:
                raise ValueError(
                    f"State is universal for multiple symbols {continuous} — "
                    f"FST is likely non-functional (see FST.is_functional())"
                )
            if continuous:
                continue

            for x, next_state in dfa.arcs(state):
                if next_state not in states:
                    worklist.append(next_state)
                    states.add(next_state)

                arcs.append((state, x, next_state))

        result = {}
        for y in self.target_alphabet:
            d = precover[y]
            q = FSA(start=set(dfa.start()), arcs=arcs, stop=d.quotient)
            r = FSA(start=set(dfa.start()), arcs=arcs, stop=d.remainder)
            result[y] = (q.trim(), r.trim())

        return result

    def __call__(self, target):
        """Backward-compat: Peekaboo(fst)(target) -> {y: DecompositionResult}."""
        p = Peekaboo(self.fst, target)
        return {y: DecompositionResult(*qr) for y, qr in p._results.items()}

    def decompose_next(self):
        return {y: Peekaboo(self.fst, self.target + (y,), _parent=self, _symbol=y)
                for y in self.target_alphabet}

    @cached_property
    def _qr(self):
        parent = self._parent
        assert parent is not None, "Root Peekaboo has no quotient/remainder"
        return parent._results[self._symbol]

    @property
    def quotient(self):
        return self._qr[0]

    @property
    def remainder(self):
        return self._qr[1]


class FilteredDFA(Lazy):
    """Augments a determinized PeekabooPrecover semi-automaton with ``is_final``.

    Used by ``PeekabooNonrecursive`` (the non-incremental variant).  DFA states
    are frozensets of ``(fst_state, buffer)`` pairs from the underlying NFA.
    A state is final iff any element has a buffer that starts with the target
    and the FST state is accepting.

    Arcs are passed through from the underlying DFA unchanged (no refinement
    or truncation).  Compare with ``TruncatedDFA`` in ``peekaboo_incremental``,
    which additionally clips and filters NFA elements in each DFA state to
    normalize powerset representations across incremental ``>>`` steps.
    """

    def __init__(self, *, dfa, fst, target):
        self.dfa = dfa
        self.fst = fst
        self.target = target

    def start(self):
        return self.dfa.start()

    def arcs(self, state):
        return self.dfa.arcs(state)

    def arcs_x(self, state, x):
        return self.dfa.arcs_x(state, x)

    def is_final(self, state):
        N = len(self.target)
        return any(ys[:N] == self.target and self.fst.is_final(i) for (i, ys) in state)
