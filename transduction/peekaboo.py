from transduction.base import PrecoverDecomp
from transduction.lazy import Lazy
from transduction.fsa import FSA
from transduction.fst import EPSILON
from transduction.eager_nonrecursive import Precover

from arsenal import colors
from collections import deque


# [2025-11-21 Fri] This version of the next-symbol prediction algorithm
# enumerates strings (and states).  We also have a version that enumrates just
# the states.
class PeekabooStrings:

    def __init__(self, fst, max_steps=float('inf')):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.max_steps = max_steps

    def __call__(self, target):
        precover = {y: PrecoverDecomp(set(), set()) for y in self.target_alphabet}
        worklist = deque()

        dfa = PeekabooPrecover(self.fst, target).det()
        for state in dfa.start():
            worklist.append(('', state))

        t = 0
        N = len(target)
        while worklist:
            (xs, state) = worklist.popleft()
            t += 1
            if t > self.max_steps:
                print(colors.light.red % 'stopped early')
                break

            relevant_symbols = {ys[N] for _, ys in state if len(ys) > N}

            # Shortcut: At most one of the `relevant_symbols` can be
            # continuous. If we find one, we can stop expanding.
            continuous = set()
            for y in relevant_symbols:
                dfa_filtered = FilteredDFA(dfa=dfa, fst=self.fst, target=target + y)
                #assert dfa_filtered.materialize().equal(Precover(self.fst, target + y).min)
                #print('ok:', repr(target + y))
                if dfa_filtered.accepts_universal(state, self.source_alphabet):
                    precover[y].quotient.add(xs)
                    continuous.add(y)
                elif dfa_filtered.is_final(state):
                    precover[y].remainder.add(xs)
            assert len(continuous) <= 1
            if continuous:
                continue    # we have found a quotient and can skip

            for x, next_state in dfa.arcs(state):
                worklist.append((xs + x, next_state))

        return precover


class Peekaboo:

    def __init__(self, fst, max_steps=float('inf')):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.max_steps = max_steps

    def __call__(self, target):
        precover = {y: PrecoverDecomp(set(), set()) for y in self.target_alphabet}
        worklist = deque()
        states = set()

        dfa = PeekabooPrecover(self.fst, target).det()
        for state in dfa.start():
            worklist.append(state)
            states.add(state)

        arcs = []

        t = 0
        N = len(target)
        while worklist:
            state = worklist.popleft()
            t += 1
            if t > self.max_steps:
                print(colors.light.red % 'stopped early')
                break

            relevant_symbols = {ys[N] for _, ys in state if len(ys) > N}

            # Shortcut: At most one of the `relevant_symbols` can be
            # continuous. If we find one, we can stop expanding.
            continuous = set()
            for y in relevant_symbols:
                dfa_filtered = FilteredDFA(dfa=dfa, fst=self.fst, target=target + y)
                #assert dfa_filtered.materialize().equal(Precover(self.fst, target + y).min)
                #print('ok:', repr(target + y))
                if dfa_filtered.accepts_universal(state, self.source_alphabet):
                    precover[y].quotient.add(state)
                    continuous.add(y)
                elif dfa_filtered.is_final(state):
                    precover[y].remainder.add(state)
            assert len(continuous) <= 1
            if continuous:
                continue    # we have found a quotient and can skip

            for x, next_state in dfa.arcs(state):
                if next_state not in states:
                    worklist.append(next_state)
                    states.add(next_state)

                arcs.append((state, x, next_state))

        foo = {y: PrecoverDecomp(set(), set()) for y in self.target_alphabet}
        for y in precover:
            Q, R = precover[y]
            q = FSA(start=set(dfa.start()), arcs=arcs, stop=Q)
            r = FSA(start=set(dfa.start()), arcs=arcs, stop=R)
            foo[y] = PrecoverDecomp(q.trim(), r.trim())

        return foo


# TODO: in order to predict EOS, we need to extract the preimage from Q and R
class PeekabooPrecover(Lazy):
    """NOTE: this is a semi-automaton as it does not have an `is_final` method.

    It implements a state space that tracks the states of an FST `fst` along
    with the target string they generate.  It prunes the state space to just the
    states that are relevant to `target` followed by at least additional target
    symbol.

    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = target
        self.N = len(target)

    def arcs(self, state):
        (i, ys) = state
        n = len(ys)
        m = min(n, self.N)
        if self.target[:m] != ys[:m]: return

        # case: grow the buffer until we have covered all of the target string
        if n < self.N:
            for x, y, j in self.fst.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                elif y == self.target[n]:
                    yield (x, (j, self.target[:n+1]))

        # extend the buffer beyond the target string by one symbol
        elif n == self.N:
            for x, y, j in self.fst.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                else:
                    yield (x, (j, ys + y))

        # truncate the buffer after the (N+1)th symbol
        elif n == self.N + 1:
            for x, _, j in self.fst.arcs(i):
                yield (x, (j, ys))

    def start(self):
        for i in self.fst.I:
            yield (i, '')

    def is_final(self, state):
        (i, ys) = state
        return self.fst.is_final(i) and ys.startswith(self.target) and len(ys) == self.N+1


class FilteredDFA(Lazy):
    """NOTE: This class augments a determinized `PeekabooPrecover` semi-automaton by
    adding an appropriate `is_final` method so that it is a valid finite-state
    automaton that encodes `Precover(fst, target)`.
    """

    def __init__(self, *, dfa, fst, target):
        self.dfa = dfa
        self.fst = fst
        self.target = target

    def start(self):
        return self.dfa.start()

    def arcs(self, state):
        return self.dfa.arcs(state)

    def is_final(self, state):
        return any(ys.startswith(self.target) and self.fst.is_final(i) for (i, ys) in state)
