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


#_______________________________________________________________________________
# TESTING CODE BELOW

from transduction import examples


class recursive_testing:
    """
    Utility function for testing the `Peekaboo` method against a slower method.
    """
    def __init__(self, fst, target, depth, verbosity=0):
        self.fst = fst
        self.target_alphabet = self.fst.B - {EPSILON}
        self.depth = depth
        self.peekaboo = Peekaboo(fst)
        self.reference = lambda target: Precover(fst, target)
#        self.reference = LazyNonrecursive(fst)
#        self.reference = EagerNonrecursive(fst)
        self.verbosity = verbosity
        self.run(target, depth)

    def run(self, target, depth):
        if depth == 0: return

        # Check that the peekaboo machine matches the reference implementation
        have = PeekabooPrecover(self.fst, target).materialize()
        want = (self.fst @ (FSA.from_string(target) * FSA.from_strings(self.target_alphabet).p())).project(0)
        assert have.equal(want)

        # Check that the decomposition matches the reference implementation
        want = {y: self.reference(target + y) for y in self.target_alphabet}
        have = self.peekaboo(target)
        assert_equal_decomp_map(have, want)

        # Recurse
        for y in want:
            if self.verbosity > 0: print('>', repr(target + y))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                self.run(target + y, depth - 1)


def assert_equal_decomp_map(have, want):
    for y in have | want:
        assert have[y].quotient.equal(want[y].quotient)
        assert have[y].remainder.equal(want[y].remainder)


def test_abc():
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])
    recursive_testing(fst, '', depth=4)


def test_samuel():
    fst = examples.samuel_example()
    recursive_testing(fst, '', depth=5)


def test_small():
    fst = examples.small()
    recursive_testing(fst, '', depth=5)


def test_sdd1():
    fst = examples.sdd1_fst()
    recursive_testing(fst, '', depth=5)


def test_duplicate():
    fst = examples.duplicate(set('12345'))
    recursive_testing(fst, '', depth=5)


def test_number_comma_separator():
    #import string
    #fst = examples.number_comma_separator(set(string.printable) - set('\t\n\r\x0b\x0c'))
    fst = examples.number_comma_separator({'a', ',', ' ', '0'}, Digit={'0'})
    recursive_testing(fst, '', depth=4, verbosity=1)
    recursive_testing(fst, '0,| 0,', depth=1, verbosity=1)
    recursive_testing(fst, '0,| 0,|', depth=1, verbosity=1)


def test_newspeak2():
    fst = examples.newspeak2()
    recursive_testing(fst, '', depth=1)
    recursive_testing(fst, 'ba', depth=1)
    recursive_testing(fst, 'bad', depth=1)


def test_lookahead():
    fst = examples.lookahead()
    recursive_testing(fst, '', depth=6, verbosity=1)


def test_weird_copy():
    fst = examples.weird_copy()
    recursive_testing(fst, '', depth=5, verbosity=0)


def test_triplets_of_doom():
    fst = examples.triplets_of_doom()
    recursive_testing(fst, '', depth=13, verbosity=0)


def test_infinite_quotient():
    fst = examples.infinite_quotient()
    recursive_testing(fst, '', depth=5, verbosity=1)


def test_parity():
    fst = examples.parity({'a', 'b'})
    recursive_testing(fst, '', depth=5, verbosity=1)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
