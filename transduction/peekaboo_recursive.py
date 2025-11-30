from transduction.base import PrecoverDecomp
from transduction.lazy import Lazy
from transduction.fsa import FSA
from transduction.fst import FST, EPSILON
from transduction.eager_nonrecursive import EagerNonrecursive
from transduction.lazy_recursive import LazyRecursive
#from transduction.lazy_nonrecursive import LazyNonrecursive

from arsenal import colors
from collections import deque


class Peekaboo:
    """
    Recursive, batched computation of next-target-symbol optimal DFA-decomposition.
    """

    def __init__(self, fst, max_steps=float('inf')):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.max_steps = max_steps
        self._cache = {}

    def __call__(self, target):
        val = self._cache.get(target)
        if val is None:
            val = self._cache[target] = self.compute(target)
        return val

    def compute(self, target):

        dfa = BufferedRelevance(self.fst, target).det()

        if len(target) == 0:
            worklist = deque()
            states = set()
            arcs = []

            for state in dfa.start():
                worklist.append(state)
                states.add(state)

        else:

            parent = self(target[:-1])[target[-1]]

            # put previous and Q and R states on the worklist
            worklist = deque()
            worklist.extend(parent.quotient.stop)
            worklist.extend(parent.remainder.stop)

            # TODO: copying is slow
            states = parent.quotient.states.copy()
            arcs = [(i,x,j) for i in states for x, j in parent.quotient.arcs(i)]


        precover = {y: PrecoverDecomp(set(), set()) for y in self.target_alphabet}


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
                dfa_truncated = TruncatedDFA(dfa=dfa, fst=self.fst, target=target + y)
                #assert dfa_filtered.materialize().equal(Precover(self.fst, target + y).min)
                #print('ok:', repr(target + y))
                if dfa_truncated.accepts_universal(state, self.source_alphabet):
                    precover[y].quotient.add(state)
                    continuous.add(y)
                elif dfa_truncated.is_final(state):
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
class BufferedRelevance(Lazy):
    """NOTE: this is a semi-automaton as it does not have an `is_final` method.

    It implements a state space that tracks the states of an FST `fst` along
    with the target string they generate.  It prunes the state space to just the
    states that are relevant to `target` followed by at least additional target
    symbol.

    """

    # TODO: this machine is not longer specific to the peekaboo construction it
    # is just the target-side buffering construction with the target-relevance
    # filter.
    def __init__(self, fst, target):
        self.fst = fst
        self.target = target

    def arcs(self, state):
        (i, ys) = state
        n = len(ys)
        N = len(self.target)
        m = min(n, N)
        if ys[:m] == self.target[:m]:
            for x,y,j in self.fst.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                else:
                    yield (x, (j, ys + y))

    def start(self):
        for i in self.fst.I:
            yield (i, '')


class TruncatedDFA(Lazy):
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

    def refine(self, frontier):
        # clip the target side side to `y` in order to mimick the states of the
        # composition machine that we used in the new lazy, nonrecursive
        # algorithm.
        N = len(self.target)
        return frozenset({
            (i, ys[:N]) for i, ys in frontier
            if ys[:min(N, len(ys))] == self.target[:min(N, len(ys))]
        })

    def arcs(self, state):
        for x, next_state in self.dfa.arcs(state):
            yield x, self.refine(next_state)

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
        self.depth = depth
        self.peekaboo = Peekaboo(fst)
        self.reference = LazyRecursive(fst)
#        self.reference = LazyNonrecursive(fst)
#        self.reference = EagerNonrecursive(fst)
        self.verbosity = verbosity
        self.run(target, depth)

    def run(self, target, depth):
        if depth == 0: return
        want = {y: self.reference(target + y) for y in self.reference.target_alphabet}
        have = self.peekaboo(target)
        #assert have == want, f"""\ntarget = {target!r}\nhave = {have}\nwant = {want}\n"""
        assert_equal_decomp_map(have, want)

        for y in want:
            if self.verbosity > 0: print('>', repr(target + y))
            if want[y].quotient or want[y].remainder:   # nonempty
                self.run(target + y, depth - 1)


def assert_equal_decomp_map(have, want):
    for y in have | want:
        assert have[y].quotient.equal(want[y].quotient)
        assert have[y].remainder.equal(want[y].remainder)


def test_abc():
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])

    p = Peekaboo(fst)
    target = ''
    have = p(target)
    tmp = EagerNonrecursive(fst)
    want = {y: tmp(target + y) for y in tmp.target_alphabet}
    assert_equal_decomp_map(have, want)

    target = 'abc'
    have = p(target)
    tmp = EagerNonrecursive(fst)
    want = {y: tmp(target + y) for y in tmp.target_alphabet}
    assert_equal_decomp_map(have, want)

    recursive_testing(fst, '', depth=5)


def test_samuel():
    fst = examples.samuel_example()

    p = Peekaboo(fst)
    target = 'y'
    have = p(target)
    tmp = EagerNonrecursive(fst)
    want = {y: tmp(target + y) for y in tmp.target_alphabet}

    #print(have)
    #print(want)

    assert_equal_decomp_map(have, want)

    recursive_testing(fst, '', depth=5)


def test_small():

    fst = FST()
    fst.add_I(0)
    fst.add_F(0)

    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'x', 2)

    fst.add_arc(2, 'a', 'a', 3)
    fst.add_arc(2, 'b', 'b', 3)

    fst.add_arc(3, 'a', 'a', 3)
    fst.add_arc(3, 'b', 'b', 3)

    fst.add_F(1)
    fst.add_F(3)

    recursive_testing(fst, '', depth=5)


def test_sdd1():
    fst = examples.sdd1_fst()
    recursive_testing(fst, '', depth=5)


def test_duplicate():
    fst = examples.duplicate(set('12345'))
    recursive_testing(fst, '', depth=5)


def test_number_comma_separator():
    import string
    #fst = examples.number_comma_separator(set(string.printable) - set('\t\n\r\x0b\x0c'))
    fst = examples.number_comma_separator({'a','b',',',' ','0','1'}, Digit={'0', '1'})

    recursive_testing(fst, '', depth=5)

    recursive_testing(fst, '0,| 1,', depth=1, verbosity=1)
    recursive_testing(fst, '0,| 1,|', depth=1, verbosity=1)


def test_newspeak2():
    from transduction import Precover
    fst = examples.newspeak2()
    p = Peekaboo(fst, max_steps=500)
    target = ''
    have = p(target)
    tmp = EagerNonrecursive(fst)
    want = {y: tmp(target + y) for y in tmp.target_alphabet}

    assert_equal_decomp_map(have, want)

    #print('have=', have)
    #print('want=', want)

#    for y in have | want:
#        if have.get(y) == want.get(y):
#            print(colors.mark(True), repr(y))
#        else:
#            print(colors.mark(False), repr(y))
#            print('  have=', have.get(y))
#            print('  want=', want.get(y))
#            #Precover(fst, target + y).check_decomposition(*want[y], throw=True)
#            Precover(fst, target + y).check_decomposition(*have[y], throw=False)
#    assert have == want

    #recursive_testing(fst, '', depth=5)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
