from transduction import (
    FSA, FST, EPSILON, PrecoverDecomp, examples, Precover,
    display_table, format_table, HTML, colors, LazyPrecoverNFA
)


from transduction.lazy import Lazy

class TargetSideBuffer(Lazy):

    def __init__(self, f):
        self.f = f

    def arcs(self, state):
        (i, ys) = state
        for x,y,j in self.f.arcs(i):
            yield x, (j, ys + y)

    def start(self):
        for i in self.f.I:
            yield (i, '')

    def is_final(self, state):
        raise NotImplementedError()


class Relevance(Lazy):

    def __init__(self, base, target):
        self.base = base
        self.target = target

    def arcs(self, state):
        N = len(self.target)
        for x, (i, ys) in self.base.arcs(state):
            if self.target.startswith(ys) or ys.startswith(self.target):
                yield x, (i, ys)

    def start(self):
        yield from self.base.start()

    def is_final(self, state):
        raise NotImplementedError()


class RecursiveDFADecomposition:

    def __init__(self, fst, target, parent=None):

        self.parent = parent
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.fst = fst
        self.target = target

        # this fast because this machine is lazy
        self.dfa = Relevance(TargetSideBuffer(fst), target).det()

        if parent is None:
            worklist = []

            self._start = set()
            self._states = set()
            self._arcs = []
            for i in self.dfa.start():
                worklist.append(i)
                self._start.add(i)

        else:
            assert fst is parent.fst
            self._start = self.parent._start

            # put previous and Q and R states on the worklist
            worklist = []
            worklist.extend(parent.Q.stop)
            worklist.extend(parent.R.stop)

            # TODO: copying is slow
            self._states = self.parent._states.copy()
            self._arcs = parent._arcs[:]
            #self._arcs = [(i,x,j) for (i,x,j) in parent._arcs if i in parent.R.stop]

        self._stop_Q = set()
        self._stop_R = set()

        while worklist:
            i = worklist.pop()
            #print(colors.yellow % 'work', i)

            if self.is_final(i, target):
                if self.is_universal(i, target):
                    self._stop_Q.add(i)
                    continue             # will not expand further
                else:
                    self._stop_R.add(i)  # will expand further

            for a, j in self.dfa.arcs(i):
                self._arcs.append((i, a, j))
                if j not in self._states:
                    worklist.append(j)
                    self._states.add(j)

        #assert len(worklist) == 0
        #assert Q.states == R.states == visited

    # TODO: better to rename Q -> quotient and R -> remainder for consistency.
    @property
    def quotient(self):
        return self.Q

    @property
    def remainder(self):
        return self.R

    @property
    def Q(self):
        return FSA(start=self._start, arcs=self._arcs, stop=self._stop_Q)

    @property
    def R(self):
        return FSA(start=self._start, arcs=self._arcs, stop=self._stop_R)

    def is_final(self, frontier, target):
        return any(ys.startswith(target) for i, ys in frontier if self.fst.is_final(i))

    def arcs(self, i):
        yield from self.dfa.arcs(i)

    def refine(self, frontier, target):
        # Clip the target string state variable to `target`, as this mimics the states of
        # the composition machine for the complete string `target`, we haven't limited it
        # to this point, which means that we have an infinite-state machine.
        N = len(target)
        return frozenset({
            (i, ys[:N]) for i, ys in frontier
            if ys[:min(N, len(ys))] == target[:min(N, len(ys))]
        })

    def is_universal(self, frontier, target):
        worklist = []
        worklist.append(frontier)
        visited = {self.refine(frontier, target)}
        while worklist:
            i = worklist.pop()
            if not self.is_final(i, target): return False
            dest = dict(self.arcs(i))
            for a in self.source_alphabet:
                if a not in dest: return False
                j = dest[a]
                jj = self.refine(j, target)
                if jj not in visited:
                    visited.add(jj)
                    worklist.append(j)
        return True

    def __rshift__(self, y):
        return RecursiveDFADecomposition(self.fst, self.target + y, parent=self)

    def __iter__(self):
        return iter([self.Q.trim(), self.R.trim()])

    # TODO: I think we should store and visualize the machine over the same
    # state space and just use different notations for each of the final state
    # types.
    def _repr_html_(self, *args, **kwargs):
        return format_table([self], headings=['quotient', 'remainder'])


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
        self.reference = lambda target: Precover(fst, target)
#        self.reference = LazyNonrecursive(fst)
#        self.reference = EagerNonrecursive(fst)
        self.verbosity = verbosity
        self.run(target, depth, RecursiveDFADecomposition(fst, target))

    def run(self, target, depth, state):
        if depth == 0: return
        want = {y: self.reference(target + y) for y in self.target_alphabet}
        have = {y: state >> y for y in self.target_alphabet}
        assert_equal_decomp_map(have, want)
        for y in want:
            if self.verbosity > 0: print('>', repr(target + y))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                self.run(target + y, depth - 1, have[y])


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
    #fst = examples.triplets_of_doom()
    #recursive_testing(fst, '', depth=13, verbosity=0)
    assert False, 'this test does not terminate'


def test_infinite_quotient():
    fst = examples.infinite_quotient()
    recursive_testing(fst, '', depth=5, verbosity=1)


def test_parity():
    fst = examples.parity({'a', 'b'})
    recursive_testing(fst, '', depth=5, verbosity=1)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
