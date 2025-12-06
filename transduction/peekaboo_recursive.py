from transduction.base import PrecoverDecomp
from transduction.lazy import Lazy
from transduction.fsa import FSA, frozenset
from transduction.fst import FST, EPSILON
from transduction.eager_nonrecursive import EagerNonrecursive
from transduction.lazy_recursive import LazyRecursive
#from transduction.lazy_nonrecursive import LazyNonrecursive

from arsenal import colors
from collections import deque

MAX_STEPS = float('inf')


class Peekaboo:
    """
    Recursive, batched computation of next-target-symbol optimal DFA-decomposition.
    """
    def __init__(self, fst):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

    def __call__(self, target):
        s = PeekabooState(self.fst, '', parent=None)

        ARCS = [(i,x,j) for j, arcs in s._arcs.items() for x,i in arcs]
        for x in target:
            s >>= x
            ARCS.extend((i,x,j) for j, arcs in s._arcs.items() for x,i in arcs)

        foo = {}
        for y in self.target_alphabet:
            q = FSA(start=set(s.dfa.start()), arcs=ARCS, stop=s.foo[y].quotient)
            r = FSA(start=set(s.dfa.start()), arcs=ARCS, stop=s.foo[y].remainder)
            foo[y] = PrecoverDecomp(q, r)

        return foo


class PeekabooState:

    def __init__(self, fst, target, parent):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.target = target
        self.parent = parent

        assert parent is None or parent.target == target[:-1]

        # Warning: The BufferedRelevance machine is not finite state!
        #
        # [2025-12-02 Tue] Possible issue: we could do an unbounded amount of
        #   useless work by enumerating states in the buffered relevance machine
        #   that are not on track to participate in any relevant target string's
        #   precover.
        #
        # [2025-12-04 Thu] I think we ned to do something similar ot the
        #   PeekabooPrecover construction so that we can only explore a finite
        #   number of states and, thus, terminate in finite time.  The key to
        #   this is going to be to either tweak the state space as in
        #   PeekabooPrecover or to use something like we did with `refine`.  The
        #   concert about the PeekabooPrecover strategy is that we might add
        #   edges to the machine that need to be removed/ignored in the next
        #   layer. (The picture that I have in my head is something like change
        #   propagation where we have to invalidate some of the work that we
        #   done under the assumption that the target string ended here.
        #   Aesthetically and practically, I would rather not require invalidate
        #   work.  There is some guiding principle that is lacking here and I
        #   don't know how to think about it.)  Maybe the best way to move
        #   forward with this is to make a few visualizations of what I expected
        #   the layered structure to look like (e.g., by grafting a set of
        #   correct machines for each next symbol together and seeing where
        #   things diverge from from that idealization).

        dfa = BufferedRelevance(self.fst, target).det()
#        dfa = PeekabooPrecover(self.fst, target).det()

        self.dfa = dfa

        if len(target) == 0:
            assert parent is None
            worklist = deque()

            self._arcs = {}
            for state in dfa.start():
                worklist.append(state)
                self._arcs[state] = set()

        else:

            # select the relevant next symbol from the previous computation
            p = parent.foo[target[-1]]

            # put previous and Q and R states on the worklist
            worklist = deque()
            worklist.extend(p.quotient)
            worklist.extend(p.remainder)
            self._arcs = {} #parent._arcs.copy()
            for x in worklist:
                assert x in parent._arcs
                self._arcs[x] = set()

            #self._arcs.update(parent._arcs)

        precover = {y: PrecoverDecomp(set(), set()) for y in self.target_alphabet}


        def refine(frontier):
            "For caching and cycle-detection, we truncate the state representation after N+1 target symbols."
            #return frontier
            return frozenset((i, ys[:N+1]) for i, ys in frontier)


        N = len(target)
        t = 0
        visited = set()
        while worklist:
            state = worklist.popleft()
            t += 1
#            print(state)

            if t >= MAX_STEPS:
                print(colors.light.red % 'TOOK TOO LONG')
                break

            relevant_symbols = {ys[N] for _, ys in state if len(ys) > N}
#            print(f'{relevant_symbols=}')

            # Shortcut: At most one of the `relevant_symbols` can be
            # continuous. If we find one, we can stop expanding.
            continuous = set()
            for y in relevant_symbols:

                # XXX: we may have already constructed this machine
                dfa_truncated = TruncatedDFA(dfa=dfa, fst=self.fst, target=target + y)

                if dfa_truncated.accepts_universal(state, self.source_alphabet):
                    precover[y].quotient.add(state)
                    continuous.add(y)

                elif dfa_truncated.is_final(state):
                    precover[y].remainder.add(state)

            assert len(continuous) <= 1
#            print(f'{continuous=}')
            if continuous:
                continue    # we have found a quotient and can skip

            for x, next_state in dfa.arcs(state):

                # [2025-12-05 Fri] A possible solution to the infinite running
                # time is to do memoization/cycle detection on coarse grained
                # nodes---much like we do the universality test.
                r = refine(next_state)
                if r not in visited:
                    worklist.append(next_state)

                if next_state not in self._arcs:
                    self._arcs[next_state] = set()

                self._arcs[next_state].add((x, state))

                # [2025-12-02 Tue] unfortunately, these graphs aren't always
                # cleanly layered; we can have arcs that go backward (e.g., when
                # Q and R are cyclical) [TODO: add examples] We can also have
                # empty layers (these are layers where the nodes are in previous
                # layers - just imagine a case where Q(abc) = Q(ab)).

                #p = parent
                #while p is not None:
                #    assert next_state not in p._arcs
                #    p = p.parent

        self.foo = precover

    def __rshift__(self, y):
        return PeekabooState(self.fst, self.target + y, parent=self)


# TODO: in order to predict EOS, we need to extract the preimage from Q and R
class BufferedRelevance(Lazy):
    """NOTE: this is a semi-automaton as it does not have an `is_final` method.

    It implements a state space that tracks the states of an FST `fst` along
    with the target string they generate.  It prunes the state space to just the
    states that are relevant to `target` followed by at least additional target
    symbol.

    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = target

    def arcs(self, state):
        (i, ys) = state
        n = len(ys)
        N = len(self.target)
        m = min(n, N)
        assert ys[:m] == self.target[:m]

        for x,y,j in self.fst.arcs(i):
            if y == EPSILON:
                yield (x, (j, ys))
            elif n >= N or y == self.target[n]:
                yield (x, (j, ys + y))

#        assert n <= N+1
#        assert ys.startswith(self.target) or self.target.startswith(ys)
#
#        #if ys[:m] == self.target[:m]:
#        for x,y,j in self.fst.arcs(i):
#            if y == EPSILON:
#                yield (x, (j, ys))
#            ######################################################################
#            # XXX: I think the crux is that we have to insert some temporary
#            # arcs for the specific ply but then we might have to delete them
#            # when we move on to the next ply.
#            elif n > N:
#                yield (x, (j, ys))
#            ######################################################################
#            elif n == N:
#                yield (x, (j, ys + y))
#            elif n < N:
#                if y == self.target[n]:
#                    yield (x, (j, ys + y))

    def start(self):
        for i in self.fst.I:
            yield (i, '')


## TODO: in order to predict EOS, we need to extract the preimage from Q and R
#class PeekabooPrecover(Lazy):
#    """NOTE: this is a semi-automaton as it does not have an `is_final` method.
#
#    It implements a state space that tracks the states of an FST `fst` along
#    with the target string they generate.  It prunes the state space to just the
#    states that are relevant to `target` followed by at least additional target
#    symbol.
#
#    """
#
#    def __init__(self, fst, target):
#        self.fst = fst
#        self.target = target
#
#    def arcs(self, state):
#        (i, ys) = state
#        n = len(ys)
#        N = len(self.target)
#        if ys == self.target and n >= N:
#            for x,y,j in self.fst.arcs(i):
#                if y == EPSILON:
#                    yield (x, (j, ys))
#                else:
#                    yield (x, (j, ys + y))
#
#        # Note: we truncate the buffer after the (N+1)th symbol
#        # XXX: In the recursive algorithm, we would not do this!
#        elif ys.startswith(self.target) and n == N + 1:
#            for a,b,j in self.fst.arcs(i):
#                yield (a, (j, ys))
#
#        elif self.target.startswith(ys) and n < N:
#            for x,y,j in self.fst.arcs(i):
#                if y == EPSILON:
#                    yield (x, (j, ys))
#                elif y == self.target[len(ys)]:
#                    yield (x, (j, ys + y))
#
#    def start(self):
#        for i in self.fst.I:
#            yield (i, '')


class TruncatedDFA(Lazy):
    """NOTE: This class augments a determinized `PeekabooPrecover` semi-automaton by
    adding an appropriate `is_final` method so that it is a valid finite-state
    automaton that encodes `Precover(fst, target)`.


    # In other words, for all `target` strings:
    # dfa_truncated = TruncatedDFA(dfa=dfa, fst=self.fst, target=target)
    # assert dfa_filtered.materialize().equal(Precover(self.fst, target).dfa)


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
        return frozenset(
            (i, ys[:N]) for i, ys in frontier
            if ys[:min(N, len(ys))] == self.target[:min(N, len(ys))]
        )

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
    p = Peekaboo(fst)
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


def test_benchmark():
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])

    tmp = Peekaboo(fst)
    tmp('a'*1000)



if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
