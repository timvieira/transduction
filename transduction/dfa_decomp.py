from transduction import FSA, EPSILON, Precover, format_table
from transduction.eager_nonrecursive import LazyPrecoverNFA
from transduction.lazy import Lazy
from collections import deque


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
        for x, (i, ys) in self.base.arcs(state):
            if self.target.startswith(ys) or ys.startswith(self.target):
                yield x, (i, ys)

#    def arcs(self, state):
#        N = len(self.target)
#        for x, (i, ys) in self.base.arcs(state):
#            if self.target.startswith(ys) or ys.startswith(self.target):
#                yield x, (i, ys[:N])

    def start(self):
        yield from self.base.start()

    def is_final(self, state):
        raise NotImplementedError()


class NonrecursiveDFADecomp:

    def __init__(self, fst, target):
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        dfa = LazyPrecoverNFA(fst, target).det()

        Q = FSA()
        R = FSA()

        worklist = deque()
        visited = set()

        for i in dfa.start():
            worklist.append(i)
            visited.add(i)
            Q.add_start(i)
            R.add_start(i)

        while worklist:
            i = worklist.popleft()

            if dfa.is_final(i):
                if dfa.accepts_universal(i, self.source_alphabet):
                    Q.add_stop(i)
                    continue       # will not expand further
                else:
                    R.add_stop(i)  # will expand further

            for a, j in dfa.arcs(i):
                if j not in visited:
                    worklist.append(j)
                    visited.add(j)

                Q.add_arc(i, a, j)
                R.add_arc(i, a, j)

        self.fst = fst
        self.dfa = dfa
        self.target = target
        self.quotient = Q
        self.remainder = R


# XXX: Warning: this algorithm doesn't work in all cases.  It currently fails to
# terminate on the `triplets_of_doom` test case.  The issue is that it does not
# truncate the target buffer.
class RecursiveDFADecomposition:

    def __init__(self, fst, target, parent=None):

        self.parent = parent
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.fst = fst
        self.target = target

        # this fast because this machine is lazy
        self.dfa = Relevance(TargetSideBuffer(fst), target).det()

        worklist = deque()

        if parent is None:

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
            worklist.extend(parent._stop_Q)
            worklist.extend(parent._stop_R)

            # TODO: copying is slow
            self._states = self.parent._states.copy()
            #self._arcs = parent._arcs[:]
            self._arcs = [(i,x,j) for (i,x,j) in parent._arcs if i not in parent._stop_R]

        self._stop_Q = set()
        self._stop_R = set()

        while worklist:
            i = worklist.popleft()

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

    @property
    def quotient(self):
        return FSA(start=self._start, arcs=self._arcs, stop=self._stop_Q)

    @property
    def remainder(self):
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
        worklist = deque()
        worklist.append(frontier)
        visited = {self.refine(frontier, target)}
        while worklist:
            i = worklist.popleft()
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
        return iter([self.quotient, self.remainder])

    def _repr_html_(self):
        return format_table([self], headings=['quotient', 'remainder'])

    def check(self):
        P = Precover(self.fst, self.target)
        assert self.quotient.equal(P.quotient)
        assert self.remainder.equal(P.remainder)
