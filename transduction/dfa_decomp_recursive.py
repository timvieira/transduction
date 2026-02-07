from transduction import FSA, EPSILON
from transduction.precover_nfa import TargetSideBuffer, Relevance
from collections import deque


# XXX: Warning: this algorithm doesn't work in all cases.  It currently fails to
# terminate on the `triplets_of_doom` test case.  The issue is that it does not
# truncate the target buffer.
class RecursiveDFADecomp:

    def __init__(self, fst, target, parent=None):

        self.parent = parent
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.fst = fst
        self.target = target

        # XXX: Warning: this machine may have infinitely many states.
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
        return RecursiveDFADecomp(self.fst, self.target + y, parent=self)
