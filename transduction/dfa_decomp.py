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


def check(a):
    Q, R = Precover(a.fst, a.target).decomposition
    assert Q.equal(a.Q) and R.equal(a.R)
    print(colors.mark(True), 'pass')


def test_samuel():
    f = examples.samuel_example()
    target = 'cxx'

    start = RecursiveDFADecomposition(f, target[:0])
    check(start)

    a = (start >> target[0])
    check(a)

    ab = (a >> target[1])
    check(ab)

    abc = (ab >> target[2])
    check(abc)


def test_newspeak2():

    f = examples.newspeak2()
    target = 'bad'

    start = RecursiveDFADecomposition(f, target[:0])
    check(start)

    a = (start >> target[0])
    check(a)

    ab = (a >> target[1])
    check(ab)

    abc = (ab >> target[2])
    check(abc)


def test_simple_replace():
    f = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])
    target = 'abc'

    start = RecursiveDFADecomposition(f, target[:0])
    check(start)

    a = (start >> target[0])
    check(a)

    ab = (a >> target[1])
    check(ab)

    abc = (ab >> target[2])
    check(abc)



def test_parity():
    # Note: this doesn't really test the recursion.
    f = examples.parity({'a'})
    s = RecursiveDFADecomposition(f, '')
    check(s)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
