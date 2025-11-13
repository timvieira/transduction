from transduction.lazy import Lazy
from transduction.base import AbstractAlgorithm, PrecoverDecomp
from transduction.fst import EPSILON
from transduction.fsa import FSA

from arsenal.cache import memoize


class LazyPrecoverNFA(Lazy):
    r"""`LazyPrecoverNFA(f, target)` implements the precover for the string `target` in the
    FST `f` as a lazy, nondeterministic finite-state automaton.  Mathematically, the precover
    is given by the following automata-theoretic operations:
    $$
    \mathrm{proj}_{\mathcal{X}}\Big( \texttt{f} \circ \boldsymbol{y}\mathcal{Y}^* \Big)
    $$
    where target is $\boldsymbol{y} \in \mathcal{Y}^*$ and $\texttt{f}$ is a transducer
    implementing a function from some space $\mathcal{X}^*$ to $\mathcal{Y}^*$.

    Mathematically, the precover represents the subset of $\mathcal{X}^*$ that
    we need to sum over in order to compute the prefix probability of
    $\boldsymbol{y}$ in the pushforward of $\texttt{f}$.

    """

    def __init__(self, f, target):
        self.f = f
        self.target = target
        self.N = len(target)

    def arcs(self, state):
        # states of this machine are pairs `(i,n)` where `i` is a state in `f` and `n` is a state in the line+loop machine on the right.
        (i, ys) = state
        assert isinstance(ys, (bytes, tuple, str)), [type(ys), ys]
        # Handle the right machine's loop state separately
        if ys == self.target:
            for a,b,j in self.f.arcs(i):
                if b == EPSILON:
                    # we don't advance in the right machine because we didn't actually match anything... not that it
                    # actually matters because we stay put here anyway!
                    yield (a, (j, ys))
                else:
                    # we advance in the left machine, but stay in the right machine's final state (because we traverse a self-arc)
                    yield (a, (j, ys))
        else:
            n = len(ys)
            # TODO: it would be nice if this iteration were filtered more efficiently
            for a,b,j in self.f.arcs(i):
                if b == EPSILON:
                    # we don't advance in the right machine because we didn't actually match anything
                    yield (a, (j, ys))
                elif b == self.target[n]:
                    # advance in both the left and the right machine along this new edge.
                    # note that we input-project, so we drop the output symbol (i.e., the output is an FSA not an FST)
                    yield (a, (j, self.target[:n+1]))

    def start(self):
        for i in self.f.I:
            yield (i, self.target[:0])

    def is_final(self, state):
        (i, ys) = state
        return (i in self.f.F) and ys == self.target


class LazyNonrecursive(AbstractAlgorithm):
    """
    Lazy, non-recursive DFA-based algorithm.
    """

    def __init__(self, fst, **kwargs):
        super().__init__(fst, **kwargs)
        # The variables below need to be used carefully
        self.state = None
        self.nfa = None
        self.dfa = None

    def initialize(self, target):
        self.state = {}
        self.nfa = LazyPrecoverNFA(self.fst, target)
        self.dfa = self.nfa.det()
        for state in self.dfa.start():
            self.state[self.empty_source] = state
            yield self.empty_source

    def candidates(self, xs, target): # pylint: disable=W0613
        for source_symbol, next_state in self.dfa.arcs(self.state[xs]):
            next_xs = self.extend(xs, source_symbol)
            self.state[next_xs] = next_state
            yield next_xs

    def discontinuity(self, xs, target):
        return self.dfa.is_final(self.state[xs])

    def continuity(self, xs, target):
        return self._continuity(target, self.state[xs])

    @memoize
    def _continuity(self, target, state):  # pylint: disable=W0613
        return self.dfa.accepts_universal(state, self.source_alphabet)

    # TODO: note that we can do the partition lazily -- it's just a change of
    # the final state.  However, it is not clear how to share work between the
    # traversals.
    def dfa_decomposition(self, target):

        dfa = LazyPrecoverNFA(self.fst, target).det()

        Q = FSA()
        R = FSA()

        worklist = []
        visited = set()

        for i in dfa.start():
            worklist.append(i)
            Q.add_start(i)
            R.add_start(i)

        while worklist:
            i = worklist.pop()
            if i in visited: continue
            visited.add(i)

            if dfa.is_final(i):
                if dfa.accepts_universal(i, self.source_alphabet):
                    Q.add_stop(i)
                    continue       # will not expand further
                else:
                    R.add_stop(i)  # will expand further

            for a, j in dfa.arcs(i):
                worklist.append(j)

                Q.add_arc(i, a, j)
                R.add_arc(i, a, j)

        return PrecoverDecomp(Q.trim(), R.trim())
