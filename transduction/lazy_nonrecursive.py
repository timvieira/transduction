from transduction.lazy import Lazy
from transduction.base import AbstractAlgorithm, PrecoverDecomp
from transduction.fst import EPSILON
from transduction.fsa import FSA
from transduction.eager_nonrecursive import LazyPrecoverNFA

from arsenal.cache import memoize


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
