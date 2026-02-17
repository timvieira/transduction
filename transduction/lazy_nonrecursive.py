from transduction.base import AbstractAlgorithm
from transduction.precover_nfa import PrecoverNFA as LazyPrecoverNFA

from transduction.util import memoize


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
        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")
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
