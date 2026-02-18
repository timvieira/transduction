from transduction.base import AbstractAlgorithm
from transduction.lazy import is_universal
from transduction.precover import Precover
from transduction.util import memoize


class EagerNonrecursive(AbstractAlgorithm):
    """
    Eager, non-recursive DFA-based algorithm.
    """

    def __init__(self, fst, **kwargs):
        super().__init__(fst, **kwargs)
        # the variables below need to be used very carefully
        self.state = None
        self.dfa = None

    def initialize(self, target):
        self.state = {}
        self.dfa = Precover(self.fst, target).min
        for state in self.dfa.start:
            self.state[()] = state
            yield ()

    def candidates(self, xs, target):
        for source_symbol, next_state in self.dfa.arcs(self.state[xs]):
            next_xs = xs + (source_symbol,)
            self.state[next_xs] = next_state
            yield next_xs

    def discontinuity(self, xs, target):
        return self.dfa.is_final(self.state[xs])

    def continuity(self, xs, target):
        return self._continuity(target, self.state[xs])

    @memoize
    def _continuity(self, target, state):   # pylint: disable=W0613
        return is_universal(self.dfa, state, self.source_alphabet)
