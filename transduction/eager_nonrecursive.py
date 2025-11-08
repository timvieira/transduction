from transduction.fst import FST
from transduction.base import AbstractAlgorithm
from transduction.fsa import FSA

from arsenal.cache import memoize


def build_precover_dfa(fst, target, target_alphabet):
    "DFA representing the complete precover."
    # this is a copy machine for target \targetAlphabet^*
    m = FST()
    m.add_I(0)
    N = len(target)
    for i in range(N):
        m.add_arc(i, target[i], target[i], i+1)
    for x in target_alphabet:
        m.add_arc(N, x, x, N)
    m.add_F(N)
    return (fst @ m).project(0).min()


def force_start(fsa, start_states):
    new = FSA()
    for q in start_states:
        new.add_start(q)
    for i, a, j in fsa.arcs():
        new.add(i, a, j)
    for i in fsa.stop:
        new.add_stop(i)
    return new


def is_universal(fsa, q, alphabet):
    # universality test; below we create the force-start machine
    q_fsa = force_start(fsa, [q]).min()
    if len(q_fsa.states) != 1:
        return False
    [i] = q_fsa.states
    for a in alphabet:
        if set(q_fsa.arcs(i, a)) != {i}:
            return False
    return True


class EagerNonrecursive(AbstractAlgorithm):
    """
    Eager, non-recursive DFA-based algorithm.
    """

    def __init__(self, fst, **kwargs):
        super().__init__(fst, **kwargs)
        # the variables below need to be used very carefully
        self.state = None

    def decompose_fsa(self, target):

        # make sure that the DFA is minimizd so that we can use the simple
        # self-loop-elimination strategy on universal state to get the quotient.
        precover_dfa = self.precover_dfa(target).min()

        # identify all universal states
        universal_states = {i for i in precover_dfa.states
                            if is_universal(precover_dfa, i, self.source_alphabet)}
        #print(universal_states)

        Q = FSA()
        # copy start states
        for i in precover_dfa.start:
            Q.add_start(i)
        # copy all but self-arcs on the [minimized] universal states.
        for i,a,j in precover_dfa.arcs():
            # skip self-arcs
            if i == j and i in universal_states: continue  # drop these
            Q.add(i, a, j)
        # replace accepting states; make universal states accepting
        for i in universal_states:
            Q.add_stop(i)
        Q = Q.min()

        # XXX: There might be a simpler algorithm that returns the set of
        # remainders by eliminating the unviersal states and keeping the other
        # accepting states.

        U = FSA.universal(self.source_alphabet)

        # figure out the remainder by set subtraction (i.e., P - Q X = P & ~(Q X))
        R = (precover_dfa - Q * U).min()

        return (Q, R)

    def initialize(self, target):
        self.state = {}
        dfa = self.precover_dfa(target)
        if len(dfa.start) == 0: return    # empty!
        [state] = dfa.start
        self.state[self.empty_source] = state
        yield self.empty_source

    def candidates(self, xs, target):
        dfa = self.precover_dfa(target)
        state = self.state[xs]
        for source_symbol, next_state in dfa.arcs(state):   # DFA
            next_xs = self.extend(xs, source_symbol)
            self.state[next_xs] = next_state
            yield next_xs

    @memoize
    def precover_dfa(self, target):
        return build_precover_dfa(self.fst, target, self.target_alphabet)

    def discontinuity(self, xs, target):
        #assert not self.continuity(xs, target)
        return (self.state[xs] in self.precover_dfa(target).stop)

    def continuity(self, xs, target):
        return self._continuity(target, self.state[xs])

    @memoize
    def _continuity(self, target, state):
        dfa = self.precover_dfa(target)
        assert state in dfa.states
        return is_universal(dfa, state, self.source_alphabet)
