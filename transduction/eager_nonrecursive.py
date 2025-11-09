from transduction.base import AbstractAlgorithm
from transduction.fst import FST, EPSILON
from transduction.fsa import FSA

from arsenal import colors
from arsenal.cache import memoize
from functools import cached_property


class Precover:
    """Representation of the precover of target string `target` in the FST `fst`.

    Supports factoring into an optimal quotient--remainder pair, even when they
    may be infinite.  The key is to represent them each as automata.

    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = target
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

    @cached_property
    def dfa(self):
        "DFA representing the complete precover."
        return self.fsa.min()

    @cached_property
    def fsa(self):
        "FSA representing the complete precover."
        # this is a copy machine for target \targetAlphabet^*
        m = FST()
        m.add_I(0)
        N = len(self.target)
        for i in range(N):
            m.add_arc(i, self.target[i], self.target[i], i+1)
        for x in self.target_alphabet:
            m.add_arc(N, x, x, N)
        m.add_F(N)
        return (self.fst @ m).project(0)

    @cached_property
    def decomposition(self):
        # make sure that the DFA is minimizd so that we can use the simple
        # self-loop-elimination strategy on universal state to get the quotient.
        precover_dfa = self.dfa.min()

        # identify all universal states
        universal_states = {i for i in precover_dfa.states
                            if is_universal(precover_dfa, i, self.source_alphabet)}
        #print(universal_states)
        assert len(universal_states) <= 1

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
        # remainders by eliminating the universal states and keeping the other
        # accepting states.

        U = FSA.universal(self.source_alphabet)

        # figure out the remainder by set subtraction (i.e., P - Q X = P & ~(Q X))
        R = (precover_dfa - Q * U).min()

        return (Q, R)

    @cached_property
    def quotient(self):
        "Optimal quotient automaton"
        return self.decomposition[0]

    @cached_property
    def remainder(self):
        "Optimal remainder automaton"
        return self.decomposition[1]

    def _repr_mimebundle_(self, *args, **kwargs):
        "For visualization purposes in notebook."
        return self.dfa._repr_mimebundle_(*args, **kwargs)

    def check_decomposition(self, Q, R, throw=False):
        "Analyze the decompositions Q and R: is it valid? optimal?"
        P = self.dfa

        U = FSA.universal(self.source_alphabet)

        ok = True

        # check validity
        z = P.equal(FSA.from_strings(Q) * U + FSA.from_strings(R))

        ok &= z
        print('check decomposition:')
        print('├─ valid:', colors.mark(z), 'equal to precover')
        assert z

        if Q: print('├─ quotient:')
        for xs in Q:
            z = FSA.from_string(xs) * U <= P
            print('├─', colors.mark(z), repr(xs), 'is a valid cylinder')
            ok &= z

            # minimality -- if there is a strict prefix that is a cylinder of
            # the precover, then `xs` is not minimal.  in terms of the precover
            # automaton, `xs` leads to a universal state, but we want to make
            # sure that there wasn't an earlier time that it arrived there.
            for t in range(len(xs)):
                xss = xs[:t]
                if FSA.from_string(xss) * U <= P:
                    print('├─', colors.mark(False), repr(xs), 'is not minimal because its strict prefix', repr(xss), 'is a cylinder of the precover')
                    ok &= False

        if len(R): print('├─ remainder:')
        for xs in R:
            z = (xs in P)
            zz = not (FSA.from_string(xs) * U <= P)
            print('├─', colors.mark(z & zz), repr(xs), 'in precover and not a cylinder')
            ok &= z
            ok &= zz
        print('└─ overall:', colors.mark(ok))
        assert not throw or ok


def force_start(fsa, start_state):
    assert start_state in fsa.states
    new = FSA()
    new.add_start(start_state)
    for i, a, j in fsa.arcs():
        new.add(i, a, j)
    for i in fsa.stop:
        new.add_stop(i)
    return new


def is_universal(fsa, q, alphabet):
    # universality test; below we create the force-start machine
    q_fsa = force_start(fsa, q).min()
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
        return Precover(self.fst, target).dfa

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
