from transduction.base import AbstractAlgorithm, PrecoverDecomp
from transduction.fst import FST, EPSILON
from transduction.fsa import FSA
from transduction.util import display_table

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
        self.U = FSA.universal(self.source_alphabet)

    @cached_property
    def det(self):
        "DFA representing the complete precover."
        return self.fsa.det()

    @cached_property
    def min(self):
        "Minimal DFA representing the complete precover."
        return self.fsa.min()

    @cached_property
    def fsa(self):
        "FSA representing the complete precover."
        # this is a copy machine for target \targetAlphabet^*
        return (self.fst @ self.target_prefixes).project(0)

    @cached_property
    def target_prefixes(self):
        m = FST()
        m.add_I(self.target[:0])
        N = len(self.target)
        for i in range(N):
            m.add_arc(self.target[:i], self.target[i], self.target[i], self.target[:i+1])
        for x in self.target_alphabet:
            m.add_arc(self.target, x, x, self.target)
        m.add_F(self.target)
        return m

    @cached_property
    def decomposition(self):
        P = self.det

        # identify all universal states
        universal_states = {i for i in P.stop if is_universal(P, i, self.source_alphabet)}

        # copy all arcs except those leaving universal states
        arcs = [(i,a,j) for i in P.states - universal_states for a,j in P.arcs(i)]

        # copy start states
        Q = FSA(start=P.start, arcs=arcs, stop=universal_states)           # replace accepting states with just universal states
        R = FSA(start=P.start, arcs=arcs, stop=P.stop - universal_states)  # keep non-universal accepting states

        # Double-check the remainder through set subtraction
        #assert R.equal(P - Q * self.U)

        return PrecoverDecomp(Q, R)

    @cached_property
    def quotient(self):
        "Optimal quotient automaton"
        return self.decomposition.quotient

    @cached_property
    def remainder(self):
        "Optimal remainder automaton"
        return self.decomposition.remainder

    def is_cylinder(self, xs):
        "Is the source string `xs` a cylinder of the precover?"
        return FSA.from_string(xs) * self.U <= self.min

    def is_valid(self, Q, R):
        "Is the decompositions (Q, R) valid?"
        return self.min.equal(FSA.from_strings(Q) * self.U + FSA.from_strings(R))

    def find_cylinder_prefixes(self, xs):
        "Find all strict prefixes that are cylinders of the precover."
        for t in range(len(xs)):
            xss = xs[:t]
            if self.is_cylinder(xss):
                yield xss

    def check_decomposition(self, Q, R, throw=False):
        "Analyze the decompositions Q and R: is it valid? optimal?"
        ok = True
        z = self.is_valid(Q, R)   # check validity of the decomposition
        ok &= z
        print('check decomposition:')
        print('├─ valid:', colors.mark(z), 'equal to precover')
        assert z
        if Q: print('├─ quotient:')
        for xs in Q:
            # Elements of the quotinet should all be cylinders
            z = self.is_cylinder(xs)
            print('├─', colors.mark(z), repr(xs), 'is a valid cylinder')
            ok &= z
            # minimality -- if there is a strict prefix that is a cylinder of
            # the precover, then `xs` is not minimal.  in terms of the precover
            # automaton, `xs` leads to a universal state, but we want to make
            # sure that there wasn't an earlier time that it arrived there.
            for xss in self.find_cylinder_prefixes(xs):
                print('├─', colors.mark(False), repr(xs), 'is not minimal because its strict prefix', repr(xss), 'is a cylinder of the precover')
                ok &= False
        if len(R): print('├─ remainder:')
        for xs in R:
            z = (xs in self.min)
            zz = not self.is_cylinder(xs)
            print('├─', colors.mark(z & zz), repr(xs), 'in precover and not a cylinder')
            ok &= z
            ok &= zz
        print('└─ overall:', colors.mark(ok))
        assert not throw or ok
        return ok

    def show_decomposition(self, minimize=True, trim=True):
        Q,R = self.decomposition
        if minimize:
            Q = Q.min()
            R = R.min()
        if trim:
            Q = Q.trim()
            R = R.trim()
        display_table([[Q, R]], headings=['quotient', 'remainder'])

    def graphviz(self):
        dfa = self.det
        universal_states = {i for i in dfa.stop if is_universal(dfa, i, self.source_alphabet)}
        dead_states = dfa.states - dfa.trim().states
        def color_node(x):
            if x in universal_states: return '#E6F0E6'
            elif dfa.is_final(x): return '#f26fec'
            elif x in dead_states: return 'white'
            else: return '#f2d66f'
        return dfa.graphviz(fmt_node=set, sty_node=lambda x: {'style': 'filled,rounded', 'fillcolor': color_node(x)})

    def _repr_mimebundle_(self, *args, **kwargs):
        "For visualization purposes in notebook."
        return self.graphviz()._repr_mimebundle_(*args, **kwargs)


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
    # universality test; best used on a DFA
    m = force_start(fsa, q).min()
    return len(m.states) == 1 and all(set(m.arcs(i, a)) == {i} for i in m.states for a in alphabet)


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
        if len(self.dfa.start) == 0: return    # empty!
        [state] = self.dfa.start
        self.state[self.empty_source] = state
        yield self.empty_source

    def candidates(self, xs, target):
        for source_symbol, next_state in self.dfa.arcs(self.state[xs]):
            next_xs = self.extend(xs, source_symbol)
            self.state[next_xs] = next_state
            yield next_xs

    def discontinuity(self, xs, target):
        return self.dfa.is_final(self.state[xs])

    def continuity(self, xs, target):
        return self._continuity(target, self.state[xs])

    @memoize
    def _continuity(self, target, state):
        return is_universal(self.dfa, state, self.source_alphabet)
