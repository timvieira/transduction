from transduction.base import AbstractAlgorithm, DecompositionResult
from transduction.fsa import FSA, EPSILON
from transduction.universality import UniversalityFilter
from transduction.util import display_table
from transduction.lazy import is_universal
from transduction.precover_nfa import (
    PrecoverNFA as LazyPrecoverNFA,
    TruncationMarkerPrecoverNFA as LazyPrecoverNFAWithTruncationMarker,
    PopPrecoverNFA as PopPrecover,
)

from arsenal import colors
from arsenal.cache import memoize
from functools import cached_property
from collections import deque


class Precover(DecompositionResult):
    """
    Representation of the precover of target string `target` in the FST `fst`.

    Supports factoring into an optimal quotient--remainder pair, even when they
    may be infinite.  The key is to represent them each as automata.

    """

    @classmethod
    def factory(cls, fst, **kwargs):
        """Return a callable that creates Precover instances for a fixed FST.

        Usage::

            ref = Precover.factory(fst)
            result = ref('abc')   # -> Precover(fst, 'abc')
        """
        return lambda target: cls(fst, target, **kwargs)

    def __init__(self, fst, target, impl='push'):
        self.fst = fst
        self.target = target
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.U = FSA.universal(self.source_alphabet)
        self.impl = impl

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
        if self.impl == 'push':
            return LazyPrecoverNFA(self.fst, self.target)
        elif self.impl == 'push-truncated':
            return LazyPrecoverNFAWithTruncationMarker(self.fst, self.target)
        elif self.impl == 'pop':
            return PopPrecover(self.fst, self.target)
        else:
            raise ValueError(self.impl)

    @cached_property
    def decomposition(self):
        "Produce a quotient--remainder pair each represented as automata."
        P = self.det
        P = P.materialize()

        # identify all universal states
        filt = UniversalityFilter(self.fst, self.target, P, self.source_alphabet)
        universal_states = {i for i in P.stop if filt.is_universal(i)}

        # copy all arcs except those leaving universal states
        arcs = [(i,a,j) for i in P.states - universal_states for a,j in P.arcs(i)]

        # copy start states
        Q = FSA(start=P.start, arcs=arcs, stop=universal_states)           # replace accepting states with just universal states
        R = FSA(start=P.start, arcs=arcs, stop=P.stop - universal_states)  # keep non-universal accepting states

        # Double-check the remainder through set subtraction
        #assert R.equal(P - Q * self.U)

        return (Q, R)

    @property
    def quotient(self):
        "Optimal quotient automaton"
        return self.decomposition[0]

    @property
    def remainder(self):
        "Optimal remainder automaton"
        return self.decomposition[1]

    def is_cylinder(self, xs):
        "Is the source string `xs` a cylinder of the precover?"
        return FSA.from_string(xs) * self.U <= self.min

    def is_valid(self, Q, R):
        "Is the decomposition (Q, R) valid?"
        return self.min.equal(FSA.from_strings(Q) * self.U + FSA.from_strings(R))

    def find_cylinder_prefixes(self, xs):
        "Find all strict prefixes that are cylinders of the precover."
        for t in range(len(xs)):
            xss = xs[:t]
            if self.is_cylinder(xss):
                yield xss

    def check_decomposition(self, Q, R, throw=False, skip_validity=False):
        "Analyze the decompositions Q and R: is it valid? optimal?  Note that these tests only terminate when Q and R are finite sets."
        if isinstance(Q, FSA): Q = Q.language()
        if isinstance(R, FSA): R = R.language()
        ok = True
        z = skip_validity or self.is_valid(Q, R)   # check validity of the decomposition
        ok &= z
        print('check decomposition:')
        print('├─ valid:', colors.mark(z), 'equal to precover')
        assert z
        if Q: print('├─ quotient:')
        for xs in Q:
            # Elements of the quotient should all be cylinders
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
        "A simple visualization of the decomposition for IPython notebooks."
        Q,R = self.decomposition
        if minimize: Q, R = Q.min(), R.min()
        if trim:     Q, R = Q.trim(), R.trim()
        display_table([[Q, R]], headings=['quotient', 'remainder'])

    def graphviz(self, trim=False):
        "Stylized graphviz representation of the precover DFA where colors denotes properties of the state (green: quotient, magenta: remainder, white: useless)"
        dfa = self.det.materialize()
        if trim: dfa = dfa.trim()
        universal_states = {i for i in dfa.stop if is_universal(dfa, i, self.source_alphabet)}
        dead_states = dfa.states - dfa.trim().states
        def color_node(x):
            if x in universal_states: return '#90EE90'
            elif dfa.is_final(x): return '#f26fec'
            elif x in dead_states: return 'white'
            else: return '#f2d66f'
        return dfa.graphviz(fmt_node=set, sty_node=lambda x: {'style': 'filled,rounded', 'fillcolor': color_node(x)})

    def _repr_mimebundle_(self, *args, **kwargs):
        "For visualization purposes in notebook."
        return self.graphviz()._repr_mimebundle_(*args, **kwargs)


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
    def _continuity(self, target, state):   # pylint: disable=W0613
        return is_universal(self.dfa, state, self.source_alphabet)
