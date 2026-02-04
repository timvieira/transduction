from transduction.base import AbstractAlgorithm, PrecoverDecomp
from transduction.fsa import FSA, EPSILON
from transduction.fst import check_all_input_universal
from transduction.util import display_table
from transduction.lazy import Lazy

from arsenal import colors
from arsenal.cache import memoize
from functools import cached_property
from collections import deque


class LazyPrecoverNFAWithTruncationMarker(Lazy):
    """
    Alternative implementation of LazyPrecoverNFA, which augments the states
    with an additional bit, indicating if their output context has been
    truncated (i.e., if there are more symbols in the output buffer than there
    are in the state representation).
    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = target
        self.N = len(target)

    def arcs(self, state):
        (i, ys, truncated) = state
        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:      # target and ys are incompatible
            return
        if m == self.N:                    # i.e, target <= ys
            for x, y, j in self.fst.arcs(i):
                if y == EPSILON or truncated:
                    yield (x, (j, self.target, truncated))
                else:
                    yield (x, (j, self.target, True))
        else:                              # i.e, ys < target)
            assert not truncated
            for x, y, j in self.fst.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys, False))
                elif y == self.target[n]:
                    yield (x, (j, self.target[:n+1], False))

    def start(self):
        for i in self.fst.I:
            yield (i, self.target[:0], False)

    def is_final(self, state):
        (i, ys, _) = state
        return self.fst.is_final(i) and ys[:self.N] == self.target


class LazyPrecoverNFA(Lazy):
    r"""
    `LazyPrecoverNFA(f, target)` implements the precover for the string `target` in the
    FST `f` as a lazy, nondeterministic finite-state automaton.  Mathematically, the
    precover is given by the following automata-theoretic operations:
    $$
    \mathrm{proj}_{\mathcal{X}}\Big( \texttt{f} \circ \boldsymbol{y}\mathcal{Y}^* \Big)
    $$
    where target is $\boldsymbol{y} \in \mathcal{Y}^*$ and $\texttt{f}$ is a transducer
    implementing a function from some space $\mathcal{X}^*$ to $\mathcal{Y}^*$.

    Mathematically, the precover represents the subset of $\mathcal{X}^*$ that
    we need to sum over in order to compute the prefix probability of
    $\boldsymbol{y}$ in the pushforward of $\texttt{f}$.

    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = target
        self.N = len(target)

    def arcs(self, state):
        (i, ys) = state
        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:      # target and ys are not prefixes of one another.
            return
        if m == self.N:                    # i.e, target <= ys
            for x, _, j in self.fst.arcs(i):
                yield (x, (j, self.target))
        else:                              # i.e, ys < target)
            for x, y, j in self.fst.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                elif y == self.target[n]:
                    yield (x, (j, self.target[:n+1]))

    def start(self):
        for i in self.fst.I:
            yield (i, self.target[:0])

    def is_final(self, state):
        (i, ys) = state
        return self.fst.is_final(i) and ys[:self.N] == self.target


# [2025-12-14 Sun] Is it possible that with the pop version of the precover
#   automaton that we don't have to worry about truncation and other inefficient
#   things like that?  Is there some way in which we could run in "both
#   direction" some how?  I have this feeling that it is possible to switch
#   between the states as they ought to be totally isomorphic.  My worry with
#   the pop version below is that there would appear to be inherently less
#   sharing of work because the precovers of prefixes aren't guaranteed to be
#   useful (that said, due to truncation they aren't).  Maybe this is an
#   empirical question?
class PopPrecover(Lazy):
    """
    Equivalent to LazyPrecoverNFA, but the states are named differently as they are
    designed to pop from the target-string buffer rather than push.  This
    construction is better-suited for exploiting work on common suffixes.
    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = target
        self.N = len(target)

    def arcs(self, state):
        (i, ys) = state
        n = len(ys)
        if n == 0:
            for x, _, j in self.fst.arcs(i):
                yield (x, (j, ys))
        else:
            for x, y, j in self.fst.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                elif y == ys[0]:
                    yield (x, (j, ys[1:]))

    def start(self):
        for i in self.fst.I:
            yield (i, self.target)

    def is_final(self, state):
        (i, ys) = state
        return self.fst.is_final(i) and len(ys) == 0


class Precover:
    """
    Representation of the precover of target string `target` in the FST `fst`.

    Supports factoring into an optimal quotient--remainder pair, even when they
    may be infinite.  The key is to represent them each as automata.

    """

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
        if check_all_input_universal(self.fst):
            universal_states = set(P.stop)
        else:
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


def is_universal(dfa, state, alphabet):
    "[True/False] This `state` accepts the universal language (alphabet$^*$)."
    #
    # Rationale: a DFA accepts `alphabet`$^*$ iff all reachable states are
    # accepting and complete (i.e., has a transition for each symbol in
    # `alphabet`).
    #
    # Warning: If the reachable subset automaton is infinite, the search may
    # not terminate (as expected, NFA universality is PSPACE-complete in
    # general), but in many practical FSAs this halts quickly.
    #

    visited = set()
    worklist = deque()

    # DFA start state
    visited.add(state)
    worklist.append(state)

    while worklist:
        i = worklist.popleft()

        # All-final check in the DFA view
        if not dfa.is_final(i):
            return False

        # Build a symbol-to-destination mapping
        dest = dict(dfa.arcs(i))

        # Completeness on Σ
        for a in alphabet:
            # if we're missing an arc labeled `a` in state `i`, then state
            # `i` is not universal!  Moreover, `state` is not universal.
            if a not in dest:
                return False
            j = dest[a]
            if j not in visited:
                visited.add(j)
                worklist.append(j)

    return True


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
