from transduction.base import AbstractAlgorithm, PrecoverDecomp
from transduction.fst import FST, EPSILON
from transduction.fsa import FSA
from transduction.util import display_table
from transduction.lazy import Lazy

from arsenal import colors
from arsenal.cache import memoize
from functools import cached_property
from collections import deque


class LazyPrecoverNFAWithTruncationMarker(Lazy):
    """
    Alternative implmementation of LazyPrecoverNFA, which augments the states
    with an additional bit, indicating if their output context has been
    truncated (i.e., if there are more symbols in the output buffer than their
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
#        if self.target[:m] != ys[:m]:      # target and ys are incompatible
#            return
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
            yield (i, '', False)

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
#        if self.target[:m] != ys[:m]:      # target and ys are not prefixes of one another.
#            return
#        if ys.startswith(self.target):
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


class Precover:
    """
    Representation of the precover of target string `target` in the FST `fst`.

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
#        return LazyPrecoverNFA(self.fst, self.target).det().materialize()
#        return LazyPrecoverNFAWithTruncationMarker(self.fst, self.target).det().materialize()

    @cached_property
    def det_universal_states(self):
        "set of universal states in `self.det`"
        P = self.det
        return {i for i in P.stop if is_universal(P, i, self.source_alphabet)}

    @cached_property
    def min(self):
        "Minimal DFA representing the complete precover."
        return self.fsa.min()

    @cached_property
    def fsa(self):
        "FSA representing the complete precover."
        # this is a copy machine for target \targetAlphabet^*
        #want = (self.fst @ self.target_prefixes).project(0)
        have = LazyPrecoverNFA(self.fst, self.target).materialize()
        #assert have.equal(want), [want.min(), have.min()]
        #return want
        return have

    @cached_property
    def target_prefixes(self):
        "An automaton that denotes the `target` string's cylinder set."
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
        "Produce a quotient--remainder pair each represented as automata."
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
        "Analyze the decompositions Q and R: is it valid? optimal?  Note that these tests only terminal each Q and R are finite sets."
        if isinstance(Q, FSA): Q = Q.language(np.inf)
        if isinstance(R, FSA): R = R.language(np.inf)
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
        "A simple visualization of the decomposition for IPython notebooks."
        Q,R = self.decomposition
        if minimize: Q, R = Q.min(), R.min()
        if trim:     Q, R = Q.trim(), R.trim()
        display_table([[Q, R]], headings=['quotient', 'remainder'])

    def graphviz(self, trim=False):
        "Stylized graphviz representation of the precover DFA where colors denotes properties of the state (green: quotient, magenta: remainder, white: useless)"
        dfa = self.det
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


def force_start(fsa, start_state):
    assert start_state in fsa.states
    new = FSA()
    new.add_start(start_state)
    for i, a, j in fsa.arcs():
        new.add(i, a, j)
    for i in fsa.stop:
        new.add_stop(i)
    return new


#def is_universal(fsa, q, alphabet):
#    # universality test; best used on a DFA
#    m = force_start(fsa, q).min()
#    return len(m.states) == 1 and all(set(m.arcs(i, a)) == {i} for i in m.states for a in alphabet)


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
