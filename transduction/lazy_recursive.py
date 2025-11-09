from transduction.base import AbstractAlgorithm, PrecoverDecomp
from transduction.fst import EPSILON
from transduction.fsa import FSA
from arsenal.cache import memoize
from arsenal import colors


class BuggyLazyRecursive(AbstractAlgorithm):
    """This algorithm returns a valid, but sometimes suboptimal, `PrecoverDecomp`.

    UPDATE: See `transduction3.Fixed` for a version that correctly identifies
    the optimal `PrecoverDecomp`.

    However, it has some useful structure to it, as it is essentially a
    recursive implementation of the abstract algorithm.

    """

    def __init__(self, fst, empty_target = '', **kwargs):
        super().__init__(fst, **kwargs)
        self.empty_target = empty_target

    def initialize(self, target):
        pass   # does nothing.

    # Note: this version overrides the base because it is recursive;
    # TODO: make a base class for abstract *recursive* algorithms!
    def __call__(self, y):

        N = len(y)
        if N == 0:
            # Note: this base case is making a (reasonable) assumption that
            # `fst` is a total function.  We may want to consider supporting
            # partial functions, which will require revising this step.
            return PrecoverDecomp({self.empty_source}, set())

        prev = self(y[:-1])   # recurse on prefix
        curr = PrecoverDecomp(set(), set())

        # filter previous remainders
        for xs in prev.remainder:
            if self.discontinuity(xs, y):
                curr.remainder.add(xs)

        # filter previous calls output so that it satsifies the invariant
        worklist = []
        for xs in prev.quotient:
            if self.candidacy(xs, y):
                worklist.append(xs)

        # Another invariant: we never pop a xs more than once.  All
        # of a source string's prefixes will pop before it.
        t = 0
        while worklist:
            xs = worklist.pop()
            t += 1
            if t > self.max_steps:
                print(colors.light.red % '~~~~ stopped early ~~~~')
                break

            if self.continuity(xs, y):
                curr.quotient.add(xs)
                continue

            if self.discontinuity(xs, y):
                curr.remainder.add(xs)

            for next_xs in self.candidates(xs, y):
                worklist.append(next_xs)

        return curr

    def as_fsa(self, target):
        "Take the output and represent it as an FSA."
        decomp = self(target)
        return (FSA.from_strings(decomp.quotient) * FSA.universal(self.source_alphabet)
                + FSA.from_strings(decomp.remainder))

    def _as_fsa(self, target):
        "Equivalent to `as_fsa`, but less efficient."
        decomp = self(target)
        # The method below is better implemented by the else clause.
        states = {x[:t] for x in decomp.quotient | decomp.remainder
                  for t in range(len(x)+1)}
        m = FSA()
        m.add_start('')
        for x in states:
            if self.continuity(x, target):
                m.add_stop(x)
                for a in self.source_alphabet:
                    m.add(x, a, x)
                continue
            if self.discontinuity(x, target):
                m.add_stop(x)
            for xx in self.candidates(x, target):
                m.add(x, xx[-1], xx)
        return m

    def candidates(self, xs, target):
        for source_symbol in self.source_alphabet:
            next_xs = self.extend(xs, source_symbol)
            if self.candidacy(next_xs, target):
                yield next_xs

    def candidacy(self, xs, target):
        return any(
            (ys.startswith(target) or target.startswith(ys))
            for (s, ys) in self.frontier(xs)
        )

    def discontinuity(self, xs, target):   # pylint: disable=W0613
        #assert not self.continuity(xs, y)
        return any((s in self.fst.F) for (s, ys) in self.frontier(xs)
                   if ys.startswith(target))

    # XXX: technically, this state depends on the `target` as it was used for filtering.
    @memoize
    def frontier(self, xs):
        """This method is primarily for debugging.  It rReturns the state of
        `xs` in the powerset construction augmented where each state
        is paired with a target-side string.

        Note to self: An FST is a weighted automaton where the weights represent
        fragments of the target language. Recall, that in /weighted/ automata
        the powerset construction is augmented with the weight of the state.

        """

        if len(xs) == 0:
            return self._epsilon_closure_frontier({(s, self.empty_target) for s in self.fst.I})
        else:
            return self.next_frontier(self.frontier(xs[:-1]), xs[-1])

    # XXX: these are the arcs of the "lazy frontier machine"
    def next_frontier(self, frontier, source_symbol):
        "Transitions in the augmented-powerstate construction."
        assert source_symbol != EPSILON
        next_frontier = set()
        for s, ys in frontier:
            for a, b, j in self.fst.arcs(s):
                if a == source_symbol:
                    next_frontier.add((j, self.extend(ys, b)))
        return self._epsilon_closure_frontier(next_frontier)

    def _epsilon_closure_frontier(self, frontier):
        "Extend `frontier` to include everything reachable by source-side epsilon transitions."
        worklist = set(frontier)
        next_frontier = set()
        while worklist:
            (s, ys) = worklist.pop()
            if (s, ys) in next_frontier: continue
            next_frontier.add((s, ys))
            for tmp, b, next_state in self.fst.arcs(s):
                if tmp == EPSILON:
                    worklist.add((next_state, self.extend(ys, b)))
        return next_frontier

    def continuity(self, xs, target):
        """
        Is `xs` a member of y's quotient? (not necessarily miminal)

        Warning: this test is not exact, and it leads to a suboptimal quotient.
        """
        return self.is_universal(frozenset(s for (s, ys) in self.frontier(xs)
                                           if ys.startswith(target)))

    @memoize
    def is_universal(self, S):
        """
        Is this powerstate universal?  (Note: this is not a frontier.)
        """
        S_fst = self.fst.spawn(keep_arcs=True, keep_stop=True)
        for q in S: S_fst.add_I(q)
        q_fsa = S_fst.project(0)
        # use the unweighted FSA library instead
        q_dfa = q_fsa.min()
        if len(q_dfa.states) != 1:
            return False
        [i] = q_dfa.states
        for a in self.source_alphabet:
            if set(q_dfa.arcs(i, a)) != {i}:
                return False
        return True


class LazyRecursive(BuggyLazyRecursive):
    """
    Lazy, recursive DFA-based algorithm.
    """

    def continuity(self, xs, target):
        "Is `xs` a cylinder of y's precover?"
        #
        # Rationale: a DFA accepts `alphabet`$^*$ iff all reachable states are
        # accepting and complete (i.e., has a transition for each symbol in
        # `alphabet`).
        #
        # Warning: If the reachable subset automaton is infinite, the search may
        # not terminate (as expected, NFA universality is PSPACE-complete in
        # general), but in many practical FSAs this halts quickly.
        #
        alphabet = self.source_alphabet

        def refine(frontier):
            # clip the target side side to `y` in order to mimick the states of
            # the composition machine that we used in the new lazy, nonrecursive
            # algorithm.
            N = len(target)
            return frozenset({
                (i, ys[:N]) for i, ys in frontier
                if ys[:N].startswith(target)
            })

        # XXX: same as `candidates` method, except that it generates a
        # source_symbol---extended_string pair.
        def arcs(xs):
            for source_symbol in self.source_alphabet:
                next_xs = self.extend(xs, source_symbol)
                if self.candidacy(next_xs, target):
                    yield source_symbol, next_xs

        def is_final(xs):
            return self.discontinuity(xs, target)

        worklist = []

        # DFA start state
        worklist.append(xs)

        visited = {refine(self.frontier(xs))}

        while worklist:
            i = worklist.pop()

            # All-final check in the DFA view
            if not is_final(i):
                return False

            # Build a symbol-to-destination mapping
            dest = dict(arcs(i))

            # Completeness on Σ
            for a in alphabet:
                # if we're missing an arc labeled `a` in state `i`, then state
                # `i` is not universal!  Moreover, `state` is not universal.
                if a not in dest:
                    return False
                j = dest[a]
                jj = refine(self.frontier(j))
                if jj not in visited:
                    visited.add(jj)
                    worklist.append(j)

        return True
