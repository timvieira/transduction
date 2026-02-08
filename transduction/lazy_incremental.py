from transduction.base import IncrementalDecomposition
from transduction.fst import EPSILON
from transduction.fsa import FSA
from arsenal import colors
from collections import deque


class LazyIncremental(IncrementalDecomposition):
    """
    Lazy, recursive DFA-based algorithm using the incremental ``>>`` interface.

    **Finite-language only.** This algorithm enumerates source *strings* (not
    automaton states) and its universality check may diverge on FSTs whose
    quotient or remainder languages are infinite.  It is therefore excluded
    from ``test_general.py`` (which exercises FSTs with infinite languages)
    and should only be tested on finite-language FSTs.

    Usage::

        state = LazyIncremental(fst, '')
        state = state >> 'a'
        state.quotient    # FSA
        state.remainder   # FSA
    """

    def __init__(self, fst, target='', parent=None, *, empty_source='', empty_target='', extend=lambda x, y: x + y, max_steps=float('inf')):
        self.fst = fst
        self.target = target
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        if parent is None:
            # Initial state: store config and create shared frontier cache
            self.empty_source = empty_source
            self.empty_target = empty_target
            self.extend = extend
            self.max_steps = max_steps
            self._frontier_cache = {}
            # Bootstrap: build intermediate states for non-empty initial target
            if len(target) > 0:
                state = LazyIncremental(fst, '', empty_source=empty_source,
                                        empty_target=empty_target, extend=extend,
                                        max_steps=max_steps)
                for ch in target[:-1]:
                    state = state >> ch
                self.parent = state
                self._frontier_cache = state._frontier_cache
            else:
                self.parent = None
        else:
            # Incremental step: inherit config and shared cache from parent
            self.empty_source = parent.empty_source
            self.empty_target = parent.empty_target
            self.extend = parent.extend
            self.max_steps = parent.max_steps
            self._frontier_cache = parent._frontier_cache
            self.parent = parent

        self._compute()

    def _compute(self):
        self._quotient_set = set()
        self._remainder_set = set()

        worklist = deque()

        N = len(self.target)
        if N == 0:
            worklist.append(self.empty_source)
        else:
            # filter previous remainders
            for xs in self.parent._remainder_set:
                if self.discontinuity(xs, self.target):
                    self._remainder_set.add(xs)

            # filter previous quotient so that it satisfies the invariant
            for xs in self.parent._quotient_set:
                if self.candidacy(xs, self.target):
                    worklist.append(xs)

        t = 0
        while worklist:
            xs = worklist.popleft()
            t += 1
            if t > self.max_steps:
                print(colors.light.red % '~~~~ stopped early ~~~~')
                break

            if self.continuity(xs, self.target):
                self._quotient_set.add(xs)
                continue

            if self.discontinuity(xs, self.target):
                self._remainder_set.add(xs)

            for next_xs in self.candidates(xs, self.target):
                worklist.append(next_xs)

    @property
    def quotient(self):
        return FSA.from_strings(self._quotient_set)

    @property
    def remainder(self):
        return FSA.from_strings(self._remainder_set)

    def __rshift__(self, y):
        return LazyIncremental(self.fst, self.target + y, parent=self)

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

    def discontinuity(self, xs, target):
        return any((s in self.fst.stop) for (s, ys) in self.frontier(xs)
                   if ys.startswith(target))

    def frontier(self, xs):
        """Returns the state of `xs` in the powerset construction where each
        state is paired with a target-side string."""
        val = self._frontier_cache.get(xs)
        if val is None:
            val = self._compute_frontier(xs)
            self._frontier_cache[xs] = val
        return val

    def _compute_frontier(self, xs):
        if len(xs) == 0:
            return self._epsilon_closure_frontier({(s, self.empty_target) for s in self.fst.start})
        else:
            return self.next_frontier(self.frontier(xs[:-1]), xs[-1])

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
            # clip the target side to `y` in order to mimic the states of
            # the composition machine that we used in the new lazy, nonrecursive
            # algorithm.
            N = len(target)
            return frozenset({
                (i, ys[:N]) for i, ys in frontier
                if ys[:min(N, len(ys))] == target[:min(N, len(ys))]
            })

        # Note: same as `candidates` method, except that it generates a
        # source_symbol---extended_string pair.
        def arcs(xs):
            for source_symbol in self.source_alphabet:
                next_xs = self.extend(xs, source_symbol)
                if self.candidacy(next_xs, target):
                    yield source_symbol, next_xs

        def is_final(xs):
            return self.discontinuity(xs, target)

        worklist = deque()

        # DFA start state
        worklist.append(xs)

        visited = {refine(self.frontier(xs))}

        while worklist:
            i = worklist.popleft()

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
