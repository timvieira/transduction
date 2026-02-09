import heapq
from transduction.base import IncrementalDecomposition
from transduction.fst import EPSILON
from transduction.fsa import FSA
from transduction.universality import check_all_input_universal, compute_ip_universal_states
from collections import deque


class BFSHeuristic:
    """Default BFS heuristic state: orders by depth (lower = explored first)."""
    def __init__(self, depth=0):
        self.depth = depth
    def __rshift__(self, _):
        return BFSHeuristic(self.depth + 1)
    def __lt__(self, other):
        return self.depth < other.depth


class PrioritizedLazyIncremental(IncrementalDecomposition):
    """
    Variant of :class:`LazyIncremental` with priority-based exploration and
    optional pruning.

    **Finite-language only.** Same caveats as ``LazyIncremental`` — may diverge
    on FSTs with infinite quotient/remainder languages.

    Parameters
    ----------
    heuristic : object or None
        A comparable state object supporting ``>>`` and ``<``.
        ``h >> symbol`` returns a new heuristic state; ``h1 < h2`` orders
        the min-heap (smaller = explored first).  When ``None`` (default),
        uses :class:`BFSHeuristic` (orders by depth).
    max_steps : int or float
        Maximum number of heap pops before stopping early.

    Usage::

        state = PrioritizedLazyIncremental(fst, '', heuristic=my_h)
        state = state >> 'a'
        state.quotient    # FSA
        state.remainder   # FSA
    """

    def __init__(self, fst, target='', parent=None, *, empty_source='',
                 empty_target='', extend=lambda x, y: x + y,
                 max_steps=float('inf'), heuristic=BFSHeuristic()):
        self.fst = fst
        self.target = target

        if parent is None:
            self.source_alphabet = fst.A - {EPSILON}
            self.target_alphabet = fst.B - {EPSILON}
            self.empty_source = empty_source
            self.empty_target = empty_target
            self.extend = extend
            self.max_steps = max_steps
            self.heuristic = heuristic
            self._frontier_cache = {}
            self._all_input_universal = check_all_input_universal(fst)
            self._ip_universal_states = (
                frozenset() if self._all_input_universal
                else compute_ip_universal_states(fst)
            )
            assert len(target) == 0, 'Use __call__ or >> to advance from an empty target'
            self.parent = None
        else:
            self.source_alphabet = parent.source_alphabet
            self.target_alphabet = parent.target_alphabet
            self.empty_source = parent.empty_source
            self.empty_target = parent.empty_target
            self.extend = parent.extend
            self.max_steps = parent.max_steps
            self.heuristic = parent.heuristic
            self._frontier_cache = parent._frontier_cache
            self._all_input_universal = parent._all_input_universal
            self._ip_universal_states = parent._ip_universal_states
            self.parent = parent
            assert self.parent.target == target[:-1]

        self._compute()

    def _compute(self):
        self._quotient_hstates = {}
        self._remainder_hstates = {}

        # heap entries: (h_state, tiebreak, xs)
        heap = []
        counter = 0

        N = len(self.target)
        if N == 0:
            heapq.heappush(heap, (self.heuristic, counter, self.empty_source))
            counter += 1
        else:
            # filter previous remainders
            for xs, h_state in self.parent._remainder_hstates.items():
                if self.discontinuity(xs, self.target):
                    self._remainder_hstates[xs] = h_state

            # filter previous quotient so that it satisfies the invariant
            for xs, h_state in self.parent._quotient_hstates.items():
                if self.candidacy(xs, self.target):
                    heapq.heappush(heap, (h_state, counter, xs))
                    counter += 1

        steps = 0
        while heap and steps < self.max_steps:
            h_state, _, xs = heapq.heappop(heap)
            steps += 1

            if self.continuity(xs, self.target):
                self._quotient_hstates[xs] = h_state
                continue

            if self.discontinuity(xs, self.target):
                self._remainder_hstates[xs] = h_state

            for source_symbol in self.source_alphabet:
                next_xs = self.extend(xs, source_symbol)
                if self.candidacy(next_xs, self.target):
                    child_h = h_state >> source_symbol
                    heapq.heappush(heap, (child_h, counter, next_xs))
                    counter += 1

    @property
    def quotient(self):
        return FSA.from_strings(self._quotient_hstates)

    @property
    def remainder(self):
        return FSA.from_strings(self._remainder_hstates)

    def __rshift__(self, y):
        return PrioritizedLazyIncremental(self.fst, self.target + y, parent=self)

    def candidacy(self, xs, target):
        return any(
            (ys.startswith(target) or target.startswith(ys))
            for (_, ys) in self.frontier(xs)
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
            for b, next_state in self.fst.arcs(s, EPSILON):
                worklist.add((next_state, self.extend(ys, b)))
        return next_frontier

    def continuity(self, xs, target):
        "Is `xs` a cylinder of y's precover?"

        # Fast path: if the entire FST's input projection is universal,
        # every final frontier is universal.
        if self._all_input_universal:
            return self.discontinuity(xs, target)

        # ip-universal witness: if the frontier contains (q, ys) where q is
        # ip-universal and ys already covers the target, then all extensions
        # of xs will also have an ip-universal witness → universal.
        if self._ip_universal_states:
            for q, ys in self.frontier(xs):
                if q in self._ip_universal_states and ys.startswith(target):
                    return True

        alphabet = self.source_alphabet

        def refine(frontier):
            N = len(target)
            return frozenset({
                (i, ys[:N]) for i, ys in frontier
                if ys[:min(N, len(ys))] == target[:min(N, len(ys))]
            })

        def arcs(xs):
            for source_symbol in self.source_alphabet:
                next_xs = self.extend(xs, source_symbol)
                if self.candidacy(next_xs, target):
                    yield source_symbol, next_xs

        def is_final(xs):
            return self.discontinuity(xs, target)

        worklist = deque()
        worklist.append(xs)
        visited = {refine(self.frontier(xs))}

        while worklist:
            i = worklist.popleft()
            if not is_final(i):
                return False
            dest = dict(arcs(i))
            for a in alphabet:
                if a not in dest:
                    return False
                j = dest[a]
                jj = refine(self.frontier(j))
                if jj not in visited:
                    visited.add(jj)
                    worklist.append(j)

        return True
