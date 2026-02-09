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


class FrontierGraph:
    """Interned frontier graph: caches powerset-construction nodes and transitions.

    Each node is a canonical frozenset of (fst_state, target_buffer) pairs.
    Transitions are cached per (node, symbol) so that duplicate source strings
    reaching the same powerset state share work.
    """

    def __init__(self, fst, empty_target, extend):
        self.fst = fst
        self.extend = extend
        self._intern = {}           # frozenset -> frozenset (canonical identity)
        self._transitions = {}      # (node_id, symbol) -> node

        self.initial = self._intern_node(
            self._epsilon_closure({(s, empty_target) for s in fst.start})
        )

    def _intern_node(self, frontier_set):
        key = frozenset(frontier_set)
        return self._intern.setdefault(key, key)

    def _epsilon_closure(self, frontier):
        """Extend frontier to include everything reachable by source-side epsilon transitions."""
        worklist = set(frontier)
        result = set()
        while worklist:
            pair = worklist.pop()
            if pair in result:
                continue
            result.add(pair)
            s, ys = pair
            for b, next_state in self.fst.arcs(s, EPSILON):
                worklist.add((next_state, self.extend(ys, b)))
        return result

    def step(self, node, symbol):
        """Transition from node by source symbol, returning the interned successor."""
        key = (id(node), symbol)
        cached = self._transitions.get(key)
        if cached is not None:
            return cached
        next_set = set()
        for s, ys in node:
            for b, j in self.fst.arcs(s, symbol):
                next_set.add((j, self.extend(ys, b)))
        result = self._intern_node(self._epsilon_closure(next_set))
        self._transitions[key] = result
        return result


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
            self._graph = FrontierGraph(fst, empty_target, extend)
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
            self._graph = parent._graph
            self._all_input_universal = parent._all_input_universal
            self._ip_universal_states = parent._ip_universal_states
            self.parent = parent
            assert self.parent.target == target[:-1]

        self._compute()

    def _compute(self):
        self._quotient_hstates = {}
        self._remainder_hstates = {}
        self._quotient_nodes = {}
        self._remainder_nodes = {}

        graph = self._graph
        target = self.target

        # Per-node caches for this _compute call
        _candidacy_cache = {}
        _discontinuity_cache = {}

        def _candidacy(node):
            cached = _candidacy_cache.get(id(node))
            if cached is not None:
                return cached
            result = any(
                (ys.startswith(target) or target.startswith(ys))
                for (_, ys) in node
            )
            _candidacy_cache[id(node)] = result
            return result

        def _discontinuity(node):
            cached = _discontinuity_cache.get(id(node))
            if cached is not None:
                return cached
            result = any(
                (s in self.fst.stop) for (s, ys) in node
                if ys.startswith(target)
            )
            _discontinuity_cache[id(node)] = result
            return result

        # heap entries: (h_state, tiebreak, xs, node)
        heap = []
        counter = 0

        N = len(target)
        if N == 0:
            heapq.heappush(heap, (self.heuristic, counter, self.empty_source, graph.initial))
            counter += 1
        else:
            # filter previous remainders
            for xs, h_state in self.parent._remainder_hstates.items():
                node = self.parent._remainder_nodes[xs]
                if _discontinuity(node):
                    self._remainder_hstates[xs] = h_state
                    self._remainder_nodes[xs] = node

            # filter previous quotient so that it satisfies the invariant
            for xs, h_state in self.parent._quotient_hstates.items():
                node = self.parent._quotient_nodes[xs]
                if _candidacy(node):
                    heapq.heappush(heap, (h_state, counter, xs, node))
                    counter += 1

        steps = 0
        while heap and steps < self.max_steps:
            h_state, _, xs, node = heapq.heappop(heap)
            steps += 1

            if self._check_continuity(node, target, graph, _candidacy, _discontinuity):
                self._quotient_hstates[xs] = h_state
                self._quotient_nodes[xs] = node
                continue

            if _discontinuity(node):
                self._remainder_hstates[xs] = h_state
                self._remainder_nodes[xs] = node

            for source_symbol in self.source_alphabet:
                next_xs = self.extend(xs, source_symbol)
                next_node = graph.step(node, source_symbol)
                if _candidacy(next_node):
                    child_h = h_state >> source_symbol
                    heapq.heappush(heap, (child_h, counter, next_xs, next_node))
                    counter += 1

    def _check_continuity(self, node, target, graph, _candidacy, _discontinuity):
        """Node-based BFS universality check."""

        # Fast path: if the entire FST's input projection is universal,
        # every final frontier is universal.
        if self._all_input_universal:
            return _discontinuity(node)

        # ip-universal witness: if the frontier contains (q, ys) where q is
        # ip-universal and ys already covers the target, then all extensions
        # of xs will also have an ip-universal witness → universal.
        if self._ip_universal_states:
            for q, ys in node:
                if q in self._ip_universal_states and ys.startswith(target):
                    return True

        N = len(target)

        def refine(n):
            return frozenset({
                (i, ys[:N]) for i, ys in n
                if ys[:min(N, len(ys))] == target[:min(N, len(ys))]
            })

        worklist = deque()
        worklist.append(node)
        visited = {refine(node)}

        while worklist:
            cur = worklist.popleft()
            if not _discontinuity(cur):
                return False
            for a in self.source_alphabet:
                next_node = graph.step(cur, a)
                if not _candidacy(next_node):
                    return False
                jj = refine(next_node)
                if jj not in visited:
                    visited.add(jj)
                    worklist.append(next_node)

        return True

    @property
    def quotient(self):
        return FSA.from_strings(self._quotient_hstates)

    @property
    def remainder(self):
        return FSA.from_strings(self._remainder_hstates)

    def __rshift__(self, y):
        return PrioritizedLazyIncremental(self.fst, self.target + y, parent=self)
