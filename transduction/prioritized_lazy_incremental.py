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


class SourceTrie:
    """Interned trie of source paths.  O(1) extend, O(1) hashing.

    Each source path is represented as an integer node ID.  ``extend(node, symbol)``
    returns the child ID, allocating it on first access.  Strings are reconstructed
    only when needed (e.g., to build the output FSA), not in the inner loop.

    Shared across ``>>`` steps alongside ``FrontierGraph``.
    """

    def __init__(self, empty_source, extend):
        self._extend = extend
        self._empty_source = empty_source
        self._children = {}    # (node_id, symbol) -> child_id
        self._parent = [None]  # node_id -> (parent_id, symbol) | None
        self.initial = 0       # root = empty source

    def extend(self, node, symbol):
        """Return the child of *node* via *symbol*, creating it if new.  O(1)."""
        key = (node, symbol)
        child = self._children.get(key)
        if child is not None:
            return child
        child = len(self._parent)
        self._parent.append((node, symbol))
        self._children[key] = child
        return child

    def to_string(self, node):
        """Reconstruct the source string for *node* by walking the parent chain."""
        symbols = []
        cur = node
        while self._parent[cur] is not None:
            cur, sym = self._parent[cur]
            symbols.append(sym)
        symbols.reverse()
        result = self._empty_source
        for sym in symbols:
            result = self._extend(result, sym)
        return result


class FrontierGraph:
    """Target-independent transition graph over interned powerset-construction nodes.

    Each node is a canonical frozenset of ``(fst_state, target_buffer)`` pairs —
    the "augmented powerset state" that pairs each reachable FST state with the
    target-side output accumulated so far.  Interning guarantees that two source
    strings reaching the same powerset state share a single node object, so
    transition lookups and property checks (keyed by ``id(node)``) are never
    duplicated.

    This graph is shared across ``>>`` steps: the ``PrioritizedLazyIncremental``
    parent and all its children reference the same ``FrontierGraph``, which grows
    monotonically as new nodes are discovered.
    """

    def __init__(self, fst, empty_target, extend):
        self.fst = fst
        self.extend = extend
        # Interning table: maps each frozenset to a single canonical object.
        # Using id(node) as a cache key is safe because interned nodes are never
        # garbage-collected (held by _intern) and distinct contents always map to
        # distinct objects.
        self._intern = {}           # frozenset -> frozenset (canonical identity)
        self._transitions = {}      # (node_id, symbol) -> node
        self._arcs = {}             # node_id -> [(symbol, next_node), ...]

        self.initial = self._intern_node(
            self._epsilon_closure({(s, empty_target) for s in fst.start})
        )

    def _intern_node(self, frontier_set):
        """Return the canonical object for this frontier set, interning if new."""
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
        """Transition from *node* by a single source *symbol*, returning the interned successor."""
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

    def arcs(self, node):
        """Return ``[(symbol, next_node), ...]`` for source symbols with non-empty transitions.

        Only symbols that actually appear on arcs from states in *node* are included,
        avoiding unnecessary work for large alphabets with sparse transitions.
        """
        key = id(node)
        cached = self._arcs.get(key)
        if cached is not None:
            return cached
        symbols = set()
        for s, _ys in node:
            for a, _b, j in self.fst.arcs(s):
                if a != EPSILON:
                    symbols.add(a)
        result = [(a, self.step(node, a)) for a in symbols]
        self._arcs[key] = result
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
            self._trie = SourceTrie(empty_source, extend)
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
            self._trie = parent._trie
            self._all_input_universal = parent._all_input_universal
            self._ip_universal_states = parent._ip_universal_states
            self.parent = parent
            assert self.parent.target == target[:-1]

        self._compute()

    def _is_live(self, node):
        """Can any path through *node* still produce output matching the target?

        A node is live if some ``(state, ys)`` pair has ``ys`` compatible with
        ``target`` — either ``ys`` is a prefix of ``target`` (output so far is
        consistent) or ``target`` is a prefix of ``ys`` (output already covers
        the target and may extend it).  Dead nodes are pruned from the search.
        """
        cached = self._is_live_cache.get(id(node))
        if cached is not None:
            return cached
        target = self.target
        result = any(
            (ys.startswith(target) or target.startswith(ys))
            for (_, ys) in node
        )
        self._is_live_cache[id(node)] = result
        return result

    def _is_final(self, node):
        """Does *node* contain an accepting FST state whose output covers the target?

        True when some ``(state, ys)`` has ``state`` final and ``ys`` starts with
        ``target``, witnessing that the source string maps to a target-side string
        beginning with the current target prefix.
        """
        cached = self._is_final_cache.get(id(node))
        if cached is not None:
            return cached
        target = self.target
        result = any(
            (s in self.fst.stop) for (s, ys) in node
            if ys.startswith(target)
        )
        self._is_final_cache[id(node)] = result
        return result

    def _compute(self):
        """Heap-driven exploration to partition source strings into quotient and remainder.

        For the root (empty target), seeds the heap with the graph's initial node.
        For children (target extended by one symbol), re-filters the parent's
        quotient and remainder against the new target, then continues exploring.
        """
        self._quotient = {}     # xs -> (h_state, node)
        self._remainder = {}    # xs -> (h_state, node)
        # Property caches are target-dependent, so reset each _compute call.
        self._is_live_cache = {}
        self._is_final_cache = {}

        graph = self._graph
        trie = self._trie

        # heap entries: (h_state, tiebreak, xs, node)
        # xs is a trie node ID (int), not a string.
        heap = []
        counter = 0

        if len(self.target) == 0:
            heapq.heappush(heap, (self.heuristic, counter, trie.initial, graph.initial))
            counter += 1
        else:
            # Remainder strings that are still final under the new target stay in remainder.
            for xs, (h_state, node) in self.parent._remainder.items():
                if self._is_final(node):
                    self._remainder[xs] = (h_state, node)

            # Quotient strings that are still live go back on the heap for re-evaluation,
            # since universality may no longer hold under the extended target.
            for xs, (h_state, node) in self.parent._quotient.items():
                if self._is_live(node):
                    heapq.heappush(heap, (h_state, counter, xs, node))
                    counter += 1

        steps = 0
        while heap and steps < self.max_steps:
            h_state, _, xs, node = heapq.heappop(heap)
            steps += 1

            # If this node is universal, xs belongs in the quotient — no need to
            # explore its children.
            if self._is_universal(node):
                self._quotient[xs] = (h_state, node)
                continue

            # Not universal, but if it's final, xs belongs in the remainder.
            if self._is_final(node):
                self._remainder[xs] = (h_state, node)

            # Expand children: extend xs by each source symbol with a live successor.
            for source_symbol, next_node in graph.arcs(node):
                if self._is_live(next_node):
                    next_xs = trie.extend(xs, source_symbol)
                    child_h = h_state >> source_symbol
                    heapq.heappush(heap, (child_h, counter, next_xs, next_node))
                    counter += 1

    def _is_universal(self, node):
        """BFS check: does every extension of *node* remain final?

        A node is universal if the sub-DFA rooted at it (under the augmented
        powerset construction) accepts ``source_alphabet*`` — i.e., every
        reachable node is final and complete.  Universal nodes are "cylinders"
        of the precover: the source string and all its extensions map to the
        target, so they belong in the quotient.
        """

        # Fast path: if the entire FST's input projection is universal,
        # every final node is universal.
        if self._all_input_universal:
            return self._is_final(node)

        # ip-universal witness: if the node contains (q, ys) where q is
        # ip-universal and ys already covers the target, then all extensions
        # will also have an ip-universal witness.
        target = self.target
        if self._ip_universal_states:
            for q, ys in node:
                if q in self._ip_universal_states and ys.startswith(target):
                    return True

        N = len(target)
        graph = self._graph

        def refine(n):
            """Project node to target-relevant components for cycle detection.

            Clips target buffers to length N and drops pairs incompatible with
            the target prefix.  Two nodes with the same refined image behave
            identically for universality, so we use this as the visited-set key.
            """
            return frozenset({
                (i, ys[:N]) for i, ys in n
                if ys[:min(N, len(ys))] == target[:min(N, len(ys))]
            })

        worklist = deque()
        worklist.append(node)
        visited = {refine(node)}

        while worklist:
            cur = worklist.popleft()
            if not self._is_final(cur):
                return False
            # Check completeness: every source symbol must have a transition.
            succs = dict(graph.arcs(cur))
            if len(succs) < len(self.source_alphabet):
                return False
            for a in self.source_alphabet:
                next_node = succs[a]
                if not self._is_live(next_node):
                    return False
                jj = refine(next_node)
                if jj not in visited:
                    visited.add(jj)
                    worklist.append(next_node)

        return True

    @property
    def quotient(self):
        return FSA.from_strings(self._trie.to_string(xs) for xs in self._quotient)

    @property
    def remainder(self):
        return FSA.from_strings(self._trie.to_string(xs) for xs in self._remainder)

    def __rshift__(self, y):
        return PrioritizedLazyIncremental(self.fst, self.target + y, parent=self)
