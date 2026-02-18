import heapq
from transduction.base import IncrementalDecomposition
from transduction.fst import EPSILON
from transduction.fsa import FSA
from transduction.universality import check_all_input_universal, compute_ip_universal_states
from collections import defaultdict, deque


class BFSHeuristic:
    """Default BFS heuristic state: orders by depth (lower = explored first)."""
    def __init__(self, depth=0):
        self.depth = depth
    def __rshift__(self, _):
        return BFSHeuristic(self.depth + 1)
    def __lt__(self, other):
        return self.depth < other.depth


class Trie:
    """Interned trie of symbol paths.  O(1) extend, O(1) depth lookup.

    Each path is represented as an integer node ID.  ``extend(node, symbol)``
    returns the child ID, allocating it on first access.  Tuples are reconstructed
    only when needed (e.g., to build the output FSA), not in the inner loop.

    Shared across ``>>`` steps alongside ``FrontierGraph``.
    """

    def __init__(self):
        self._children = {}    # (node_id, symbol) -> child_id
        self._parent = [None]  # node_id -> (parent_id, symbol) | None
        self._depth = [0]      # node_id -> depth
        self.initial = 0       # root = empty path

    def extend(self, node, symbol):
        """Return the child of *node* via *symbol*, creating it if new.  O(1)."""
        key = (node, symbol)
        child = self._children.get(key)
        if child is not None:
            return child
        child = len(self._parent)
        self._parent.append((node, symbol))
        self._children[key] = child
        self._depth.append(self._depth[node] + 1)
        return child

    def depth(self, node):
        """Return the depth (path length) of *node*.  O(1)."""
        return self._depth[node]

    def ancestors(self, node):
        """Return the set of all ancestor node IDs, including *node* itself."""
        result = set()
        cur = node
        while True:
            result.add(cur)
            if self._parent[cur] is None:
                break
            cur = self._parent[cur][0]
        return result

    def to_tuple(self, node):
        """Reconstruct the tuple for *node* by walking the parent chain."""
        symbols = []
        cur = node
        while self._parent[cur] is not None:
            cur, sym = self._parent[cur]
            symbols.append(sym)
        symbols.reverse()
        return tuple(symbols)


class FrontierGraph:
    """Target-independent transition graph over interned powerset-construction nodes.

    Each node is a canonical frozenset of ``(fst_state, ys_trie_id)`` pairs —
    the "augmented powerset state" that pairs each reachable FST state with the
    interned target-side output accumulated so far.  Interning guarantees that two
    source strings reaching the same powerset state share a single node object, so
    transition lookups and property checks (keyed by ``id(node)``) are never
    duplicated.

    This graph is shared across ``>>`` steps: the ``PrioritizedLazyIncremental``
    parent and all its children reference the same ``FrontierGraph``, which grows
    monotonically as new nodes are discovered.
    """

    def __init__(self, fst, target_trie):
        self.fst = fst
        self._target_trie = target_trie
        # Interning table: maps each frozenset to a single canonical object.
        # Using id(node) as a cache key is safe because interned nodes are never
        # garbage-collected (held by _intern) and distinct contents always map to
        # distinct objects.
        self._intern = {}           # frozenset -> frozenset (canonical identity)
        self._transitions = {}      # (node_id, symbol) -> node
        self._arcs = {}             # node_id -> [(symbol, next_node), ...]

        self.initial = self._intern_node(
            self._epsilon_closure({(s, target_trie.initial) for s in fst.start})
        )

    def _intern_node(self, frontier_set):
        """Return the canonical object for this frontier set, interning if new."""
        key = frozenset(frontier_set)
        return self._intern.setdefault(key, key)

    def _epsilon_closure(self, frontier):
        """Extend frontier to include everything reachable by source-side epsilon transitions."""
        worklist = set(frontier)
        result = set()
        trie = self._target_trie
        while worklist:
            pair = worklist.pop()
            if pair in result:
                continue
            result.add(pair)
            s, ys = pair
            for b, next_state in self.fst.arcs(s, EPSILON):
                new_ys = ys if b == EPSILON else trie.extend(ys, b)
                worklist.add((next_state, new_ys))
        return result

    def step(self, node, symbol):
        """Transition from *node* by a single source *symbol*, returning the interned successor."""
        key = (id(node), symbol)
        cached = self._transitions.get(key)
        if cached is not None:
            return cached
        trie = self._target_trie
        next_set = set()
        for s, ys in node:
            for b, j in self.fst.arcs(s, symbol):
                new_ys = ys if b == EPSILON else trie.extend(ys, b)
                next_set.add((j, new_ys))
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
            symbols.update(self.fst.delta[s].keys())
        symbols.discard(EPSILON)
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

        If the heuristic state exposes a ``.logp`` attribute (cumulative
        log-probability), the ``logp_gap`` stopping criterion can be used.
    logp_gap : float or None
        Stop exploring when the best remaining heap entry's ``logp`` falls
        more than ``logp_gap`` below the best quotient entry's ``logp``.
        Requires the heuristic to expose a ``.logp`` attribute.
        ``None`` (default) disables gap-based stopping.
    max_steps : int or float
        Hard safety limit on heap pops.
    max_beam : int or float
        Hard safety limit on quotient/remainder entries propagated from
        the parent into each ``>>`` step.

    Usage::

        state = PrioritizedLazyIncremental(fst, heuristic=my_h, logp_gap=10)
        state = state >> 'a'
        state.quotient    # FSA
        state.remainder   # FSA
    """

    def __init__(self, fst, target=None, parent=None, *,
                 logp_gap=None,
                 max_steps=float('inf'), max_beam=float('inf'),
                 heuristic=BFSHeuristic()):
        self.fst = fst

        if parent is None:
            self.source_alphabet = fst.A - {EPSILON}
            self.target_alphabet = fst.B - {EPSILON}
            self.logp_gap = logp_gap
            self.max_steps = max_steps
            self.max_beam = max_beam
            self.heuristic = heuristic
            self._target_trie = Trie()
            self._source_trie = Trie()
            self._graph = FrontierGraph(fst, self._target_trie)
            self._all_input_universal = check_all_input_universal(fst)
            self._ip_universal_states = (
                frozenset() if self._all_input_universal
                else compute_ip_universal_states(fst)
            )
            self.target = self._target_trie.initial
            self.parent = None
        else:
            self.source_alphabet = parent.source_alphabet
            self.target_alphabet = parent.target_alphabet
            self.logp_gap = parent.logp_gap
            self.max_steps = parent.max_steps
            self.max_beam = parent.max_beam
            self.heuristic = parent.heuristic
            self._target_trie = parent._target_trie
            self._source_trie = parent._source_trie
            self._graph = parent._graph
            self._all_input_universal = parent._all_input_universal
            self._ip_universal_states = parent._ip_universal_states
            self.target = target
            self.parent = parent
            assert self._target_trie._parent[self.target][0] == parent.target

        self._compute()

    def _clip(self, ys):
        """Clip *ys* to target depth: return ancestor at target depth, or *ys* if shallower.

        Cached per ``_compute`` call.  O(1) amortized.
        """
        cached = self._clip_cache.get(ys)
        if cached is not None:
            return cached
        trie = self._target_trie
        if trie._depth[ys] <= self._target_depth:
            result = ys
        else:
            cur = ys
            while trie._depth[cur] > self._target_depth:
                cur = trie._parent[cur][0]
            result = cur
        self._clip_cache[ys] = result
        return result

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
        target_ancestors = self._target_ancestors
        result = any(
            ys in target_ancestors or self._clip(ys) == target
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
            if self._clip(ys) == target
        )
        self._is_final_cache[id(node)] = result
        return result

    def _compute(self):
        """Lazy-expansion search with deferred ``graph.step`` calls.

        Each heap entry is either *resolved* (string is at a known graph node)
        or *deferred* (string wants to transition through a source symbol, but
        the expensive epsilon-closure ``graph.step`` hasn't been called yet).

        Deferred entries are cheap to push.  The expensive ``graph.step`` only
        runs when an entry is actually popped from the heap.  With an informed
        heuristic (e.g. an LM), low-probability symbols are never popped and
        their transitions are never computed — exploration cost scales with
        ``max_steps`` rather than ``max_steps × |alphabet|``.
        """
        self._quotient = {}     # xs -> (h_state, node)
        self._remainder = {}    # xs -> (h_state, node)
        # Property caches are target-dependent, so reset each _compute call.
        self._is_live_cache = {}
        self._is_final_cache = {}
        self._is_universal_cache = {}
        self._clip_cache = {}
        # Monotonicity caches for universality (element-indexed).
        self._pos_index = defaultdict(set)
        self._pos_sizes = {}
        self._pos_next = 0
        self._neg_index = defaultdict(set)
        self._neg_next = 0
        self._target_depth = self._target_trie.depth(self.target)
        self._target_ancestors = self._target_trie.ancestors(self.target)

        graph = self._graph
        source_trie = self._source_trie
        fst = self.fst

        # Flat priority heap: (h_state, tiebreak, payload)
        # payload = (xs, node, None)  — resolved: string xs is at node
        # payload = (xs, parent, sym) — deferred: needs graph.step(parent, sym)
        heap = []
        counter = 0

        # Per-node caches (keyed by id(node); safe because nodes are interned).
        node_class = {}           # 'universal' | 'remainder' | 'expand'
        node_available_syms = {}  # set of source symbols (cheap to compute)

        def enqueue(h_state, xs, node, sym=None):
            """Push a heap entry.  sym=None → resolved; sym=<symbol> → deferred."""
            nonlocal counter
            # Eager fast-path: resolved entries at known-universal nodes.
            if sym is None and node_class.get(id(node)) == 'universal':
                self._quotient[xs] = (h_state, node)
                return
            heapq.heappush(heap, (h_state, counter, (xs, node, sym)))
            counter += 1

        # --- Seed ---
        if self._target_depth == 0:
            enqueue(self.heuristic, source_trie.initial, graph.initial)
        else:
            K = self.max_beam

            # Remainder strings that are still final under the new target stay in remainder.
            # When beam-limited, keep only the top-K by heuristic score.
            rem_candidates = [
                (h_state, xs, node)
                for xs, (h_state, node) in self.parent._remainder.items()
                if self._is_final(node)
            ]
            if len(rem_candidates) > K:
                rem_candidates.sort()
                rem_candidates = rem_candidates[:K]
            for h_state, xs, node in rem_candidates:
                self._remainder[xs] = (h_state, node)

            # Quotient strings that are still live go back for re-evaluation,
            # since universality may no longer hold under the extended target.
            # When beam-limited, keep only the top-K by heuristic score.
            quot_candidates = [
                (h_state, xs, node)
                for xs, (h_state, node) in self.parent._quotient.items()
                if self._is_live(node)
            ]
            if len(quot_candidates) > K:
                quot_candidates.sort()
                quot_candidates = quot_candidates[:K]
            for h_state, xs, node in quot_candidates:
                enqueue(h_state, xs, node)

        # --- Main loop ---
        steps = 0
        logp_gap = self.logp_gap

        # Reference logp for gap stopping: use the parent's best quotient logp
        # as the baseline.  This avoids a cold-start problem where the local
        # best_quotient_logp starts at -inf and the gap check is disabled until
        # the first local quotient is found.  The parent's estimate is already
        # a good approximation of the prefix probability mass.
        if self.parent is not None:
            best_quotient_logp = max(
                (getattr(h, 'logp', -float('inf'))
                 for h, _node in self.parent._quotient.values()),
                default=-float('inf'),
            )
        else:
            best_quotient_logp = -float('inf')

        while heap and steps < self.max_steps:
            h_state, _, (xs, node, sym) = heapq.heappop(heap)

            # Log-probability gap stopping: if the best remaining entry
            # is far below the best quotient entry, the remaining mass is
            # negligible — stop early.
            if logp_gap is not None and best_quotient_logp > -float('inf'):
                h_logp = getattr(h_state, 'logp', None)
                if h_logp is not None and best_quotient_logp - h_logp > logp_gap:
                    break

            # Resolve deferred entries: call graph.step now.
            if sym is not None:
                node = graph.step(node, sym)
                if not self._is_live(node):
                    continue
                # Fast-path: already known universal → straight to quotient.
                if node_class.get(id(node)) == 'universal':
                    self._quotient[xs] = (h_state, node)
                    continue

            nid = id(node)
            steps += 1

            # Classify node (cached per node per _compute call).
            cls = node_class.get(nid)
            if cls is None:
                if self._is_final(node):
                    cls = 'universal' if self._is_universal(node) else 'remainder'
                else:
                    cls = 'expand'
                node_class[nid] = cls
                # Compute available source symbols (cheap: just scan arc labels).
                if cls != 'universal':
                    syms = set()
                    for s, _ys in node:
                        syms.update(fst.delta[s].keys())
                    syms.discard(EPSILON)
                    node_available_syms[nid] = syms

            if cls == 'universal':
                self._quotient[xs] = (h_state, node)
                # Track best quotient logp for gap stopping.
                h_logp = getattr(h_state, 'logp', None)
                if h_logp is not None and h_logp > best_quotient_logp:
                    best_quotient_logp = h_logp
                continue

            if cls == 'remainder':
                self._remainder[xs] = (h_state, node)

            # Expand: push DEFERRED entries — graph.step not called until pop.
            for a in node_available_syms[nid]:
                enqueue(h_state >> a, source_trie.extend(xs, a), node, sym=a)

    def _add_pos(self, node):
        """Record *node* as known-universal in the element-indexed positive cache."""
        eid = self._pos_next
        self._pos_next += 1
        self._pos_sizes[eid] = len(node)
        for e in node:
            self._pos_index[e].add(eid)

    def _add_neg(self, node):
        """Record *node* as known-non-universal in the element-indexed negative cache."""
        eid = self._neg_next
        self._neg_next += 1
        for e in node:
            self._neg_index[e].add(eid)

    def _has_pos_subset(self, node):
        """Is there a known-universal set u such that u ⊆ node?"""
        hits = {}
        for e in node:
            for eid in self._pos_index.get(e, ()):
                h = hits.get(eid, 0) + 1
                if h == self._pos_sizes[eid]:
                    return True
                hits[eid] = h
        return False

    def _has_neg_superset(self, node):
        """Is there a known-non-universal set nu such that node ⊆ nu?"""
        candidates = None
        for e in node:
            entry_ids = self._neg_index.get(e)
            if entry_ids is None:
                return False
            if candidates is None:
                candidates = set(entry_ids)
            else:
                candidates &= entry_ids
            if not candidates:
                return False
        return bool(candidates)

    def _is_universal(self, node):
        """BFS check: does every extension of *node* remain final?

        A node is universal if the sub-DFA rooted at it (under the augmented
        powerset construction) accepts ``source_alphabet*`` — i.e., every
        reachable node is final and complete.  Universal nodes are "cylinders"
        of the precover: the source string and all its extensions map to the
        target, so they belong in the quotient.
        """

        # Per-node cache (keyed by id; safe because nodes are interned).
        cached = self._is_universal_cache.get(id(node))
        if cached is not None:
            return cached

        result = self._is_universal_impl(node)
        self._is_universal_cache[id(node)] = result
        return result

    def _is_universal_impl(self, node):

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
                if q in self._ip_universal_states and self._clip(ys) == target:
                    self._add_pos(node)
                    return True

        # Superset monotonicity: node ⊇ some known-universal set?
        if self._has_pos_subset(node):
            return True

        # Subset monotonicity: node ⊆ some known-non-universal set?
        if self._has_neg_superset(node):
            return False

        result = self._is_universal_bfs(node)
        return result

    def _is_universal_bfs(self, node):
        """BFS over clipped states to check universality without graph bloat.

        Works entirely with (fst_state, clipped_ys) pairs where ys is clipped
        to ``target_depth``.  This avoids creating new interned graph nodes or
        trie entries beyond the target depth, preventing the O(steps × BFS_size)
        memory growth of the graph-based BFS.

        Correctness relies on ``clip(extend(ys, b)) == clip(extend(clip(ys), b))``
        — clipping commutes with extension — so the clipped BFS explores the
        same refined state space as the original.
        """
        fst = self.fst
        trie = self._target_trie
        target = self.target
        target_depth = self._target_depth
        target_ancestors = self._target_ancestors
        source_alphabet_len = len(self.source_alphabet)
        fst_stop = fst.stop
        trie_depth = trie._depth

        def _extend_clipped(ys, b):
            """Extend ys by output symbol b, clipping to target_depth."""
            if b == EPSILON:
                return ys
            if trie_depth[ys] >= target_depth:
                return ys   # at/past target depth; extension clips back
            return trie.extend(ys, b)

        def _eps_closure_clipped(pairs):
            """Epsilon closure keeping ys clipped to target_depth."""
            worklist = set(pairs)
            result = set()
            while worklist:
                pair = worklist.pop()
                if pair in result:
                    continue
                result.add(pair)
                s, ys = pair
                for b, next_s in fst.arcs(s, EPSILON):
                    worklist.add((next_s, _extend_clipped(ys, b)))
            return result

        def _step_clipped(clipped_state, symbol):
            """Transition clipped state by source symbol, returning clipped successor."""
            next_pairs = set()
            for s, ys in clipped_state:
                for b, j in fst.arcs(s, symbol):
                    next_pairs.add((j, _extend_clipped(ys, b)))
            return frozenset(_eps_closure_clipped(next_pairs))

        def _compatible(clipped_state):
            """Filter to target-compatible pairs (for cycle detection)."""
            return frozenset(
                (s, ys) for s, ys in clipped_state
                if ys in target_ancestors or ys == target
            )

        def _is_final_clipped(clipped_state):
            return any(s in fst_stop for s, ys in clipped_state if ys == target)

        # Build initial clipped state from raw node.
        initial = frozenset((s, self._clip(ys)) for s, ys in node)
        root_compat = _compatible(initial)

        worklist = deque()
        worklist.append(initial)
        visited = {root_compat}

        while worklist:
            cur = worklist.popleft()
            if not _is_final_clipped(cur):
                self._add_neg(node)
                return False
            # Completeness: source symbols from ALL pairs (including incompatible).
            src_symbols = set()
            for s, _ys in cur:
                src_symbols.update(fst.delta[s].keys())
            src_symbols.discard(EPSILON)
            if len(src_symbols) < source_alphabet_len:
                self._add_neg(node)
                return False
            for a in src_symbols:
                next_clipped = _step_clipped(cur, a)
                next_compat = _compatible(next_clipped)
                if not next_compat:
                    self._add_neg(node)
                    return False
                if next_compat not in visited:
                    visited.add(next_compat)
                    worklist.append(next_clipped)

        self._add_pos(node)
        return True

    @property
    def quotient(self):
        return FSA.from_strings(self._source_trie.to_tuple(xs) for xs in self._quotient)

    @property
    def remainder(self):
        return FSA.from_strings(self._source_trie.to_tuple(xs) for xs in self._remainder)

    def __rshift__(self, y):
        if y not in self.target_alphabet:
            raise ValueError(f"Out of vocabulary target symbol: {y!r}")
        return PrioritizedLazyIncremental(self.fst, self._target_trie.extend(self.target, y), parent=self)
