from collections import defaultdict, deque

from transduction.fsa import EPSILON

eps = EPSILON


def check_all_input_universal(fst):
    """
    O(|arcs|) check: does the input projection of this FST accept Σ* from the
    start state?

    Works by checking that:
    1. The eps-closed start set contains a final state
    2. The start set has arcs for every source symbol (completeness)
    3. Every symbol's successor eps-closure contains the start set

    If (3) holds, all reachable DFA states of the input projection contain the
    start set, hence are all final and complete → the start state is universal.
    """
    source_alphabet = fst.A - {eps}
    if not source_alphabet:
        # Empty alphabet: universal iff some start state is final
        return any(fst.is_final(s) for s in fst.start)

    # Eps-close start states over input-side ε arcs
    def ip_eps_close(states):
        visited = set(states)
        worklist = deque(states)
        while worklist:
            s = worklist.popleft()
            for a, _b, j in fst.arcs(s):
                if a == eps and j not in visited:
                    visited.add(j)
                    worklist.append(j)
        return visited

    start_set = ip_eps_close(fst.start)

    # Must contain a final state
    if not any(fst.is_final(s) for s in start_set):
        return False

    # Group non-ε input arcs from start_set by input symbol
    by_symbol = defaultdict(set)
    for s in start_set:
        for a, _b, j in fst.arcs(s):
            if a != eps:
                by_symbol[a].add(j)

    # Must be complete on source alphabet
    if len(by_symbol) < len(source_alphabet):
        return False

    # Check that every symbol's successor eps-closure contains the start set
    for _sym, raw_dests in by_symbol.items():
        closed = ip_eps_close(raw_dests)
        if not start_set <= closed:
            return False

    return True


def compute_ip_universal_states(fst):
    """
    Kahn-style worklist computation of ip-universal FST states.

    A state q is ip-universal if the input projection of the FST, started from
    eps_close({q}), accepts Sigma*. This is strictly more general than
    check_all_input_universal, which only checks the start set.

    Uses a two-level counting scheme to avoid materializing the ε-removed NFA
    (which can be quadratically larger than the original FST):

    - Level 1 (raw destinations): For each raw destination j of a non-ε arc,
      alive[j] tracks how many states in eps_close(j) are still in U.
    - Level 2 (per-state symbol counts): raw_count[q][x] tracks how many raw
      x-destinations of q still have alive[j] > 0.

    When a state q' is removed from U, we decrement alive[j] for each j whose
    ε-closure contains q'. When alive[j] hits 0, we decrement raw_count[q][x]
    for each (q, x) that had j as a raw destination.
    """
    source_alphabet = fst.A - {eps}
    if not source_alphabet:
        return frozenset(q for q in fst.states if fst.is_final(q))

    # ε-closures on the input projection
    def ip_eps_close(state):
        visited = {state}
        worklist = deque([state])
        while worklist:
            s = worklist.popleft()
            for a, _b, j in fst.arcs(s):
                if a == eps and j not in visited:
                    visited.add(j)
                    worklist.append(j)
        return frozenset(visited)

    closures = {q: ip_eps_close(q) for q in fst.states}

    # eps_close_rev[q'] = {j | q' ∈ closures[j]} — reverse ε-closure index
    eps_close_rev = defaultdict(set)
    for j in fst.states:
        for qp in closures[j]:
            eps_close_rev[qp].add(j)

    # Raw destination sets: raw[q][x] = {j | s ∈ closures[q], (s,x,j) ∈ arcs}
    # raw_reverse[j] = [(q, x)] — which (state, symbol) pairs have j as raw dest
    raw = {q: defaultdict(set) for q in fst.states}
    raw_reverse = defaultdict(list)
    for q in fst.states:
        for s in closures[q]:
            for a, _b, j in fst.arcs(s):
                if a != eps and j not in raw[q][a]:
                    raw[q][a].add(j)
                    raw_reverse[j].append((q, a))

    # U = final states of the ε-removed NFA
    U = {q for q in fst.states if closures[q] & fst.stop}

    # Level 1: alive[j] = |closures[j] ∩ U|
    alive = {j: len(closures[j] & U) for j in fst.states}

    # Level 2: raw_count[q][x] = |{j ∈ raw[q][x] | alive[j] > 0}|
    raw_count = {}
    for q in fst.states:
        raw_count[q] = {}
        for x, dests in raw[q].items():
            raw_count[q][x] = sum(1 for j in dests if alive[j] > 0)

    # Seed queue: states in U missing a live successor for some symbol
    queue = set()
    for q in U:
        for x in source_alphabet:
            if raw_count[q].get(x, 0) == 0:
                queue.add(q)
                break

    # Two-level Kahn worklist
    while queue:
        qp = queue.pop()
        U.discard(qp)
        # Level 1: qp left U, so alive[j] decreases for each j whose
        # ε-closure contained qp
        for j in eps_close_rev[qp]:
            alive[j] -= 1
            if alive[j] == 0:
                # Level 2: j is now dead, decrement raw_count for its users
                for q, x in raw_reverse[j]:
                    raw_count[q][x] -= 1
                    if raw_count[q][x] == 0 and q in U:
                        queue.add(q)

    return frozenset(U)


class UniversalityFilter:
    """
    Encapsulates universality short-circuit optimizations:
    - Fast path: if check_all_input_universal, every final state is universal
    - ip-universal witness check via set intersection
    - Superset monotonicity: if S is universal, any S' ⊇ S is too
    - Subset monotonicity: if S is not universal, any S' ⊆ S isn't either
    - Fallback: BFS universality check on the DFA

    Monotonicity caches use element-indexed lookups rather than linear scans.
    """

    def __init__(self, fst, target, dfa, source_alphabet, *,
                 all_input_universal=None, witnesses=None):
        self.dfa = dfa
        self.source_alphabet = source_alphabet
        self.all_input_universal = (
            check_all_input_universal(fst) if all_input_universal is None
            else all_input_universal
        )
        if not self.all_input_universal:
            if witnesses is not None:
                self._witnesses = witnesses
            else:
                ip_univ = compute_ip_universal_states(fst)
                self._witnesses = frozenset((q, target) for q in ip_univ)
        # Element-indexed positive cache (known universal states).
        # _pos_index[element] = set of entry IDs whose stored set contains element.
        # A stored set u ⊆ dfa_state iff every element of u is in dfa_state,
        # i.e., the entry's hit count equals its size.
        self._pos_index = defaultdict(set)
        self._pos_sizes = {}   # entry_id -> len(stored set)
        self._pos_next = 0
        # Element-indexed negative cache (known non-universal states).
        # A stored set nu ⊇ dfa_state iff every element of dfa_state is in nu,
        # i.e., the intersection of entry-ID sets across all elements is non-empty.
        self._neg_index = defaultdict(set)
        self._neg_next = 0

    def _add_pos(self, s):
        eid = self._pos_next
        self._pos_next += 1
        self._pos_sizes[eid] = len(s)
        for e in s:
            self._pos_index[e].add(eid)

    def _add_neg(self, s):
        eid = self._neg_next
        self._neg_next += 1
        for e in s:
            self._neg_index[e].add(eid)

    def evict_frontier(self, old_depth, new_target, new_witnesses):
        """Evict stale cache entries and update witnesses for a new target.

        An NFA state ``(fst_state, buffer)`` is "dirty" if
        ``len(buffer) >= old_depth`` — it's a frontier state whose behavior
        changes when the target grows.

        Removes any positive/negative cache entry that was indexed by a dirty
        NFA element, and replaces witnesses with *new_witnesses*.
        """
        self._witnesses = new_witnesses

        def is_dirty(elem):
            return len(elem[1]) >= old_depth

        # --- Positive cache eviction ---
        dirty_pos_eids = set()
        dirty_pos_keys = []
        for elem, eids in self._pos_index.items():
            if is_dirty(elem):
                dirty_pos_keys.append(elem)
                dirty_pos_eids |= eids

        if dirty_pos_eids:
            for elem in list(self._pos_index):
                self._pos_index[elem] -= dirty_pos_eids
            for eid in dirty_pos_eids:
                # Poison so has_pos_subset can never match this entry
                self._pos_sizes[eid] = float('inf')
        for key in dirty_pos_keys:
            del self._pos_index[key]

        # --- Negative cache eviction ---
        dirty_neg_eids = set()
        dirty_neg_keys = []
        for elem, eids in self._neg_index.items():
            if is_dirty(elem):
                dirty_neg_keys.append(elem)
                dirty_neg_eids |= eids

        if dirty_neg_eids:
            for elem in list(self._neg_index):
                self._neg_index[elem] -= dirty_neg_eids
        for key in dirty_neg_keys:
            del self._neg_index[key]

    def _has_pos_subset(self, dfa_state):
        """Is there a known-universal set u such that u ⊆ dfa_state?"""
        hits = {}
        for e in dfa_state:
            for eid in self._pos_index.get(e, ()):
                h = hits.get(eid, 0) + 1
                if h == self._pos_sizes[eid]:
                    return True
                hits[eid] = h
        return False

    def _has_neg_superset(self, dfa_state):
        """Is there a known-non-universal set nu such that dfa_state ⊆ nu?"""
        candidates = None
        for e in dfa_state:
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

    def _bfs_universal(self, state):
        """BFS check: does `state` accept source_alphabet* in the DFA?"""
        visited = set()
        worklist = deque()
        visited.add(state)
        worklist.append(state)
        while worklist:
            i = worklist.popleft()
            if not self.dfa.is_final(i):
                return False
            dest = dict(self.dfa.arcs(i))
            for a in self.source_alphabet:
                if a not in dest:
                    return False
                j = dest[a]
                if j not in visited:
                    visited.add(j)
                    worklist.append(j)
        return True

    def is_universal(self, dfa_state):
        """Returns True/False for whether dfa_state accepts Sigma*."""

        # A state must be final to accept Sigma* (since epsilon is in Sigma*)
        if not self.dfa.is_final(dfa_state):
            return False

        # Fast path: all input universal means every final state is universal
        if self.all_input_universal:
            return True

        # ip-universal witness check: short-circuits on first common element
        if not self._witnesses.isdisjoint(dfa_state):
            self._add_pos(dfa_state)
            return True

        # Superset monotonicity: is dfa_state ⊇ some known-universal set?
        if self._has_pos_subset(dfa_state):
            return True

        # Subset monotonicity: is dfa_state ⊆ some known-non-universal set?
        if self._has_neg_superset(dfa_state):
            return False

        # BFS fallback
        result = self._bfs_universal(dfa_state)
        (self._add_pos if result else self._add_neg)(dfa_state)
        return result
