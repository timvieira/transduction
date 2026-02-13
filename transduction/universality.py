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
    Greatest-fixpoint computation of ip-universal FST states.

    A state q is ip-universal if the input projection of the FST, started from
    eps_close({q}), accepts Sigma*. This is strictly more general than
    check_all_input_universal, which only checks the start set.

    Algorithm:
    1. Precompute eps_close({q}) for all FST states q
    2. Initialize candidates = set(fst.states)
    3. Iteratively remove states that violate universality:
       - eps_close({q}) must contain a final state
       - eps_close({q}) must have arcs for every symbol in Sigma
       - For each symbol, the successor eps-closure must contain >= 1 candidate
    4. Fixed point = set of ip-universal states
    """
    source_alphabet = fst.A - {eps}
    if not source_alphabet:
        return {q for q in fst.states if fst.is_final(q)}

    def ip_eps_close(states):
        visited = set(states)
        worklist = deque(states)
        while worklist:
            s = worklist.popleft()
            for a, _b, j in fst.arcs(s):
                if a == eps and j not in visited:
                    visited.add(j)
                    worklist.append(j)
        return frozenset(visited)

    # Precompute closures
    closures = {q: ip_eps_close({q}) for q in fst.states}

    # Precompute per-closure: successor sets by symbol
    # For each closure, for each symbol, the raw destinations (before eps-close)
    closure_symbol_succs = {}
    for q in fst.states:
        by_symbol = defaultdict(set)
        for s in closures[q]:
            for a, _b, j in fst.arcs(s):
                if a != eps:
                    by_symbol[a].add(j)
        closure_symbol_succs[q] = by_symbol

    candidates = set(fst.states)

    changed = True
    while changed:
        changed = False
        to_remove = set()
        for q in candidates:
            closure = closures[q]

            # Must contain a final state
            if not any(fst.is_final(s) for s in closure):
                to_remove.add(q)
                continue

            # Must be complete on source alphabet
            by_symbol = closure_symbol_succs[q]
            if not all(a in by_symbol for a in source_alphabet):
                to_remove.add(q)
                continue

            # For each symbol, successor eps-closure must contain a candidate
            ok = True
            for a in source_alphabet:
                succ_closure = ip_eps_close(by_symbol[a])
                if not (succ_closure & candidates):
                    ok = False
                    break
            if not ok:
                to_remove.add(q)

        if to_remove:
            candidates -= to_remove
            changed = True

    return frozenset(candidates)


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
