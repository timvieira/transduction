from transduction import FSA, EPSILON
from transduction.base import IncrementalDecomposition
from transduction.precover_nfa import PrecoverNFA
from transduction.universality import (
    UniversalityFilter,
    check_all_input_universal,
    compute_ip_universal_states,
)
from collections import deque
from functools import cached_property
from transduction.lazy import EpsilonRemove

# State status constants (matching Rust DirtyDecomp)
STATUS_NEW = 0       # needs full expansion
STATUS_INTERIOR = 1  # non-final, expanded (has cached arcs)
STATUS_QSTOP = 2     # universal final, no outgoing arcs
STATUS_RSTOP = 3     # non-universal final, expanded (has cached arcs)


def _trimmed_fsa(start_states, stop_states, get_incoming):
    """Build a trimmed FSA by backward BFS from stop states through reverse arcs.

    All states in the incoming index are forward-reachable (built by BFS),
    so backward reachability from stops gives exactly the trim set.

    Args:
        start_states: iterable of start states
        stop_states: set of stop states
        get_incoming: callable(state) -> iterable of (label, predecessor) pairs
    """
    if not stop_states:
        return FSA()
    backward_reachable = set()
    worklist = deque(stop_states)
    while worklist:
        state = worklist.popleft()
        if state in backward_reachable:
            continue
        backward_reachable.add(state)
        for _, pred in get_incoming(state):
            if pred not in backward_reachable:
                worklist.append(pred)
    arcs = [
        (pred, x, state)
        for state in backward_reachable
        for x, pred in get_incoming(state)
        if pred in backward_reachable
    ]
    return FSA(
        start={s for s in start_states if s in backward_reachable},
        arcs=arcs,
        stop=stop_states,
    )


class TruncatedIncrementalDFADecomp(IncrementalDecomposition):
    """Dirty-state incremental decomposition with lazy Q/R materialization.

    Ports the Rust ``DirtyDecomp`` strategy to Python: the DFA structure
    (per-state transitions, universality classification) is persisted across
    ``>>`` steps.  On each extension of the target by one symbol, only
    "dirty" states (whose NFA powerset contains frontier elements) and
    "border" states (clean states with an arc into a dirty state) are
    re-expanded.  All other states reuse their cached arcs.

    Key optimizations over a naive approach:

    - ``_incoming`` reverse-arc index enables O(|incoming arcs to dirty|) border
      identification instead of scanning all arcs.
    - ``_frontier`` tracking identifies dirty states in O(|frontier|) instead of
      scanning all states.
    - Expansion BFS starts from dirty+border only, never visiting clean interior
      states.
    - Q/R FSAs are built via backward BFS from stop states through ``_incoming``,
      avoiding the O(|full DFA|) forward-reachability pass.
    - ``decompose_next()`` creates lightweight overlay branches that share the
      parent's clean-state arcs; each branch stores only its dirty/border/new
      arc overrides.
    """

    def __init__(self, fst, target='', parent=None):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")
        self.target = target
        self.parent = parent
        self._consumed = False
        self._has_eps = EPSILON in fst.A

        if parent is not None:
            assert fst is parent.fst
            self._all_input_universal = parent._all_input_universal
            self._ip_universal_states = parent._ip_universal_states
            self._fst_univ_cache = parent._fst_univ_cache
            self._build_incremental(parent)
        else:
            self._all_input_universal = check_all_input_universal(fst)
            self._ip_universal_states = (
                frozenset() if self._all_input_universal
                else compute_ip_universal_states(fst)
            )
            self._eps_cache = {}
            self._fst_univ_cache = {}
            self._build_fresh()

    def _make_filter(self, dfa):
        witnesses = frozenset(
            (q, self.target) for q in self._ip_universal_states
        )
        return UniversalityFilter(
            self.fst, self.target, dfa, self.source_alphabet,
            all_input_universal=self._all_input_universal,
            witnesses=witnesses,
        )

    def _build_fresh(self):
        dfa = PrecoverNFA(self.fst, self.target).det()
        filt = self._make_filter(dfa)

        self._dfa_trans = {}
        self._dfa_status = {}
        self._incoming = {}   # dest -> {(label, src), ...}
        self._max_bufpos = {}
        self._filt = filt

        N = len(self.target)
        frontier = set()
        q_stops = set()
        r_stops = set()

        worklist = deque()
        visited = set()
        start_states = []

        for i in dfa.start():
            worklist.append(i)
            visited.add(i)
            start_states.append(i)
            self._incoming.setdefault(i, set())

        while worklist:
            i = worklist.popleft()

            # Compute max_bufpos for O(1) frontier/dirty detection
            mbp = max(len(buf) for (_, buf) in i)
            self._max_bufpos[i] = mbp

            # Track frontier: states with any NFA element at buffer length == N
            if mbp >= N:
                frontier.add(i)

            if dfa.is_final(i):
                # Check fst_univ_cache for pure frontier states
                if mbp == N and all(len(buf) == N for (_, buf) in i):
                    fst_key = frozenset(fst_state for (fst_state, _) in i)
                    is_uni = self._fst_univ_cache.get(fst_key)
                    if is_uni is None:
                        is_uni = filt.is_universal(i)
                        self._fst_univ_cache[fst_key] = is_uni
                else:
                    is_uni = filt.is_universal(i)
                if is_uni:
                    self._dfa_status[i] = STATUS_QSTOP
                    self._dfa_trans[i] = {}
                    q_stops.add(i)
                    continue

            arcs = {}
            for a, j in dfa.arcs(i):
                arcs[a] = j
                self._incoming.setdefault(j, set()).add((a, i))
                if j not in visited:
                    worklist.append(j)
                    visited.add(j)

            self._dfa_trans[i] = arcs
            if i not in self._dfa_status:
                if dfa.is_final(i):
                    self._dfa_status[i] = STATUS_RSTOP
                    r_stops.add(i)
                else:
                    self._dfa_status[i] = STATUS_INTERIOR

        # Persist eps closure cache for incremental reuse
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            self._eps_cache = dfa.fsa._closure_cache

        self._start_states = start_states
        self._frontier = frontier
        self._q_stops = q_stops
        self._r_stops = r_stops
        self._stats = dict(
            n_dirty=0, n_border=0,
            n_expanded=len(visited),
            n_arcs=sum(len(a) for a in self._dfa_trans.values()),
            total_dfa_states=len(self._dfa_status),
            n_frontier=len(frontier),
        )

    def _build_incremental(self, parent):
        old_depth = len(parent.target)

        # Take ownership of parent's dicts (O(1) reference transfer).
        # Parent is invalidated after this.
        self._dfa_trans = parent._dfa_trans
        self._dfa_status = parent._dfa_status
        self._incoming = parent._incoming
        self._max_bufpos = parent._max_bufpos

        # Transfer and evict stale eps closure cache
        if self._has_eps:
            eps_cache = parent._eps_cache
            stale = [k for k, v in eps_cache.items()
                     if len(k[1]) >= old_depth
                     or any(len(s[1]) >= old_depth for s in v)]
            for k in stale:
                del eps_cache[k]
            self._eps_cache = eps_cache
        else:
            self._eps_cache = {}

        # Dirty = parent's frontier states (tracked, not scanned)
        dirty = parent._frontier

        # Border = predecessors of dirty states via _incoming (not in dirty)
        border = set()
        for D in dirty:
            for (_, src) in self._incoming.get(D, ()):
                if src not in dirty:
                    border.add(src)

        # Remove old outgoing arcs of dirty|border from _incoming[dest],
        # then clear their transitions and reset status.
        invalidated = dirty | border
        for s in invalidated:
            old_arcs = self._dfa_trans.get(s, {})
            for label, dest in old_arcs.items():
                inc = self._incoming.get(dest)
                if inc is not None:
                    inc.discard((label, s))
            self._dfa_trans[s] = {}
            self._dfa_status[s] = STATUS_NEW

        # Start with parent's stops, minus invalidated states
        q_stops = (parent._q_stops - invalidated)
        r_stops = (parent._r_stops - invalidated)

        # Build new lazy DFA for extended target
        dfa = PrecoverNFA(self.fst, self.target).det()

        # Inject persisted eps closure cache
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            dfa.fsa._closure_cache = self._eps_cache

        # Recompute start states from the new DFA (they can change when the
        # target grows, especially from empty to non-empty).
        self._start_states = list(dfa.start())

        # Evict stale universality cache entries and update witnesses
        filt = parent._filt
        new_witnesses = frozenset(
            (q, self.target) for q in self._ip_universal_states
        )
        filt.evict_frontier(old_depth, self.target, new_witnesses)
        filt.dfa = dfa
        self._filt = filt

        N = len(self.target)
        frontier = set()
        n_expanded = 0
        n_arcs = 0

        # BFS from dirty|border plus any new start states.
        worklist = deque(invalidated)
        expanding = set(invalidated)
        for s in self._start_states:
            self._incoming.setdefault(s, set())
            if s not in expanding and self._dfa_status.get(s, STATUS_NEW) == STATUS_NEW:
                worklist.append(s)
                expanding.add(s)

        while worklist:
            i = worklist.popleft()
            status = self._dfa_status.get(i, STATUS_NEW)

            if status != STATUS_NEW:
                # Clean state reached as a successor — nothing to expand.
                continue

            n_expanded += 1

            # Compute max_bufpos for O(1) frontier/dirty detection
            mbp = max(len(buf) for (_, buf) in i)
            self._max_bufpos[i] = mbp

            # Track frontier
            if mbp >= N:
                frontier.add(i)

            if dfa.is_final(i):
                # Check fst_univ_cache for pure frontier states
                if mbp == N and all(len(buf) == N for (_, buf) in i):
                    fst_key = frozenset(fst_state for (fst_state, _) in i)
                    is_uni = self._fst_univ_cache.get(fst_key)
                    if is_uni is None:
                        is_uni = filt.is_universal(i)
                        self._fst_univ_cache[fst_key] = is_uni
                else:
                    is_uni = filt.is_universal(i)
                if is_uni:
                    self._dfa_status[i] = STATUS_QSTOP
                    self._dfa_trans[i] = {}
                    q_stops.add(i)
                    continue

            arcs = {}
            for a, j in dfa.arcs(i):
                arcs[a] = j
                n_arcs += 1
                self._incoming.setdefault(j, set()).add((a, i))
                if j not in expanding and self._dfa_status.get(j, STATUS_NEW) == STATUS_NEW:
                    worklist.append(j)
                    expanding.add(j)

            self._dfa_trans[i] = arcs
            if dfa.is_final(i):
                self._dfa_status[i] = STATUS_RSTOP
                r_stops.add(i)
            else:
                self._dfa_status[i] = STATUS_INTERIOR

        # Persist eps closure cache for next incremental step
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            self._eps_cache = dfa.fsa._closure_cache

        self._frontier = frontier
        self._q_stops = q_stops
        self._r_stops = r_stops
        self._stats = dict(
            n_dirty=len(dirty), n_border=len(border),
            n_expanded=n_expanded, n_arcs=n_arcs,
            total_dfa_states=len(self._dfa_status),
            n_frontier=len(frontier),
        )

    def _get_incoming(self, state):
        """Get incoming arcs for a state. Override in overlay subclass."""
        return self._incoming.get(state, ())

    @cached_property
    def quotient(self):
        return _trimmed_fsa(self._start_states, self._q_stops, self._get_incoming)

    @cached_property
    def remainder(self):
        return _trimmed_fsa(self._start_states, self._r_stops, self._get_incoming)

    def __rshift__(self, y):
        if self._consumed:
            raise RuntimeError(
                "This decomposition state has already been consumed by >> or decompose_next(). "
                "Use decompose_next() to create multiple branches."
            )
        self._consumed = True
        return TruncatedIncrementalDFADecomp(self.fst, self.target + y, parent=self)

    def decompose_next(self):
        """Decompose for every next target symbol using lightweight overlays.

        All branches share the parent's clean-state arcs.  Each branch stores
        only its dirty/border/new arc overrides in small dicts.
        """
        if self._consumed:
            raise RuntimeError(
                "This decomposition state has already been consumed by >> or decompose_next()."
            )
        self._consumed = True

        dirty = self._frontier
        border = set()
        for D in dirty:
            for (_, src) in self._incoming.get(D, ()):
                if src not in dirty:
                    border.add(src)

        # Snapshot old arcs from dirty|border for incoming removal in overlays
        invalidated = dirty | border
        old_arcs = {s: dict(self._dfa_trans.get(s, {})) for s in invalidated}

        results = {}
        for y in sorted(self.target_alphabet):
            results[y] = _OverlayChild(
                parent=self, y=y,
                dirty=dirty, border=border,
                invalidated=invalidated, old_arcs=old_arcs,
            )
        return results


class _OverlayChild(IncrementalDecomposition):
    """Lightweight overlay branch created by decompose_next().

    Shares the parent's clean-state arcs via read-only references.
    Stores only dirty/border/new arc overrides in small dicts.
    """

    def __init__(self, parent, y, dirty, border, invalidated, old_arcs):
        self.fst = parent.fst
        self.source_alphabet = parent.source_alphabet
        self.target_alphabet = parent.target_alphabet
        self.target = parent.target + y
        self.parent = parent
        self._consumed = False
        self._all_input_universal = parent._all_input_universal
        self._ip_universal_states = parent._ip_universal_states
        self._has_eps = EPSILON in parent.fst.A
        self._fst_univ_cache = parent._fst_univ_cache

        # Shared base (read-only references to parent's clean state)
        self._base_trans = parent._dfa_trans
        self._base_status = parent._dfa_status
        self._base_incoming = parent._incoming

        # Per-branch overlay: only dirty/border/new states
        self._overlay_trans = {}
        self._overlay_status = {}
        self._overlay_max_bufpos = {}
        # incoming diffs: add/remove sets per dest state
        self._overlay_incoming_add = {}
        self._overlay_incoming_remove = {}

        self._expand(parent, dirty, border, invalidated, old_arcs)

    def _expand(self, parent, dirty, border, invalidated, old_arcs):
        """Run the expansion BFS for this branch's target extension."""
        old_depth = len(parent.target)

        # Remove old outgoing arcs of invalidated states from incoming
        for s in invalidated:
            for label, dest in old_arcs.get(s, {}).items():
                self._overlay_incoming_remove.setdefault(dest, set()).add((label, s))
            self._overlay_trans[s] = {}
            self._overlay_status[s] = STATUS_NEW

        # Build lazy DFA for this branch's extended target
        dfa = PrecoverNFA(self.fst, self.target).det()

        # Recompute start states from the new DFA
        self._start_states = list(dfa.start())

        # Create a fresh lightweight filter for this branch
        witnesses = frozenset(
            (q, self.target) for q in self._ip_universal_states
        )
        filt = UniversalityFilter(
            self.fst, self.target, dfa, self.source_alphabet,
            all_input_universal=self._all_input_universal,
            witnesses=witnesses,
        )

        N = len(self.target)
        frontier = set()
        q_stops = set(parent._q_stops - invalidated)
        r_stops = set(parent._r_stops - invalidated)

        # BFS from dirty|border plus any new start states.
        worklist = deque(invalidated)
        expanding = set(invalidated)
        for s in self._start_states:
            s_status = self._overlay_status.get(s, self._base_status.get(s, STATUS_NEW))
            if s not in expanding and s_status == STATUS_NEW:
                worklist.append(s)
                expanding.add(s)

        while worklist:
            i = worklist.popleft()
            status = self._overlay_status.get(i, self._base_status.get(i, STATUS_NEW))

            if status != STATUS_NEW:
                continue

            # Compute max_bufpos inline for overlay states
            mbp = max(len(buf) for (_, buf) in i)
            self._overlay_max_bufpos[i] = mbp

            if mbp >= N:
                frontier.add(i)

            if dfa.is_final(i):
                # Check fst_univ_cache for pure frontier states
                if mbp == N and all(len(buf) == N for (_, buf) in i):
                    fst_key = frozenset(fst_state for (fst_state, _) in i)
                    is_uni = self._fst_univ_cache.get(fst_key)
                    if is_uni is None:
                        is_uni = filt.is_universal(i)
                        self._fst_univ_cache[fst_key] = is_uni
                else:
                    is_uni = filt.is_universal(i)
                if is_uni:
                    self._overlay_status[i] = STATUS_QSTOP
                    self._overlay_trans[i] = {}
                    q_stops.add(i)
                    continue

            arcs = {}
            for a, j in dfa.arcs(i):
                arcs[a] = j
                self._overlay_incoming_add.setdefault(j, set()).add((a, i))
                j_status = self._overlay_status.get(j, self._base_status.get(j, STATUS_NEW))
                if j not in expanding and j_status == STATUS_NEW:
                    worklist.append(j)
                    expanding.add(j)

            self._overlay_trans[i] = arcs
            if dfa.is_final(i):
                self._overlay_status[i] = STATUS_RSTOP
                r_stops.add(i)
            else:
                self._overlay_status[i] = STATUS_INTERIOR

        # Save eps closure cache for _flatten
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            self._expand_eps_cache = dfa.fsa._closure_cache
        else:
            self._expand_eps_cache = {}

        self._frontier = frontier
        self._q_stops = q_stops
        self._r_stops = r_stops

    def _get_incoming(self, state):
        """Merged incoming view: base - removals + additions."""
        base = self._base_incoming.get(state, set())
        remove = self._overlay_incoming_remove.get(state, set())
        add = self._overlay_incoming_add.get(state, set())
        if not remove and not add:
            return base
        return (base - remove) | add

    @cached_property
    def quotient(self):
        return _trimmed_fsa(self._start_states, self._q_stops, self._get_incoming)

    @cached_property
    def remainder(self):
        return _trimmed_fsa(self._start_states, self._r_stops, self._get_incoming)

    def __rshift__(self, y):
        """Advance by one target symbol. Flattens the overlay into a
        TruncatedIncrementalDFADecomp for further incremental use."""
        if self._consumed:
            raise RuntimeError(
                "This decomposition state has already been consumed by >> or decompose_next()."
            )
        self._consumed = True
        flat = self._flatten()
        return TruncatedIncrementalDFADecomp(flat.fst, flat.target + y, parent=flat)

    def decompose_next(self):
        """Flatten and delegate to TruncatedIncrementalDFADecomp.decompose_next()."""
        if self._consumed:
            raise RuntimeError(
                "This decomposition state has already been consumed by >> or decompose_next()."
            )
        self._consumed = True
        return self._flatten().decompose_next()

    def _flatten(self):
        """Merge overlay into flat dicts, returning a TruncatedIncrementalDFADecomp."""
        # Build flat dicts by applying overlay on top of base
        dfa_trans = dict(self._base_trans)
        dfa_trans.update(self._overlay_trans)

        dfa_status = dict(self._base_status)
        dfa_status.update(self._overlay_status)

        incoming = {}
        # Start from base incoming
        for state, arcs in self._base_incoming.items():
            incoming[state] = set(arcs)
        # Apply removals
        for state, removals in self._overlay_incoming_remove.items():
            if state in incoming:
                incoming[state] -= removals
        # Apply additions
        for state, additions in self._overlay_incoming_add.items():
            incoming.setdefault(state, set()).update(additions)

        # Merge max_bufpos: parent base + overlay
        max_bufpos = dict(self.parent._max_bufpos)
        max_bufpos.update(self._overlay_max_bufpos)

        # Build a shell object with flat dicts
        obj = object.__new__(TruncatedIncrementalDFADecomp)
        obj.fst = self.fst
        obj.source_alphabet = self.source_alphabet
        obj.target_alphabet = self.target_alphabet
        obj.target = self.target
        obj.parent = self.parent
        obj._consumed = False
        obj._has_eps = self._has_eps
        obj._all_input_universal = self._all_input_universal
        obj._ip_universal_states = self._ip_universal_states
        obj._fst_univ_cache = self._fst_univ_cache
        obj._eps_cache = self._expand_eps_cache
        obj._dfa_trans = dfa_trans
        obj._dfa_status = dfa_status
        obj._incoming = incoming
        obj._max_bufpos = max_bufpos
        obj._start_states = self._start_states
        obj._frontier = self._frontier
        obj._q_stops = self._q_stops
        obj._r_stops = self._r_stops
        # Create a fresh filter for the flat object
        dfa = PrecoverNFA(self.fst, self.target).det()
        obj._filt = UniversalityFilter(
            self.fst, self.target, dfa, self.source_alphabet,
            all_input_universal=self._all_input_universal,
            witnesses=frozenset(
                (q, self.target) for q in self._ip_universal_states
            ),
        )
        return obj
