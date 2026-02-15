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


def _check_universality(filt, fst_univ_cache, i, mbp, N):
    """Check if a final DFA state is universal, with caching for pure-frontier states.

    Pure-frontier states (where every NFA element has buffer length == N) are cached
    by their FST state set, since universality depends only on the FST states.
    """
    if mbp == N and all(len(buf) == N for (_, buf) in i):
        fst_key = frozenset(fst_state for (fst_state, _) in i)
        is_uni = fst_univ_cache.get(fst_key)
        if is_uni is None:
            is_uni = filt.is_universal(i)
            fst_univ_cache[fst_key] = is_uni
        return is_uni
    return filt.is_universal(i)


def _consume(obj):
    """Mark a decomposition state as consumed, raising if already consumed."""
    if obj._consumed:
        raise RuntimeError(
            "This decomposition state has already been consumed by >> or decompose_next()."
        )
    obj._consumed = True


def _make_filter(fst, target, dfa, source_alphabet, all_input_universal, ip_universal_states):
    """Create a UniversalityFilter with witness states for the given target."""
    witnesses = frozenset((q, target) for q in ip_universal_states)
    return UniversalityFilter(
        fst, target, dfa, source_alphabet,
        all_input_universal=all_input_universal,
        witnesses=witnesses,
    )


def _trimmed_fsa(start_states, stop_states, get_incoming):
    """Build a trimmed FSA by backward BFS from stop states through reverse arcs.

    All states in the incoming index are forward-reachable (built by BFS),
    so backward reachability from stops gives exactly the trim set.
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

    def __init__(self, fst, target=(), parent=None):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        target = tuple(target)
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

    def _bfs_expand(self, dfa, filt, worklist, seen, N):
        """BFS expansion of new DFA states.

        Expands all STATUS_NEW states reachable from the worklist, classifying
        each as QSTOP (universal final), RSTOP (non-universal final), or
        INTERIOR.  Updates _dfa_trans, _dfa_status, _incoming, _max_bufpos.

        Returns:
            (frontier, q_stops, r_stops, n_expanded, n_arcs)
        """
        frontier = set()
        q_stops = set()
        r_stops = set()
        n_expanded = 0
        n_arcs = 0

        while worklist:
            i = worklist.popleft()
            if self._dfa_status.get(i, STATUS_NEW) != STATUS_NEW:
                continue

            n_expanded += 1
            mbp = max(len(buf) for (_, buf) in i)
            self._max_bufpos[i] = mbp

            if mbp >= N:
                frontier.add(i)

            if dfa.is_final(i):
                if _check_universality(filt, self._fst_univ_cache, i, mbp, N):
                    self._dfa_status[i] = STATUS_QSTOP
                    self._dfa_trans[i] = {}
                    q_stops.add(i)
                    continue

            arcs = {}
            for a, j in dfa.arcs(i):
                arcs[a] = j
                n_arcs += 1
                self._incoming.setdefault(j, set()).add((a, i))
                if j not in seen and self._dfa_status.get(j, STATUS_NEW) == STATUS_NEW:
                    worklist.append(j)
                    seen.add(j)

            self._dfa_trans[i] = arcs
            if dfa.is_final(i):
                self._dfa_status[i] = STATUS_RSTOP
                r_stops.add(i)
            else:
                self._dfa_status[i] = STATUS_INTERIOR

        return frontier, q_stops, r_stops, n_expanded, n_arcs

    def _build_fresh(self):
        dfa = PrecoverNFA(self.fst, self.target).det()
        filt = _make_filter(self.fst, self.target, dfa, self.source_alphabet,
                            self._all_input_universal, self._ip_universal_states)

        self._dfa_trans = {}
        self._dfa_status = {}
        self._incoming = {}
        self._max_bufpos = {}
        self._filt = filt

        N = len(self.target)
        worklist = deque()
        seen = set()
        start_states = []

        for i in dfa.start():
            worklist.append(i)
            seen.add(i)
            start_states.append(i)
            self._incoming.setdefault(i, set())

        frontier, q_stops, r_stops, n_expanded, n_arcs = (
            self._bfs_expand(dfa, filt, worklist, seen, N)
        )

        # Persist eps closure cache for incremental reuse
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            self._eps_cache = dfa.fsa._closure_cache

        self._start_states = start_states
        self._frontier = frontier
        self._q_stops = q_stops
        self._r_stops = r_stops
        self._stats = dict(
            n_dirty=0, n_border=0,
            n_expanded=n_expanded, n_arcs=n_arcs,
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
        q_stops_base = parent._q_stops - invalidated
        r_stops_base = parent._r_stops - invalidated

        # Build new lazy DFA for extended target
        dfa = PrecoverNFA(self.fst, self.target).det()

        # Inject persisted eps closure cache
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            dfa.fsa._closure_cache = self._eps_cache

        # Recompute start states from the new DFA
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

        # BFS from dirty|border plus any new start states.
        worklist = deque(invalidated)
        seen = set(invalidated)
        for s in self._start_states:
            self._incoming.setdefault(s, set())
            if s not in seen and self._dfa_status.get(s, STATUS_NEW) == STATUS_NEW:
                worklist.append(s)
                seen.add(s)

        frontier, new_q, new_r, n_expanded, n_arcs = (
            self._bfs_expand(dfa, filt, worklist, seen, N)
        )

        # Persist eps closure cache for next incremental step
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            self._eps_cache = dfa.fsa._closure_cache

        self._frontier = frontier
        self._q_stops = q_stops_base | new_q
        self._r_stops = r_stops_base | new_r
        self._stats = dict(
            n_dirty=len(dirty), n_border=len(border),
            n_expanded=n_expanded, n_arcs=n_arcs,
            total_dfa_states=len(self._dfa_status),
            n_frontier=len(frontier),
        )

    def _get_incoming(self, state):
        """Get incoming arcs for a state."""
        return self._incoming.get(state, ())

    @cached_property
    def quotient(self):
        return _trimmed_fsa(self._start_states, self._q_stops, self._get_incoming)

    @cached_property
    def remainder(self):
        return _trimmed_fsa(self._start_states, self._r_stops, self._get_incoming)

    def __rshift__(self, y):
        _consume(self)
        return TruncatedIncrementalDFADecomp(self.fst, self.target + (y,), parent=self)

    def decompose_next(self):
        """Decompose for every next target symbol using lightweight overlays.

        All branches share the parent's clean-state arcs.  Each branch stores
        only its dirty/border/new arc overrides in small dicts.
        """
        _consume(self)

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
        self.target = parent.target + (y,)
        self.parent = parent
        self._consumed = False
        self._all_input_universal = parent._all_input_universal
        self._ip_universal_states = parent._ip_universal_states
        self._has_eps = parent._has_eps
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

        self._expand(parent, invalidated, old_arcs)

    def _get_status(self, i):
        """Get state status, checking overlay then base."""
        return self._overlay_status.get(i, self._base_status.get(i, STATUS_NEW))

    def _expand(self, parent, invalidated, old_arcs):
        """Run the expansion BFS for this branch's target extension."""
        # Remove old outgoing arcs of invalidated states from incoming
        for s in invalidated:
            for label, dest in old_arcs.get(s, {}).items():
                self._overlay_incoming_remove.setdefault(dest, set()).add((label, s))
            self._overlay_trans[s] = {}
            self._overlay_status[s] = STATUS_NEW

        # Build lazy DFA for this branch's extended target
        dfa = PrecoverNFA(self.fst, self.target).det()
        self._start_states = list(dfa.start())

        filt = _make_filter(self.fst, self.target, dfa, self.source_alphabet,
                            self._all_input_universal, self._ip_universal_states)

        N = len(self.target)
        frontier = set()
        q_stops = set(parent._q_stops - invalidated)
        r_stops = set(parent._r_stops - invalidated)

        # BFS from invalidated plus any new start states.
        worklist = deque(invalidated)
        seen = set(invalidated)
        for s in self._start_states:
            if s not in seen and self._get_status(s) == STATUS_NEW:
                worklist.append(s)
                seen.add(s)

        while worklist:
            i = worklist.popleft()
            if self._get_status(i) != STATUS_NEW:
                continue

            mbp = max(len(buf) for (_, buf) in i)
            self._overlay_max_bufpos[i] = mbp

            if mbp >= N:
                frontier.add(i)

            if dfa.is_final(i):
                if _check_universality(filt, self._fst_univ_cache, i, mbp, N):
                    self._overlay_status[i] = STATUS_QSTOP
                    self._overlay_trans[i] = {}
                    q_stops.add(i)
                    continue

            arcs = {}
            for a, j in dfa.arcs(i):
                arcs[a] = j
                self._overlay_incoming_add.setdefault(j, set()).add((a, i))
                if j not in seen and self._get_status(j) == STATUS_NEW:
                    worklist.append(j)
                    seen.add(j)

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
        _consume(self)
        flat = self._flatten()
        return TruncatedIncrementalDFADecomp(flat.fst, flat.target + (y,), parent=flat)

    def decompose_next(self):
        """Flatten and delegate to TruncatedIncrementalDFADecomp.decompose_next()."""
        _consume(self)
        return self._flatten().decompose_next()

    def _flatten(self):
        """Merge overlay into flat dicts, returning a TruncatedIncrementalDFADecomp."""
        dfa_trans = dict(self._base_trans)
        dfa_trans.update(self._overlay_trans)

        dfa_status = dict(self._base_status)
        dfa_status.update(self._overlay_status)

        incoming = {}
        for state, arcs in self._base_incoming.items():
            incoming[state] = set(arcs)
        for state, removals in self._overlay_incoming_remove.items():
            if state in incoming:
                incoming[state] -= removals
        for state, additions in self._overlay_incoming_add.items():
            incoming.setdefault(state, set()).update(additions)

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
        dfa = PrecoverNFA(self.fst, self.target).det()
        obj._filt = _make_filter(self.fst, self.target, dfa, self.source_alphabet,
                                 self._all_input_universal, self._ip_universal_states)
        return obj
