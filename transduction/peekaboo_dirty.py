"""True dirty-state incremental peekaboo decomposition.

Combines dirty-state DFA persistence (as in TruncatedIncrementalDFADecomp)
with per-symbol Q/R classification from PeekabooFixedNFA (K=1 lookahead).

Uses PeekabooFixedNFA (2-tuple NFA states ``(i, ys)``) rather than
PeekabooLookaheadNFA (3-tuple ``(i, ys, truncated)``) because the 2-tuple
format provides stable state identities across target extensions — the
truncation flag in PeekabooLookaheadNFA changes meaning when the target
grows, violating the dirty-state invariant that clean states are
identical in both the old and new NFAs.

On each target extension by one symbol:
- Dirty states (NFA elements at the frontier) and border states are identified
- Only dirty+border states are re-expanded with the new NFA
- Clean interior states reuse cached arcs
- Per-symbol Q/R emerges from the peekaboo BFS classification
"""

from transduction.base import IncrementalDecomposition
from transduction.fst import EPSILON
from transduction.lazy import Lazy, EpsilonRemove
from transduction.universality import (
    UniversalityFilter,
    check_all_input_universal,
    compute_ip_universal_states,
)
from transduction.precover_nfa import PeekabooFixedNFA
from transduction.peekaboo_incremental import _trimmed_fsa
from collections import deque
from functools import cached_property

STATUS_NEW = 0
STATUS_INTERIOR = 1
STATUS_QSTOP = 2
STATUS_RSTOP = 3


class _FixedTruncatedDFA(Lazy):
    """TruncatedDFA variant for PeekabooFixedNFA (2-tuple NFA states).

    Wraps a determinized PeekabooFixedNFA, providing ``is_final`` for a
    specific target extension and ``refine`` normalization that clips and
    filters NFA elements.  Used by the universality sub-BFS.
    """

    def __init__(self, *, dfa, fst, target):
        self.dfa = dfa
        self.fst = fst
        self.target = target

    def start(self):
        return self.dfa.start()

    def refine(self, frontier):
        N = len(self.target)
        return frozenset(
            (i, ys[:N]) for i, ys in frontier
            if ys[:min(N, len(ys))] == self.target[:min(N, len(ys))]
        )

    def arcs(self, state):
        for x, next_state in self.dfa.arcs(state):
            yield x, self.refine(next_state)

    def arcs_x(self, state, x):
        for next_state in self.dfa.arcs_x(state, x):
            yield self.refine(next_state)

    def is_final(self, state):
        N = len(self.target)
        return any(
            ys[:N] == self.target and self.fst.is_final(i)
            for (i, ys) in state
        )


class DirtyPeekaboo(IncrementalDecomposition):
    """Dirty-state incremental peekaboo decomposition.

    Persists the full DFA structure (arcs, status, reverse arcs) across
    target extensions.  On each extension by one symbol, only "dirty"
    states (whose NFA powerset contains frontier elements) and "border"
    states (clean predecessors of dirty) are re-expanded.

    Per-symbol Q/R comes from PeekabooFixedNFA's buffer: NFA elements
    with buffer length > N peek at the next output symbol, yielding
    per-symbol quotient and remainder stops from a single BFS.
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
        self._has_eps = EPSILON in fst.A
        self._cat = (lambda s, y: s + (y,)) if isinstance(target, tuple) else (lambda s, y: s + y)

        if parent is not None:
            assert fst is parent.fst
            self._all_input_universal = parent._all_input_universal
            self._ip_universal_states = parent._ip_universal_states
            self._build_incremental(parent)
        else:
            self._all_input_universal = check_all_input_universal(fst)
            self._ip_universal_states = (
                frozenset() if self._all_input_universal
                else compute_ip_universal_states(fst)
            )
            self._eps_cache = {}
            self._build_fresh()

    def _bfs_expand(self, dfa, worklist, seen, N):
        """BFS expansion with per-symbol Q/R classification.

        Returns (frontier, q_stops, r_stops) where q_stops and r_stops
        are dicts mapping symbol -> set of DFA states.
        """
        frontier = set()
        q_stops = {}
        r_stops = {}
        univ_filters = {}
        _all_input_universal = self._all_input_universal
        _fst_is_final = self.fst.is_final
        target = self.target

        _cat = self._cat

        def ensure_filter(y):
            if y not in univ_filters and not _all_input_universal:
                target_ext = _cat(target, y)
                trunc_dfa = _FixedTruncatedDFA(
                    dfa=dfa, fst=self.fst, target=target_ext,
                )
                witnesses = frozenset(
                    (q, target_ext) for q in self._ip_universal_states
                )
                univ_filters[y] = UniversalityFilter(
                    fst=self.fst, target=target_ext, dfa=trunc_dfa,
                    source_alphabet=self.source_alphabet,
                    all_input_universal=_all_input_universal,
                    witnesses=witnesses,
                )

        while worklist:
            i = worklist.popleft()
            if self._dfa_status.get(i, STATUS_NEW) != STATUS_NEW:
                continue

            mbp = max(len(ys) for (_, ys) in i)
            self._max_bufpos[i] = mbp

            if mbp >= N:
                frontier.add(i)

            # Per-symbol classification
            relevant_symbols = set()
            final_symbols = set()
            for fst_state, ys in i:
                if len(ys) > N:
                    y = ys[N]
                    relevant_symbols.add(y)
                    if ys[:N] == target and _fst_is_final(fst_state):
                        final_symbols.add(y)

            # At most one relevant_symbol can be continuous (universal).
            # NOTE: This assumes the FST is functional.  A productive
            # input-epsilon cycle (eps-input arcs that produce non-epsilon
            # output) makes an FST non-functional, since the cycle can be
            # traversed any number of times yielding distinct outputs for
            # the same input.  Non-functional FSTs may violate this
            # uniqueness invariant.
            continuous = False
            for y in relevant_symbols:
                ensure_filter(y)
                if not continuous:
                    is_univ = (
                        y in final_symbols if _all_input_universal
                        else univ_filters[y].is_universal(i)
                    )
                    if is_univ:
                        q_stops.setdefault(y, set()).add(i)
                        continuous = True
                        continue
                if y in final_symbols:
                    r_stops.setdefault(y, set()).add(i)

            if continuous:
                self._dfa_status[i] = STATUS_QSTOP
                self._dfa_trans[i] = {}
                continue

            arcs = {}
            for a, j in dfa.arcs(i):
                arcs[a] = j
                self._incoming.setdefault(j, set()).add((a, i))
                if j not in seen and self._dfa_status.get(j, STATUS_NEW) == STATUS_NEW:
                    worklist.append(j)
                    seen.add(j)

            self._dfa_trans[i] = arcs
            if final_symbols:
                self._dfa_status[i] = STATUS_RSTOP
            else:
                self._dfa_status[i] = STATUS_INTERIOR

        return frontier, q_stops, r_stops

    def _build_fresh(self):
        dfa = PeekabooFixedNFA(self.fst, self.target).det()
        self._dfa_trans = {}
        self._dfa_status = {}
        self._incoming = {}
        self._max_bufpos = {}

        N = len(self.target)
        worklist = deque()
        seen = set()
        start_states = []

        for i in dfa.start():
            worklist.append(i)
            seen.add(i)
            start_states.append(i)
            self._incoming.setdefault(i, set())

        frontier, q_stops, r_stops = self._bfs_expand(dfa, worklist, seen, N)

        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            self._eps_cache = dfa.fsa._closure_cache

        self._start_states = start_states
        self._frontier = frontier
        self._q_stops = q_stops
        self._r_stops = r_stops

    def _build_incremental(self, parent):
        old_depth = len(parent.target)

        # Copy parent's data (allows multiple children to continue independently)
        self._dfa_trans = dict(parent._dfa_trans)
        self._dfa_status = dict(parent._dfa_status)
        self._incoming = {s: set(arcs) for s, arcs in parent._incoming.items()}
        self._max_bufpos = dict(parent._max_bufpos)

        # Evict stale eps closure cache entries
        if self._has_eps:
            eps_cache = dict(parent._eps_cache)
            stale = [k for k, v in eps_cache.items()
                     if len(k[1]) >= old_depth
                     or any(len(s[1]) >= old_depth for s in v)]
            for k in stale:
                del eps_cache[k]
            self._eps_cache = eps_cache
        else:
            self._eps_cache = {}

        # Dirty = parent's frontier (states with NFA elements at boundary)
        dirty = parent._frontier

        # Border = clean predecessors of dirty states
        border = set()
        for D in dirty:
            for (_, src) in self._incoming.get(D, ()):
                if src not in dirty:
                    border.add(src)

        # Remove old arcs of dirty|border and reset their status
        invalidated = dirty | border
        for s in invalidated:
            old_arcs = self._dfa_trans.get(s, {})
            for label, dest in old_arcs.items():
                inc = self._incoming.get(dest)
                if inc is not None:
                    inc.discard((label, s))
            self._dfa_trans[s] = {}
            self._dfa_status[s] = STATUS_NEW

        # Build new lazy DFA for extended target
        dfa = PeekabooFixedNFA(self.fst, self.target).det()
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            dfa.fsa._closure_cache = self._eps_cache

        self._start_states = list(dfa.start())
        N = len(self.target)

        worklist = deque(invalidated)
        seen = set(invalidated)
        for s in self._start_states:
            self._incoming.setdefault(s, set())
            if s not in seen and self._dfa_status.get(s, STATUS_NEW) == STATUS_NEW:
                worklist.append(s)
                seen.add(s)

        frontier, q_stops, r_stops = self._bfs_expand(dfa, worklist, seen, N)

        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            self._eps_cache = dfa.fsa._closure_cache

        self._frontier = frontier
        # All parent Q/R stops are in the frontier (dirty), so no carry-forward
        self._q_stops = q_stops
        self._r_stops = r_stops

    def decompose_next(self):
        """Return {y: child} with per-symbol Q/R from the peekaboo BFS."""
        return {y: _DirtyPeekabooChild(self, y)
                for y in sorted(self.target_alphabet)}

    def __rshift__(self, y):
        return DirtyPeekaboo(self.fst, self._cat(self.target, y), parent=self)

    @cached_property
    def _qr(self):
        parent = self.parent
        assert parent is not None
        y = self.target[-1]
        q_stops = parent._q_stops.get(y, set())
        r_stops = parent._r_stops.get(y, set())
        return (
            _trimmed_fsa(parent._start_states, q_stops, parent._incoming),
            _trimmed_fsa(parent._start_states, r_stops, parent._incoming),
        )

    @property
    def quotient(self):
        return self._qr[0]

    @property
    def remainder(self):
        return self._qr[1]


class _DirtyPeekabooChild(IncrementalDecomposition):
    """Lightweight child from decompose_next() with pre-computed Q/R.

    Holds a reference to the parent DirtyPeekaboo for Q/R extraction.
    When decompose_next() or >> is called, builds a full DirtyPeekaboo
    incrementally from the parent.
    """

    def __init__(self, parent, y):
        self.fst = parent.fst
        self.source_alphabet = parent.source_alphabet
        self.target_alphabet = parent.target_alphabet
        self.target = parent._cat(parent.target, y)
        self._parent_ref = parent
        self._symbol = y

    @cached_property
    def quotient(self):
        parent = self._parent_ref
        q_stops = parent._q_stops.get(self._symbol, set())
        return _trimmed_fsa(parent._start_states, q_stops, parent._incoming)

    @cached_property
    def remainder(self):
        parent = self._parent_ref
        r_stops = parent._r_stops.get(self._symbol, set())
        return _trimmed_fsa(parent._start_states, r_stops, parent._incoming)

    @cached_property
    def _full(self):
        """Lazily build the full DirtyPeekaboo for incremental continuation."""
        return DirtyPeekaboo(self.fst, self.target, parent=self._parent_ref)

    def decompose_next(self):
        return self._full.decompose_next()

    def __rshift__(self, z):
        return self._full >> z
