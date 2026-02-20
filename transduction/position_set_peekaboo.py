"""
Position-set peekaboo decomposition for token-decomposable FSTs.

Quotients the PeekabooLookaheadNFA DFA by position descriptor sets: frozensets
of (buf_len, extra_sym_or_None, truncated) triples.  Under token-decomposability,
states with the same position set have identical transitions, finality, and
per-symbol Q/R classification, giving dramatic compression (e.g., 634->18 states
for PTB).

Implements the decomp_state_cls interface for TransducedLM.

Usage with TransducedLM:
    from transduction.lm.transduced import TransducedLM
    from transduction.position_set_peekaboo import (
        PositionSetPeekabooState, _PositionSetPeekabooUniv,
    )

    tlm = TransducedLM(
        inner, fst, K=100,
        decomp_state_cls=PositionSetPeekabooState,
        univ_cls=_PositionSetPeekabooUniv,
    )
"""

from collections import deque

from transduction.base import DecompositionResult
from transduction.fsa import FSA, EPSILON
from transduction.precover_nfa import PeekabooLookaheadNFA
from transduction.peekaboo_incremental import FstUniversality, TruncatedDFA


def _peekaboo_position_set(dfa_state, N):
    """Extract peekaboo position set from a PeekabooLookaheadNFA DFA state.

    Each NFA triple (fst_state, buffer, truncated) maps to a position descriptor
    (buf_len, extra_sym_or_None, truncated).  extra_sym is buf[N] when
    buf_len > N, None otherwise.  The position set is the frozenset of all
    descriptors in the DFA state.
    """
    return frozenset(
        (len(buf), buf[N] if len(buf) > N else None, truncated)
        for (q, buf, truncated) in dfa_state
    )


class _PositionSetDFA:
    """Compact DFA adapter over position-set IDs for TransducedLM beam search.

    Maps integer position-set IDs to arcs.  Provides .start(), .arcs(),
    .run() matching the interface expected by TransducedLM.
    """

    __slots__ = ('_start_id', '_arcs_list', '_arcs_dict')

    def __init__(self, start_id, arcs_list, arcs_dict):
        self._start_id = start_id
        self._arcs_list = arcs_list   # {pid: [(symbol, succ_pid), ...]}
        self._arcs_dict = arcs_dict   # {pid: {symbol: succ_pid}}

    def start(self):
        return [self._start_id]

    def arcs(self, state_id):
        return self._arcs_list.get(state_id, [])

    def run(self, source_path):
        """Run a source path from start, returning the reached state or None."""
        state = self._start_id
        for x in source_path:
            d = self._arcs_dict.get(state)
            if d is None or x not in d:
                return None
            state = d[x]
        return state


class _PositionSetPeekabooUniv(FstUniversality):
    """Universality precomputation for PositionSetPeekabooState.

    Pass as ``univ_cls`` to TransducedLM to share the FstUniversality
    computation across all ``>> y`` steps::

        TransducedLM(inner, fst, K=100,
                     decomp_state_cls=PositionSetPeekabooState,
                     univ_cls=_PositionSetPeekabooUniv)
    """
    pass


def _check_td_finality(state1, state2, ps, N, target, is_final):
    """Partial TD check: states with the same position set must agree on
    per-symbol finality and preimage status.

    Skips the transition consistency check, which would require materializing
    arcs for non-canonical DFA states (defeating the Phase 2 optimization).
    """
    def _final_syms(state):
        syms = set()
        for (q, buf, _tr) in state:
            if len(buf) > N and buf[:N] == target and is_final(q):
                syms.add(buf[N])
        return syms

    def _is_preimage(state):
        return any(len(buf) == N and is_final(q) for (q, buf, _tr) in state)

    fs1 = _final_syms(state1)
    fs2 = _final_syms(state2)
    if fs1 != fs2:
        raise ValueError(
            f"FST is not token-decomposable: per-symbol finality mismatch "
            f"at position set {ps}: {fs1} vs {fs2}"
        )

    p1 = _is_preimage(state1)
    p2 = _is_preimage(state2)
    if p1 != p2:
        raise ValueError(
            f"FST is not token-decomposable: preimage mismatch "
            f"at position set {ps}"
        )


class PositionSetPeekabooState:
    """Position-set peekaboo decomposition for TransducedLM.

    Quotients the PeekabooLookaheadNFA DFA by position descriptor sets.
    For token-decomposable FSTs, this gives identical Q/R with dramatically
    fewer states (e.g., 634->18 for PTB).

    Attributes (lazy, computed on first access):
        decomp : dict[symbol, DecompositionResult]
            Maps next target symbol to Q/R stop states (position-set IDs).
        dfa : _PositionSetDFA
            Compact DFA over position-set IDs.
        resume_frontiers : dict
            Always empty (Phase 1 -- no incremental reuse).
        preimage_stops : set[int]
            Position-set IDs where source has produced exactly ``target``
            and the FST is final.

    Raises ValueError if the FST is detected as not token-decomposable.
    """

    _LAZY_BFS_ATTRS = frozenset({
        'decomp', 'dfa', 'resume_frontiers', 'preimage_stops',
    })

    def __init__(self, fst, target=(), parent=None, *, univ=None):
        self.fst = fst
        self.target = tuple(target) if not isinstance(target, tuple) else target
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        oov = set(self.target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")
        # Accept FstUniversality (or subclass) directly; create one otherwise.
        self._univ = (
            univ if isinstance(univ, FstUniversality)
            else FstUniversality(fst)
        )
        self._parent = parent
        self._bfs_done = False

    def __getattr__(self, name):
        if name in PositionSetPeekabooState._LAZY_BFS_ATTRS:
            self._ensure_bfs()
            return self.__dict__[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute {name!r}"
        )

    def _ensure_bfs(self):
        """Run the position-set peekaboo BFS on demand.  No-op if done."""
        if self._bfs_done:
            return

        target = self.target
        N = len(target)
        fst = self.fst
        _fst_is_final = fst.is_final

        # ── Step 1: Create lazy peekaboo DFA ──────────────────────────
        raw_dfa = PeekabooLookaheadNFA(fst, target).det()

        # ── Step 2: BFS over position sets (quotiented during construction) ─
        # Only call raw_dfa.arcs() on canonical representatives — one per
        # position set.  Non-canonical successors get a cheap finality check
        # but their arcs are never materialized.  For PTB this means ~18
        # arc expansions instead of ~634.
        canonical = {}   # ps -> canonical DFA state
        ps_id = {}       # ps -> int id
        ps_arcs = {}     # pid -> {x: succ_pid}
        next_id = 0
        start_ps = None

        for s in raw_dfa.start():
            ps = _peekaboo_position_set(s, N)
            if ps not in canonical:
                canonical[ps] = s
                ps_id[ps] = next_id
                next_id += 1
            if start_ps is None:
                start_ps = ps

        worklist = deque()
        expanded = set()
        for ps in canonical:
            worklist.append(ps)

        while worklist:
            ps = worklist.popleft()
            pid = ps_id[ps]
            if pid in expanded:
                continue
            expanded.add(pid)

            rep = canonical[ps]
            arcs = {}

            for x, succ_state in raw_dfa.arcs(rep):
                succ_ps = _peekaboo_position_set(succ_state, N)

                if succ_ps not in canonical:
                    canonical[succ_ps] = succ_state
                    ps_id[succ_ps] = next_id
                    next_id += 1
                    worklist.append(succ_ps)
                else:
                    # Partial TD check: finality and preimage only
                    _check_td_finality(
                        succ_state, canonical[succ_ps], succ_ps,
                        N, target, _fst_is_final,
                    )

                arcs[x] = ps_id[succ_ps]

            ps_arcs[pid] = arcs

        # ── Step 3: Per-symbol Q/R classification ────────────────────
        _aiu = self._univ.all_input_universal
        univ_filters = {}

        def ensure_univ_filter(y):
            if y not in univ_filters and not _aiu:
                trunc_dfa = TruncatedDFA(
                    dfa=raw_dfa, fst=fst, target=target + (y,),
                )
                univ_filters[y] = self._univ.make_filter(
                    fst, target + (y,), trunc_dfa, self.source_alphabet,
                )

        decomp = {}
        preimage_stops = set()

        for ps in canonical:
            pid = ps_id[ps]
            rep = canonical[ps]

            # Relevant symbols: position descriptors with buf_len > N
            relevant_symbols = set()
            for (buf_len, extra_sym, truncated) in ps:
                if buf_len > N and extra_sym is not None:
                    relevant_symbols.add(extra_sym)

            # Preimage: buf_len == N and FST state is final
            if any(
                len(buf) == N and _fst_is_final(q)
                for (q, buf, _tr) in rep
            ):
                preimage_stops.add(pid)

            # Per-symbol finality from canonical representative
            final_symbols = set()
            for (q, buf, _tr) in rep:
                if len(buf) > N and buf[:N] == target and _fst_is_final(q):
                    final_symbols.add(buf[N])

            # Per-symbol universality — matching PeekabooState behavior:
            # universal → Q-stop, final-but-not-universal → R-stop.
            continuous = None
            for y in relevant_symbols:
                if y not in decomp:
                    decomp[y] = DecompositionResult(set(), set())
                ensure_univ_filter(y)

                if _aiu:
                    is_univ = y in final_symbols
                else:
                    is_univ = univ_filters[y].is_universal(rep)

                if is_univ:
                    decomp[y].quotient.add(pid)
                    continuous = y
                    continue

                if y in final_symbols:
                    decomp[y].remainder.add(pid)

        # ── Step 4: Build DFA adapter ────────────────────────────────
        arcs_list = {}
        arcs_dict = {}
        for pid, sym_map in ps_arcs.items():
            arcs_list[pid] = list(sym_map.items())
            arcs_dict[pid] = sym_map

        self.decomp = decomp
        self.dfa = _PositionSetDFA(ps_id[start_ps], arcs_list, arcs_dict)
        self.resume_frontiers = {}   # Phase 1: always empty
        self.preimage_stops = preimage_stops
        self._n_position_sets = len(expanded)
        self._bfs_done = True

    def _build_qr_fsa(self, y):
        """Build Q and R FSAs for symbol y from this state's decomposition.

        Forward BFS from start, stopping expansion at Q-absorbed states
        (matching PeekabooState's behavior where continuous/Q states are
        not expanded past).
        """
        d = self.decomp.get(y)
        if d is None:
            return FSA(), FSA()

        # Collect all Q-stops (across all symbols) — BFS stops at any Q-stop
        all_q_stops = set()
        for dr in self.decomp.values():
            all_q_stops.update(dr.quotient)

        q_stops = d.quotient
        r_stops = d.remainder

        Q = FSA()
        R = FSA()
        [start] = self.dfa.start()
        Q.add_start(start)
        R.add_start(start)

        visited = set()
        worklist = deque([start])
        visited.add(start)

        while worklist:
            pid = worklist.popleft()

            if pid in q_stops:
                Q.add_stop(pid)
            if pid in r_stops:
                R.add_stop(pid)

            # Don't expand past any Q-stop
            if pid in all_q_stops:
                continue

            for x, succ_pid in self.dfa.arcs(pid):
                Q.add_arc(pid, x, succ_pid)
                R.add_arc(pid, x, succ_pid)
                if succ_pid not in visited:
                    visited.add(succ_pid)
                    worklist.append(succ_pid)

        return Q.trim(), R.trim()

    @property
    def quotient(self):
        """Q FSA from parent's decomposition for this state's last target symbol."""
        if self._parent is None:
            raise AttributeError("Root state has no quotient")
        if '_qr_cache' not in self.__dict__:
            self._qr_cache = self._parent._build_qr_fsa(self.target[-1])
        return self._qr_cache[0]

    @property
    def remainder(self):
        """R FSA from parent's decomposition for this state's last target symbol."""
        if self._parent is None:
            raise AttributeError("Root state has no remainder")
        if '_qr_cache' not in self.__dict__:
            self._qr_cache = self._parent._build_qr_fsa(self.target[-1])
        return self._qr_cache[1]

    def decompose_next(self):
        """Returns {y: PositionSetPeekabooState} for all next target symbols.

        Each child's ``.quotient`` and ``.remainder`` are computed on demand.
        Forces the BFS to run so ValueError is raised early for non-TD FSTs.
        """
        if '_children' not in self.__dict__:
            self._ensure_bfs()  # detect non-TD before creating children
            self._children = {y: self >> y for y in self.target_alphabet}
        return self._children

    def __rshift__(self, y):
        """Advance by one target symbol.  Non-incremental: rebuilds from scratch."""
        assert y in self.target_alphabet, repr(y)
        return PositionSetPeekabooState(
            self.fst, self.target + (y,), parent=self, univ=self._univ,
        )
