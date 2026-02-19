"""
Bridge between the Python FST/FSA classes and the Rust `transduction_core` module.

Usage:
    from transduction.rust_bridge import RustDecomp
    result = RustDecomp(fst, target)
    Q = result.quotient   # FSA
    R = result.remainder  # FSA
"""

from transduction.fsa import FSA, EPSILON
from transduction.base import DecompositionResult, IncrementalDecomposition
from transduction.peekaboo_incremental import _trimmed_fsa
from transduction.util import Integerizer
from functools import cached_property


def to_rust_fst(fst):
    """Convert a Python FST to a Rust RustFst.  Call once per FST, cache the result.

    Returns:
        rust_fst: The Rust-side FST object (``transduction_core.RustFst``).
        sym_map: Integerizer mapping Python symbols to u32 IDs (shared by
            input and output labels).  EPSILON is mapped to ``u32::MAX``
            (0xFFFFFFFF) to match the Rust convention.
        state_map: Integerizer mapping Python state objects to contiguous u32 IDs.
    """
    import transduction_core

    sym_map = Integerizer()
    state_map = Integerizer()

    # Reserve EPSILON mapping: Python '' -> Rust u32::MAX
    RUST_EPSILON = 2**32 - 1  # u32::MAX

    # Renumber states
    for s in fst.states:
        state_map(s)

    num_states = len(state_map)
    start_states = [state_map(s) for s in fst.start]
    final_states = [state_map(s) for s in fst.stop]

    arc_src = []
    arc_in = []
    arc_out = []
    arc_dst = []

    for i in fst.states:
        for x, y, j in fst.arcs(i):
            arc_src.append(state_map(i))
            arc_in.append(RUST_EPSILON if x == EPSILON else sym_map(x))
            arc_out.append(RUST_EPSILON if y == EPSILON else sym_map(y))
            arc_dst.append(state_map(j))

    source_alphabet = [sym_map(a) for a in fst.A if a != EPSILON]

    rust_fst = transduction_core.RustFst(
        num_states,
        start_states,
        final_states,
        arc_src,
        arc_in,
        arc_out,
        arc_dst,
        source_alphabet,
    )

    return rust_fst, sym_map, state_map


def to_python_fsa(rust_fsa, sym_map):
    """Convert a Rust RustFsa back to a Python FSA.

    Args:
        rust_fsa: A RustFsa object from the Rust module.
        sym_map: The Integerizer used when building the RustFst.

    Returns:
        An FSA object.
    """
    fsa = FSA()
    inv = {v: k for k, v in sym_map.items()}

    for s in rust_fsa.start_states():
        fsa.add_start(s)

    for s in rust_fsa.final_states():
        fsa.add_stop(s)

    src, lbl, dst = rust_fsa.arcs()
    for s, a, d in zip(src, lbl, dst):
        fsa.add_arc(s, inv[a], d)

    return fsa


class RustDecomp(DecompositionResult):
    """Drop-in replacement for NonrecursiveDFADecomp using the Rust backend."""

    def __init__(self, fst, target, minimize=False):
        import transduction_core

        self.fst = fst
        self.target = tuple(target)
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        rust_fst, sym_map, _state_map = to_rust_fst(fst)

        target_u32 = [sym_map(y) for y in target]

        result = transduction_core.rust_decompose(rust_fst, target_u32, minimize=minimize)

        self.quotient = to_python_fsa(result.quotient, sym_map)
        self.remainder = to_python_fsa(result.remainder, sym_map)


class RustDirtyState(IncrementalDecomposition):
    """Rust-backed dirty-state incremental decomposition.

    Persists everything and only re-BFS from dirty DFA states whose NFA sets
    contain frontier elements.
    """

    def __init__(self, fst, target=(), *, minimize=False, _rust_state=None):
        """Create a dirty-state incremental decomposition.

        Args:
            fst: Python FST instance.
            target: Target prefix (iterable of symbols).
            minimize: If True, minimize the Q/R FSAs before returning.
            _rust_state: Internal; shared Rust state for incremental reuse.
        """
        self.fst = fst
        self.target = tuple(target)
        self.target_alphabet = fst.B - {EPSILON}
        self._minimize = minimize
        oov = set(self.target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")
        if _rust_state is not None:
            self._rust_state = _rust_state
        else:
            import transduction_core
            rust_fst, sym_map, _ = to_rust_fst(fst)
            self._rust_state = (transduction_core.RustDirtyStateDecomp(rust_fst),
                                sym_map)

    @cached_property
    def _qr(self):
        """Lazily compute and cache the (quotient, remainder) FSA pair."""
        if hasattr(self, '_precomputed_qr'):
            return self._precomputed_qr
        state, sym_map = self._rust_state
        target_u32 = [sym_map(y) for y in self.target]
        state.decompose(target_u32)
        q_rust = state.quotient(self._minimize)
        r_rust = state.remainder(self._minimize)
        return (to_python_fsa(q_rust, sym_map),
                to_python_fsa(r_rust, sym_map))

    @property
    def quotient(self):
        """FSA accepting source strings in the quotient Q(target)."""
        return self._qr[0]

    @property
    def remainder(self):
        """FSA accepting source strings in the remainder R(target)."""
        return self._qr[1]

    def __rshift__(self, y):
        """Extend the target by symbol ``y``, returning a new RustDirtyState."""
        return RustDirtyState(
            self.fst, self.target + (y,),
            minimize=self._minimize,
            _rust_state=self._rust_state,
        )

    def decompose_next(self):
        """Decompose for every next target symbol in one batch.

        Returns a dict ``{y: RustDirtyState}`` with pre-computed Q/R FSAs.
        """
        state, sym_map = self._rust_state
        target_u32 = [sym_map(y) for y in self.target]
        # Advance Rust state to current target first
        state.decompose(target_u32)
        output_u32 = [sym_map(y) for y in (self.fst.B - {EPSILON})]
        result = state.decompose_next(target_u32, output_u32)
        output = {}
        for y in (self.fst.B - {EPSILON}):
            y_u32 = sym_map(y)
            pair = result.get(y_u32)
            if pair is not None:
                q_rust, r_rust = pair
                q_fsa = to_python_fsa(q_rust, sym_map)
                r_fsa = to_python_fsa(r_rust, sym_map)
            else:
                q_fsa = FSA()
                r_fsa = FSA()
            output[y] = RustDirtyState(
                self.fst, self.target + (y,),
                minimize=self._minimize,
                _rust_state=self._rust_state,
            )
            # Inject pre-computed Q/R to avoid re-computation
            output[y]._precomputed_qr = (q_fsa, r_fsa)
        return output


class RustDirtyPeekaboo(DecompositionResult):
    """Rust-backed incremental peekaboo decomposition.

    Persists peekaboo BFS state across calls; on prefix extension, only
    runs the new step(s) instead of rebuilding from scratch.
    """

    def __init__(self, fst, target=(), *, minimize=False, _rust_state=None,
                 _parent=None, _symbol=None):
        """Create a Rust-backed incremental peekaboo decomposition.

        Args:
            fst: Python FST instance.
            target: Target prefix (iterable of symbols).
            minimize: If True, minimize the Q/R FSAs before returning.
            _rust_state: Internal; shared Rust peekaboo state for reuse.
            _parent: Internal; parent node for lazy Q/R computation.
            _symbol: Internal; the symbol that extended the parent to this node.
        """
        self.fst = fst
        target = tuple(target)
        self.target = target
        self.target_alphabet = fst.B - {EPSILON}
        self._minimize = minimize
        self._parent = _parent
        self._symbol = _symbol
        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")
        if _rust_state is not None:
            self._rust_state = _rust_state
        else:
            import transduction_core
            rust_fst, sym_map, _ = to_rust_fst(fst)
            self._rust_state = (transduction_core.RustDirtyPeekabooDecomp(rust_fst),
                                sym_map)

    def __rshift__(self, y):
        """Extend the target by symbol ``y``, returning a new RustDirtyPeekaboo."""
        return self.decompose_next()[y]

    def decompose_next(self):
        """Decompose for every next target symbol in one batched peekaboo pass.

        Returns a dict ``{y: RustDirtyPeekaboo}`` with pre-computed Q/R FSAs.
        """
        state, sym_map = self._rust_state
        target_u32 = [sym_map(y) for y in self.target]
        result = state.decompose(target_u32, self._minimize)

        output = {}
        for y in self.target_alphabet:
            y_u32 = sym_map(y)
            q_rust = result.quotient(y_u32)
            r_rust = result.remainder(y_u32)
            if q_rust is not None and r_rust is not None:
                q_fsa = to_python_fsa(q_rust, sym_map)
                r_fsa = to_python_fsa(r_rust, sym_map)
            else:
                q_fsa = FSA()
                r_fsa = FSA()
            child = RustDirtyPeekaboo(
                self.fst, self.target + (y,),
                minimize=self._minimize,
                _rust_state=self._rust_state,
                _parent=self, _symbol=y,
            )
            child._precomputed_qr = (q_fsa, r_fsa)
            output[y] = child
        return output

    @cached_property
    def _qr(self):
        """Lazily compute and cache the (quotient, remainder) FSA pair."""
        if hasattr(self, '_precomputed_qr'):
            return self._precomputed_qr
        parent = self._parent
        assert parent is not None, "Root RustDirtyPeekaboo has no quotient/remainder"
        # Compute via parent's decompose_next
        state, sym_map = self._rust_state
        parent_target_u32 = [sym_map(y) for y in parent.target]
        result = state.decompose(parent_target_u32, self._minimize)
        y_u32 = sym_map(self._symbol)
        q_rust = result.quotient(y_u32)
        r_rust = result.remainder(y_u32)
        if q_rust is not None and r_rust is not None:
            return (to_python_fsa(q_rust, sym_map), to_python_fsa(r_rust, sym_map))
        return (FSA(), FSA())

    @property
    def quotient(self):
        """FSA accepting source strings in the quotient Q(target)."""
        return self._qr[0]

    @property
    def remainder(self):
        """FSA accepting source strings in the remainder R(target)."""
        return self._qr[1]


# ---------------------------------------------------------------------------
# RustPeekabooState: adapter for TransducedLM beam search
# ---------------------------------------------------------------------------

class _RustDFAAdapter:
    """Wraps Rust DFA arc queries for beam search in TransducedLM."""

    __slots__ = ('_rust_decomp', '_inv', '_fwd', '_start_id')

    def __init__(self, rust_decomp, inv_sym_map, start_id):
        self._rust_decomp = rust_decomp
        self._inv = inv_sym_map
        self._fwd = {v: k for k, v in inv_sym_map.items()}
        self._start_id = start_id

    def arcs(self, state_id):
        raw = self._rust_decomp.arcs_for(state_id)
        return [(self._inv[lbl], dst) for lbl, dst in raw]

    def run(self, source_path):
        """Run a source path from start, returning the reached DFA state (or None for dead)."""
        path_u32 = [self._fwd[x] for x in source_path]
        return self._rust_decomp.run(path_u32)

    def start(self):
        return [self._start_id]


class _RustPeekabooUniv:
    """No-op universality placeholder — Rust handles universality internally."""
    def __init__(self, fst):
        pass


class RustPeekabooState:
    """Adapter matching PeekabooState interface, backed by Rust DirtyPeekaboo.

    Uses decompose_for_beam() to get DFA state IDs directly instead of
    constructing FSA objects. The beam search in TransducedLM walks the
    DFA via arcs_for() using interned u32 state IDs.
    """

    # Lazy attributes computed by _ensure_bfs()
    _LAZY_BFS_ATTRS = frozenset({
        'decomp', 'dfa', 'resume_frontiers', 'preimage_stops',
    })

    def __init__(self, fst, target=(), parent=None, *, univ=None,
                 _rust_state=None):
        self.fst = fst
        self.target = tuple(target) if not isinstance(target, tuple) else target
        self.target_alphabet = fst.B - {EPSILON}
        self.source_alphabet = fst.A - {EPSILON}
        oov = set(self.target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        if _rust_state is not None:
            self._rust_state = _rust_state
        else:
            import transduction_core
            rust_fst, sym_map, state_map = to_rust_fst(fst)
            rust_decomp = transduction_core.RustDirtyPeekabooDecomp(rust_fst)
            self._rust_state = (rust_decomp, sym_map, state_map)

        self._bfs_done = False

    def __getattr__(self, name):
        if name in RustPeekabooState._LAZY_BFS_ATTRS:
            self._ensure_bfs()
            return self.__dict__[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute {name!r}"
        )

    def _ensure_bfs(self):
        if self._bfs_done:
            return

        rust_decomp, sym_map, *_ = self._rust_state
        inv_sym_map = {v: k for k, v in sym_map.items()}

        target_u32 = [sym_map(y) for y in self.target]
        view = rust_decomp.decompose_for_beam(target_u32)

        # Build decomp: {py_sym: DecompositionResult(quotient=set, remainder=set)}
        decomp = {}
        for sym_u32, sids in view.decomp_q():
            py_sym = inv_sym_map[sym_u32]
            if py_sym not in decomp:
                decomp[py_sym] = DecompositionResult(set(), set())
            decomp[py_sym].quotient.update(sids)
        for sym_u32, sids in view.decomp_r():
            py_sym = inv_sym_map[sym_u32]
            if py_sym not in decomp:
                decomp[py_sym] = DecompositionResult(set(), set())
            decomp[py_sym].remainder.update(sids)

        # Build resume_frontiers: {py_sym: set[u32]}
        resume_frontiers = {}
        for sym_u32, sids in view.resume_frontiers():
            py_sym = inv_sym_map[sym_u32]
            resume_frontiers[py_sym] = set(sids)

        # Build preimage_stops: set[u32]
        preimage_stops = set(view.preimage_stops)

        # Build DFA adapter
        dfa = _RustDFAAdapter(rust_decomp, inv_sym_map, view.start_id)

        self.decomp = decomp
        self.resume_frontiers = resume_frontiers
        self.preimage_stops = preimage_stops
        self.dfa = dfa
        self._bfs_done = True

    def decode_dfa_state(self, state_id):
        """Decode a Rust DFA state ID to its NFA constituents.

        Returns frozenset of (fst_state, buffer_tuple, truncated) matching
        the Python PeekabooLookaheadNFA representation.
        """
        rust_decomp = self._rust_state[0]
        sym_map = self._rust_state[1]
        state_map = self._rust_state[2]

        # Cache inverse maps (shared across all calls)
        if not hasattr(self, '_inv_maps'):
            idx_to_sym_raw = rust_decomp.idx_to_sym_map()
            inv_sym = {v: k for k, v in sym_map.items()}
            inv_state = {v: k for k, v in state_map.items()}
            self._inv_maps = (idx_to_sym_raw, inv_sym, inv_state)
        idx_to_sym_raw, inv_sym, inv_state = self._inv_maps

        NO_EXTRA = 0xFFFF
        raw = rust_decomp.decode_state(state_id)
        target = self.target

        result = set()
        for fst_state_u32, buf_len, extra_sym_idx, truncated in raw:
            py_fst_state = inv_state.get(fst_state_u32, fst_state_u32)
            if extra_sym_idx == NO_EXTRA:
                buf = target[:buf_len]
            else:
                sym_u32 = idx_to_sym_raw[extra_sym_idx]
                py_sym = inv_sym[sym_u32]
                buf = target[:buf_len - 1] + (py_sym,)
            result.add((py_fst_state, buf, truncated))

        return frozenset(result)

    def _collect_incoming(self):
        """Forward BFS from DFA start, collecting incoming (reverse) arcs.

        Returns {state: {(label, pred), ...}} matching the format used by
        _trimmed_fsa() in peekaboo_incremental.py.  Cached per state.
        """
        if hasattr(self, '_incoming_cache'):
            return self._incoming_cache
        from collections import deque
        dfa = self.dfa
        incoming = {}
        visited = set()
        worklist = deque(dfa.start())
        for s in dfa.start():
            visited.add(s)
        while worklist:
            state = worklist.popleft()
            for label, dest in dfa.arcs(state):
                incoming.setdefault(dest, set()).add((label, state))
                if dest not in visited:
                    visited.add(dest)
                    worklist.append(dest)
        self._incoming_cache = incoming
        return incoming

    def build_qr_fsa(self, y):
        """Build trimmed Q and R FSAs for target symbol y.

        Returns (quotient_fsa, remainder_fsa) where states are DFA state IDs
        matching the beam particle state IDs.  Uses forward BFS + backward
        trim through the beam DFA (same approach as the Python reference
        implementation in peekaboo_incremental.py).
        """
        self._ensure_bfs()
        d = self.decomp.get(y)
        if d is None:
            return FSA(), FSA()
        incoming = self._collect_incoming()
        get_incoming = lambda s: incoming.get(s, ())
        start_states = self.dfa.start()
        q_fsa = _trimmed_fsa(start_states, d.quotient, get_incoming)
        r_fsa = _trimmed_fsa(start_states, d.remainder, get_incoming)
        return q_fsa, r_fsa

    def __rshift__(self, y):
        return RustPeekabooState(
            self.fst, self.target + (y,),
            _rust_state=self._rust_state,
        )


# ---------------------------------------------------------------------------
# RustRhoDeterminize: rho-factored precover DFA
# ---------------------------------------------------------------------------

class RustRhoDeterminize:
    """Rust-backed rho-factored determinization of the PrecoverNFA.

    Mirrors Python's SymbolicLazyDeterminize + ExpandRho pipeline but runs
    the BFS subset construction and rho factoring in Rust.
    """

    def __init__(self, fst, target):
        import transduction_core

        self.fst = fst
        self.target = tuple(target)

        rust_fst, sym_map, _state_map = to_rust_fst(fst)
        target_u32 = [sym_map(y) for y in self.target]

        self._rust = transduction_core.rust_rho_determinize(rust_fst, target_u32)
        self._sym_map = sym_map

    @property
    def num_rho_arcs(self):
        return self._rust.num_rho_arcs()

    @property
    def num_explicit_arcs(self):
        return self._rust.num_explicit_arcs()

    @property
    def complete_states(self):
        return self._rust.complete_states()

    @property
    def total_ms(self):
        return self._rust.total_ms()

    @property
    def num_states(self):
        return self._rust.num_states()

    @property
    def total_arcs(self):
        return self.num_rho_arcs + self.num_explicit_arcs

    def expand(self):
        """Expand RHO arcs and return a Python FSA."""
        return to_python_fsa(self._rust.expand(), self._sym_map)
