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
        self.target = target
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        rust_fst, sym_map, state_map = to_rust_fst(fst)

        target_u32 = [sym_map(y) for y in target]

        result = transduction_core.rust_decompose(rust_fst, target_u32, minimize=minimize)

        self.quotient = to_python_fsa(result.quotient, sym_map)
        self.remainder = to_python_fsa(result.remainder, sym_map)


class RustDirtyState(IncrementalDecomposition):
    """Rust-backed dirty-state incremental decomposition.

    Persists everything and only re-BFS from dirty DFA states whose NFA sets
    contain frontier elements.
    """

    def __init__(self, fst, target='', *, minimize=False, _rust_state=None):
        self.fst = fst
        self.target = target
        self._minimize = minimize
        if _rust_state is not None:
            self._rust_state = _rust_state
        else:
            import transduction_core
            rust_fst, sym_map, _ = to_rust_fst(fst)
            self._rust_state = (transduction_core.RustDirtyStateDecomp(rust_fst),
                                sym_map)

    @cached_property
    def _qr(self):
        # Use precomputed Q/R from decompose_next() if available
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
        return self._qr[0]

    @property
    def remainder(self):
        return self._qr[1]

    def __rshift__(self, y):
        return RustDirtyState(
            self.fst, self.target + y,
            minimize=self._minimize,
            _rust_state=self._rust_state,
        )

    def decompose_next(self):
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
                self.fst, self.target + y,
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

    def __init__(self, fst, target='', *, minimize=False, _rust_state=None,
                 _parent=None, _symbol=None):
        self.fst = fst
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

    def decompose_next(self):
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
                self.fst, self.target + y,
                minimize=self._minimize,
                _rust_state=self._rust_state,
                _parent=self, _symbol=y,
            )
            child._precomputed_qr = (q_fsa, r_fsa)
            output[y] = child
        return output

    @cached_property
    def _qr(self):
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
        return self._qr[0]

    @property
    def remainder(self):
        return self._qr[1]
