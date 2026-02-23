"""
Bridge between the Python FST/FSA classes and the OpenFST C++ backend.

Mirrors rust_bridge.py.  Uses _openfst_decomp Cython extension for
the underlying C++ computations.

Usage:
    from transduction.openfst_bridge import OpenFstDecomp
    result = OpenFstDecomp(fst, target)
    Q = result.quotient   # FSA
    R = result.remainder  # FSA
"""

from transduction.fsa import FSA, EPSILON
from transduction.base import DecompositionResult, IncrementalDecomposition
from transduction.rust_bridge import to_rust_fst
from functools import cached_property


# Reuse the same EPSILON convention as rust_bridge.py
RUST_EPSILON = 2**32 - 1  # u32::MAX == INTERNAL_EPSILON in C++


def _to_openfst_fst(fst):
    """Convert a Python FST to a C++ FstData object.

    Reuses to_rust_fst() symbol mapping (same Integerizer logic).
    Returns (fst_data, sym_map, state_map) where fst_data is a PyFstData.
    """
    import _openfst_decomp

    # Use rust_bridge's to_rust_fst to get the parallel arrays + mappings.
    # We don't actually need the RustFst object, just the arrays.
    from transduction.util import Integerizer

    sym_map = Integerizer()
    state_map = Integerizer()

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

    fst_data = _openfst_decomp.PyFstData(
        num_states,
        start_states,
        final_states,
        arc_src,
        arc_in,
        arc_out,
        arc_dst,
        source_alphabet,
    )

    return fst_data, sym_map, state_map


def _to_python_fsa(openfst_fsa, sym_map):
    """Convert a PyOpenFstFsa back to a Python FSA.

    Args:
        openfst_fsa: A PyOpenFstFsa object from the C++ module.
        sym_map: The Integerizer used when building the FstData.

    Returns:
        An FSA object.
    """
    fsa = FSA()
    inv = {v: k for k, v in sym_map.items()}

    for s in openfst_fsa.start_states():
        fsa.add_start(s)

    for s in openfst_fsa.final_states():
        fsa.add_stop(s)

    src, lbl, dst = openfst_fsa.arcs()
    for s, a, d in zip(src, lbl, dst):
        fsa.add_arc(s, inv[a], d)

    return fsa


class OpenFstDecomp(DecompositionResult):
    """Drop-in replacement for NonrecursiveDFADecomp / RustDecomp using the OpenFST backend."""

    def __init__(self, fst, target, minimize=False):
        import _openfst_decomp

        self.fst = fst
        self.target = tuple(target)
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        fst_data, sym_map, _state_map = _to_openfst_fst(fst)

        target_u32 = [sym_map(y) for y in target]

        result = _openfst_decomp.openfst_decompose(fst_data, target_u32, minimize=minimize)

        self.quotient = _to_python_fsa(result.quotient, sym_map)
        self.remainder = _to_python_fsa(result.remainder, sym_map)


class OpenFstDirtyPeekaboo(DecompositionResult):
    """OpenFST-backed incremental peekaboo decomposition.

    Mirrors RustDirtyPeekaboo interface.
    """

    def __init__(self, fst, target=(), *, minimize=False, _openfst_state=None,
                 _parent=None, _symbol=None):
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
        if _openfst_state is not None:
            self._openfst_state = _openfst_state
        else:
            import _openfst_decomp
            fst_data, sym_map, _ = _to_openfst_fst(fst)
            self._openfst_state = (_openfst_decomp.PyDirtyPeekaboo(fst_data),
                                   sym_map)

    def __rshift__(self, y):
        return self.decompose_next()[y]

    def decompose_next(self):
        state, sym_map = self._openfst_state
        target_u32 = [sym_map(y) for y in self.target]
        result = state.decompose(target_u32, self._minimize)

        output = {}
        for y in self.target_alphabet:
            y_u32 = sym_map(y)
            pair = result.get(y_u32)
            if pair is not None:
                q_fsa = _to_python_fsa(pair[0], sym_map)
                r_fsa = _to_python_fsa(pair[1], sym_map)
            else:
                q_fsa = FSA()
                r_fsa = FSA()
            child = OpenFstDirtyPeekaboo(
                self.fst, self.target + (y,),
                minimize=self._minimize,
                _openfst_state=self._openfst_state,
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
        assert parent is not None, "Root OpenFstDirtyPeekaboo has no quotient/remainder"
        state, sym_map = self._openfst_state
        parent_target_u32 = [sym_map(y) for y in parent.target]
        result = state.decompose(parent_target_u32, self._minimize)
        y_u32 = sym_map(self._symbol)
        pair = result.get(y_u32)
        if pair is not None:
            return (_to_python_fsa(pair[0], sym_map),
                    _to_python_fsa(pair[1], sym_map))
        return (FSA(), FSA())

    @property
    def quotient(self):
        return self._qr[0]

    @property
    def remainder(self):
        return self._qr[1]


class OpenFstLazyPeekabooDFA:
    """OpenFST-backed lazy peekaboo DFA for FusedTransducedLM.

    Same API as RustLazyPeekabooDFA — drop-in replacement via helper="openfst".
    """

    def __init__(self, fst):
        import _openfst_decomp

        fst_data, sym_map, state_map = _to_openfst_fst(fst)
        self._helper = _openfst_decomp.PyLazyPeekabooDFA(fst_data)
        self._fst_data = fst_data
        self._sym_map = sym_map
        self._state_map = state_map

    def new_step(self, target_u32):
        self._helper.new_step(target_u32)

    def start_ids(self):
        return self._helper.start_ids()

    def arcs(self, sid):
        return self._helper.arcs(sid)

    def run(self, source_path):
        return self._helper.run(source_path)

    def classify(self, sid):
        return self._helper.classify(sid)

    def expand_batch(self, sids):
        return self._helper.expand_batch(sids)

    def idx_to_sym_map(self):
        return self._helper.idx_to_sym_map()
