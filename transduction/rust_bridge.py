"""
Bridge between the Python FST/FSA classes and the Rust `transduction_core` module.

Usage:
    from transduction.rust_bridge import RustDecomp
    result = RustDecomp(fst, target)
    Q = result.quotient   # FSA
    R = result.remainder  # FSA
"""

from transduction.fsa import FSA, EPSILON
from arsenal import Integerizer


def to_rust_fst(fst):
    """Convert a Python FST to a Rust RustFst. Call once per FST, cache the result.

    Returns (rust_fst, sym_map) where sym_map is an Integerizer mapping
    Python symbols to u32 IDs.
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
    start_states = [state_map(s) for s in fst.I]
    final_states = [state_map(s) for s in fst.F]

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


class RustDecomp:
    """Drop-in replacement for NonrecursiveDFADecomp using the Rust backend."""

    def __init__(self, fst, target):
        import transduction_core

        self.fst = fst
        self.target = target
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        assert set(target) <= self.target_alphabet

        rust_fst, sym_map, state_map = to_rust_fst(fst)

        RUST_EPSILON = 2**32 - 1
        target_u32 = [sym_map(y) for y in target]

        result = transduction_core.rust_decompose(rust_fst, target_u32)

        self.quotient = to_python_fsa(result.quotient, sym_map)
        self.remainder = to_python_fsa(result.remainder, sym_map)
