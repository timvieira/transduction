"""Cached decomposer and incremental peekaboo for FST decomposition."""

from transduction.fsa import EPSILON
from transduction.peekaboo_recursive import Peekaboo, PeekabooState, _trimmed_fsa

# Use Rust decomposition for better performance
try:
    from transduction.rust_bridge import RustDecomp, to_rust_fst, to_python_fsa
    import transduction_core
    HAS_RUST = True
except ImportError:
    from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp as RustDecomp
    HAS_RUST = False


class CachedDecomposer:
    """Caches the Rust FST to avoid rebuilding it for each decomposition.

    RustDecomp calls to_rust_fst() on every invocation (~120ms each).
    This class builds the Rust FST once and reuses it for:
    - decompose(target): single-target decomposition
    - peekaboo(target): all-symbol decomposition in one Rust call
    """

    def __init__(self, fst):
        self.fst = fst
        self.target_alphabet = fst.B - {EPSILON}
        if HAS_RUST:
            self._rust_fst, self._sym_map, self._state_map = to_rust_fst(fst)

    def decompose(self, target):
        """Decompose for a single target. Returns (Q_fsa, R_fsa)."""
        if HAS_RUST:
            target_u32 = [self._sym_map(y) for y in target]
            result = transduction_core.rust_decompose(self._rust_fst, target_u32)
            Q = to_python_fsa(result.quotient, self._sym_map)
            R = to_python_fsa(result.remainder, self._sym_map)
            return Q, R
        else:
            decomp = RustDecomp(self.fst, target)
            return decomp.quotient, decomp.remainder

    def peekaboo(self, target):
        """Compute per-symbol decomposition for all next symbols after target.

        Returns a lazy result where .get(y) -> (Q_fsa, R_fsa) or None.
        The Rust call happens once; Python FSA conversion is deferred until .get().
        """
        if HAS_RUST:
            target_u32 = [self._sym_map(y) for y in target]
            rust_result = transduction_core.rust_peekaboo(self._rust_fst, target_u32)
            return _LazyPeekabooResult(rust_result, self._sym_map)
        else:
            return _FallbackPeekaboo(self.fst, target)


class _LazyPeekabooResult:
    """Lazy per-symbol access to Rust peekaboo result.

    Only converts Rust FSAs to Python when .get(y) is called,
    so symbols skipped by early stopping never pay conversion cost.
    """

    def __init__(self, rust_result, sym_map):
        self._result = rust_result
        self._sym_map = sym_map
        self._cache = {}

    def get(self, y):
        """Get (Q_fsa, R_fsa) for symbol y, or None if not reachable."""
        if y in self._cache:
            return self._cache[y]
        y_u32 = self._sym_map(y)
        q_rust = self._result.quotient(y_u32)
        r_rust = self._result.remainder(y_u32)
        if q_rust is not None and r_rust is not None:
            Q = to_python_fsa(q_rust, self._sym_map)
            R = to_python_fsa(r_rust, self._sym_map)
            val = (Q, R)
        else:
            val = None
        self._cache[y] = val
        return val


class _FallbackPeekaboo:
    """Fallback peekaboo for when Rust is not available."""

    def __init__(self, fst, target):
        self._fst = fst
        self._target = target
        self._cache = {}

    def get(self, y):
        if y in self._cache:
            return self._cache[y]
        decomp = RustDecomp(self._fst, self._target + (y,))
        val = (decomp.quotient, decomp.remainder)
        self._cache[y] = val
        return val


class IncrementalPeekaboo:
    """Incremental recursive peekaboo that maintains state across benchmark steps.

    Instead of recomputing the full target chain from scratch at each step
    (quadratic cost), this advances one symbol at a time, resuming from
    the previous step's boundary states (linear cost).

    Usage:
        peek = IncrementalPeekaboo(fst)
        # step 0: target = ()
        qr = peek.get('65')  # get (Q, R) for next symbol '65'
        peek.step(output[0]) # advance to target = (output[0],)
        # step 1: target = (output[0],)
        qr = peek.get('65')  # incremental, not from scratch
    """

    def __init__(self, fst):
        self._peekaboo = Peekaboo(fst)
        self._state = PeekabooState(fst, (), parent=None, univ=self._peekaboo._univ)
        self._merged_incoming = dict(self._state.incoming)
        self.target_alphabet = fst.B - {EPSILON}

    def step(self, y):
        """Extend target by one symbol. Only expands boundary region."""
        self._state = self._state >> y
        for state, arcs in self._state.incoming.items():
            if state in self._merged_incoming:
                self._merged_incoming[state] |= arcs
            else:
                self._merged_incoming[state] = set(arcs)

    def get(self, y):
        """Get (Q_fsa, R_fsa) for next symbol y, or None if not reachable."""
        d = self._state.decomp.get(y)
        if d is None or (not d.quotient and not d.remainder):
            return None
        start_states = set(self._state.dfa.start())
        q = _trimmed_fsa(start_states, d.quotient, self._merged_incoming)
        r = _trimmed_fsa(start_states, d.remainder, self._merged_incoming)
        return (q, r)
