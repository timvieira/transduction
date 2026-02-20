"""Pynini-based FST decomposition operations.

Provides an alternative implementation of precover, quotient, and remainder
computation using pynini/OpenFST composition and projection. This gives a
clean, readable reference implementation using standard WFST operations.

Mathematical operations for FST f : X* -> Y* and target y:

  P(y) = compose(f, y·Sigma*).project('input').optimize()
         The precover: all source strings whose transduction starts with y.

  Q(y) = P(y) with only universal finals, arcs from universal states pruned.
         The quotient: source strings after which any continuation stays in P(y).

  R(y) = P(y) - Q(y)·Sigma_in*
         The remainder: precover minus the quotient cylinder.

Requires: pynini (pip install pynini)
"""

import pynini
from collections import deque

from transduction.fst import FST, EPSILON
from transduction.fsa import FSA
from transduction.base import DecompositionResult


def native_fst_to_pynini(fst):
    """Convert a native FST to a pynini Fst.

    Returns (pynini_fst, in_sym_table, out_sym_table, in_label_map, out_label_map).
    Handles multiple start states via a super-start with epsilon arcs.
    Maps native EPSILON='' to pynini label 0 (epsilon).
    """
    # Collect input and output alphabets (excluding epsilon)
    in_syms = sorted(fst.A - {EPSILON}, key=repr)
    out_syms = sorted(fst.B - {EPSILON}, key=repr)

    # Build symbol tables
    in_sym_table = pynini.SymbolTable()
    in_sym_table.add_symbol("<eps>", 0)
    in_label_map = {EPSILON: 0}
    for i, sym in enumerate(in_syms, 1):
        in_sym_table.add_symbol(repr(sym), i)
        in_label_map[sym] = i

    out_sym_table = pynini.SymbolTable()
    out_sym_table.add_symbol("<eps>", 0)
    out_label_map = {EPSILON: 0}
    for i, sym in enumerate(out_syms, 1):
        out_sym_table.add_symbol(repr(sym), i)
        out_label_map[sym] = i

    # Map native states to integer IDs
    state_list = sorted(fst.states, key=repr)
    state_map = {s: i for i, s in enumerate(state_list)}

    # If multiple start states, reserve an extra state as super-start
    starts = sorted(fst.start, key=repr)
    needs_super_start = len(starts) > 1

    # Build pynini FST
    pfst = pynini.Fst()

    # Add all states
    n_states = len(state_list) + (1 if needs_super_start else 0)
    for _ in range(n_states):
        pfst.add_state()

    # Set start state
    if needs_super_start:
        super_start_id = len(state_list)
        pfst.set_start(super_start_id)
        for s in starts:
            pfst.add_arc(super_start_id,
                         pynini.Arc(0, 0, 0, state_map[s]))
    else:
        pfst.set_start(state_map[starts[0]])

    # Set final states
    for s in fst.stop:
        pfst.set_final(state_map[s])

    # Add arcs
    for s in fst.states:
        for a, b, t in fst.arcs(s):
            ilabel = in_label_map[a]
            olabel = out_label_map[b]
            pfst.add_arc(state_map[s],
                         pynini.Arc(ilabel, olabel, 0, state_map[t]))

    pfst.set_input_symbols(in_sym_table)
    pfst.set_output_symbols(out_sym_table)

    return pfst, in_sym_table, out_sym_table, in_label_map, out_label_map


def pynini_acceptor_to_native_fsa(dfa, label_map_inv):
    """Convert a pynini acceptor (DFA) to a native FSA.

    Args:
        dfa: pynini Fst (acceptor/DFA)
        label_map_inv: dict mapping pynini label IDs to native symbols
    """
    zero = pynini.Weight.zero(dfa.weight_type())

    fsa = FSA()
    start = dfa.start()
    if start == pynini.NO_STATE_ID:
        return fsa

    fsa.add_start(start)
    for state in dfa.states():
        if dfa.final(state) != zero:
            fsa.add_stop(state)
        for arc in dfa.arcs(state):
            label = label_map_inv.get(arc.ilabel, arc.ilabel)
            if label == 0 or label == EPSILON:
                label = EPSILON
            fsa.add_arc(state, label, arc.nextstate)

    return fsa


def _build_prefix_filter(target, out_label_map, out_sym_table):
    """Build a pynini acceptor for y·Sigma_out*.

    Accepts any string that starts with the target sequence y,
    followed by any string over the output alphabet.
    """
    n = len(target)
    filt = pynini.Fst()

    for _ in range(n + 1):
        filt.add_state()

    filt.set_start(0)
    filt.set_final(n)

    # Chain arcs for target symbols
    for i, sym in enumerate(target):
        label = out_label_map[sym]
        filt.add_arc(i, pynini.Arc(label, label, 0, i + 1))

    # Self-loops on the final state for all output symbols (Sigma_out*)
    for sym, label in out_label_map.items():
        if sym != EPSILON:
            filt.add_arc(n, pynini.Arc(label, label, 0, n))

    filt.set_input_symbols(out_sym_table)
    filt.set_output_symbols(out_sym_table)

    return filt


def _sigma_star_acceptor(label_map, sym_table):
    """Build a pynini acceptor for Sigma* over the given alphabet."""
    fsa = pynini.Fst()
    fsa.add_state()
    fsa.set_start(0)
    fsa.set_final(0)

    for sym, label in label_map.items():
        if sym != EPSILON:
            fsa.add_arc(0, pynini.Arc(label, label, 0, 0))

    fsa.set_input_symbols(sym_table)
    fsa.set_output_symbols(sym_table)

    return fsa


def _universal_states(dfa, input_alphabet_ids):
    """Find all universal states in a pynini DFA.

    A state s is universal iff all states reachable from s (including s itself)
    are final and have outgoing arcs for every symbol in input_alphabet_ids.

    Uses greatest fixpoint: start with all final+complete states as candidates,
    then iteratively remove states whose successors aren't all candidates.

    This is the pynini equivalent of UniversalityFilter._bfs_universal.
    """
    zero = pynini.Weight.zero(dfa.weight_type())

    # Build adjacency info: state -> {label: next_state}
    state_arcs = {}
    for state in dfa.states():
        arcs_by_label = {}
        for arc in dfa.arcs(state):
            arcs_by_label[arc.ilabel] = arc.nextstate
        state_arcs[state] = arcs_by_label

    input_alphabet_set = set(input_alphabet_ids)

    # Initialize candidates: all final states that are complete
    candidates = set()
    for state in dfa.states():
        if dfa.final(state) == zero:
            continue
        arcs = state_arcs.get(state, {})
        if input_alphabet_set <= arcs.keys():
            candidates.add(state)

    # Greatest fixpoint: iteratively remove non-universal states
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for state in candidates:
            arcs = state_arcs[state]
            for label in input_alphabet_ids:
                if arcs[label] not in candidates:
                    to_remove.add(state)
                    break
        if to_remove:
            candidates -= to_remove
            changed = True

    return candidates


def _run_dfa(dfa, label_sequence):
    """Run a pynini DFA on a sequence of labels.

    Returns the reached state ID or None if a transition is missing (dead).
    """
    state = dfa.start()
    if state == pynini.NO_STATE_ID:
        return None

    for label in label_sequence:
        found = False
        for arc in dfa.arcs(state):
            if arc.ilabel == label:
                state = arc.nextstate
                found = True
                break
        if not found:
            return None

    return state


class PyniniDecomposition:
    """Pynini-based FST decomposition.

    Computes precover P(y), quotient Q(y), and remainder R(y) using
    pynini composition, projection, and difference operations.

    Usage::

        pd = PyniniDecomposition(fst)
        P = pd.precover('xy')           # pynini DFA
        Q = pd.quotient('xy')           # pynini DFA
        R = pd.remainder('xy')          # pynini DFA

        pd.is_in_precover('xy', 'ab')   # bool
        pd.is_in_quotient('xy', 'ab')   # bool
        pd.is_in_remainder('xy', 'ab')  # bool
    """

    def __init__(self, fst):
        self.native_fst = fst
        (self.pfst, self.in_sym_table, self.out_sym_table,
         self.in_label_map, self.out_label_map) = native_fst_to_pynini(fst)

        self.input_alphabet_ids = sorted(
            self.in_label_map[s]
            for s in fst.A - {EPSILON}
        )

        # Inverse maps: pynini label ID -> native symbol
        self.in_label_inv = {v: k for k, v in self.in_label_map.items() if k != EPSILON}
        self.out_label_inv = {v: k for k, v in self.out_label_map.items() if k != EPSILON}

        # Precompute sigma_in_star for remainder computation
        self._sigma_in_star = _sigma_star_acceptor(self.in_label_map, self.in_sym_table)

        # Caches
        self._precover_cache = {}
        self._universal_cache = {}

    def precover(self, target):
        """Compute P(y) as a pynini DFA over the input alphabet.

        P(y) = project_input(compose(f, y·Sigma_out*)).optimize()
        """
        target = tuple(target)
        if target in self._precover_cache:
            return self._precover_cache[target]

        prefix_filter = _build_prefix_filter(
            target, self.out_label_map, self.out_sym_table
        )
        composed = pynini.compose(self.pfst, prefix_filter)
        projected = composed.project('input')
        # rmepsilon + determinize + minimize
        result = projected.optimize()

        self._precover_cache[target] = result
        return result

    def _get_universal(self, target):
        """Get universal states for precover(target), with caching."""
        target = tuple(target)
        if target in self._universal_cache:
            return self._universal_cache[target]
        dfa = self.precover(target)
        universal = _universal_states(dfa, self.input_alphabet_ids)
        self._universal_cache[target] = universal
        return universal

    def quotient(self, target):
        """Compute Q(y) as a pynini DFA.

        Q(y) = P(y) with only universal states as finals, arcs from
        universal states pruned.
        """
        target = tuple(target)
        precover_dfa = self.precover(target)
        universal = self._get_universal(target)

        # Build Q: same structure but only universal finals, pruned arcs
        q_fst = pynini.Fst()
        for _ in precover_dfa.states():
            q_fst.add_state()

        start = precover_dfa.start()
        if start != pynini.NO_STATE_ID:
            q_fst.set_start(start)

        # Only universal states are final
        for state in precover_dfa.states():
            if state in universal:
                q_fst.set_final(state)

        # Copy arcs except those leaving universal states
        for state in precover_dfa.states():
            if state in universal:
                continue
            for arc in precover_dfa.arcs(state):
                q_fst.add_arc(state, pynini.Arc(
                    arc.ilabel, arc.olabel, arc.weight, arc.nextstate
                ))

        if precover_dfa.input_symbols():
            q_fst.set_input_symbols(precover_dfa.input_symbols())
        if precover_dfa.output_symbols():
            q_fst.set_output_symbols(precover_dfa.output_symbols())

        return q_fst

    def remainder(self, target):
        """Compute R(y) = P(y) - Q(y)·Sigma_in*.

        The remainder: precover minus the quotient cylinder.
        Uses pynini.difference() for clean set subtraction.
        """
        target = tuple(target)
        precover_dfa = self.precover(target)
        quotient_dfa = self.quotient(target)

        # Q(y)·Sigma_in*
        q_cylinder = pynini.concat(quotient_dfa, self._sigma_in_star).optimize()

        # R(y) = P(y) - Q(y)·Sigma_in*
        result = pynini.difference(precover_dfa, q_cylinder).optimize()
        return result

    def precover_as_native_fsa(self, target):
        """Return P(y) as a native FSA for comparison with reference."""
        return pynini_acceptor_to_native_fsa(
            self.precover(target), self.in_label_inv
        )

    def quotient_as_native_fsa(self, target):
        """Return Q(y) as a native FSA for comparison with reference."""
        return pynini_acceptor_to_native_fsa(
            self.quotient(target), self.in_label_inv
        )

    def remainder_as_native_fsa(self, target):
        """Return R(y) as a native FSA for comparison with reference."""
        return pynini_acceptor_to_native_fsa(
            self.remainder(target), self.in_label_inv
        )

    def _source_labels(self, source):
        """Convert a source symbol sequence to pynini label IDs."""
        return [self.in_label_map[s] for s in source]

    def is_in_precover(self, target, source):
        """Check if source string x is in P(y)."""
        target, source = tuple(target), tuple(source)
        dfa = self.precover(target)
        state = _run_dfa(dfa, self._source_labels(source))
        if state is None:
            return False
        return dfa.final(state) != pynini.Weight.zero(dfa.weight_type())

    def is_in_quotient(self, target, source):
        """Check if source string x is in Q(y).

        Runs x on the quotient DFA (which has arcs from universal states
        pruned), so strings that pass through a universal state before
        consuming all of x are correctly rejected.
        """
        target, source = tuple(target), tuple(source)
        dfa = self.quotient(target)
        state = _run_dfa(dfa, self._source_labels(source))
        if state is None:
            return False
        return dfa.final(state) != pynini.Weight.zero(dfa.weight_type())

    def is_in_remainder(self, target, source):
        """Check if source string x is in R(y)."""
        target, source = tuple(target), tuple(source)
        dfa = self.remainder(target)
        state = _run_dfa(dfa, self._source_labels(source))
        if state is None:
            return False
        return dfa.final(state) != pynini.Weight.zero(dfa.weight_type())

    def is_prefix_of_precover(self, target, source_prefix):
        """Check if source_prefix is a prefix of some string in P(y).

        Works because optimize() trims non-coaccessible states, so reaching
        any state means there's a path to an accepting state.
        """
        target, source_prefix = tuple(target), tuple(source_prefix)
        dfa = self.precover(target)
        state = _run_dfa(dfa, self._source_labels(source_prefix))
        return state is not None


# ─── Drop-in backend for existing decomposition algorithms ──────────

class PyniniPrecover:
    """Pynini-accelerated precover DFA builder.

    Caches the pynini FST conversion so it's done once per native FST.
    Each call to ``build_dfa(target)`` returns a native FSA (DFA) built
    via pynini composition, replacing the PrecoverNFA + powerset det path.
    """

    def __init__(self, fst):
        self._pd = PyniniDecomposition(fst)

    def build_dfa(self, target):
        """Return precover DFA P(y) as a native FSA."""
        return pynini_acceptor_to_native_fsa(
            self._pd.precover(target), self._pd.in_label_inv
        )


class PyniniNonrecursiveDecomp(DecompositionResult):
    """Drop-in replacement for NonrecursiveDFADecomp using pynini for the DFA.

    Uses pynini composition to build the precover DFA (the expensive step),
    then runs the same BFS universality check and Q/R split in Python.

    Conforms to the same interface as NonrecursiveDFADecomp:
      - Constructor: PyniniNonrecursiveDecomp(fst, target)
      - Properties: .quotient, .remainder (native FSA)
      - Operators: >> (extend target), decompose_next()

    For repeated use on the same FST, pass a shared PyniniPrecover to avoid
    redundant pynini FST conversion::

        backend = PyniniPrecover(fst)
        d1 = PyniniNonrecursiveDecomp(fst, target1, backend=backend)
        d2 = PyniniNonrecursiveDecomp(fst, target2, backend=backend)
    """

    # Class-level cache: native FST id -> PyniniPrecover
    _backend_cache = {}

    def __init__(self, fst, target, backend=None):
        self.fst = fst
        self.target = tuple(target)
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        oov = set(self.target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        # Get or create the pynini backend (cached per FST identity)
        if backend is not None:
            self._backend = backend
        else:
            fst_id = id(fst)
            if fst_id not in PyniniNonrecursiveDecomp._backend_cache:
                PyniniNonrecursiveDecomp._backend_cache[fst_id] = PyniniPrecover(fst)
            self._backend = PyniniNonrecursiveDecomp._backend_cache[fst_id]

        # Build precover DFA via pynini (the fast part)
        dfa = self._backend.build_dfa(self.target)

        # BFS with universality check (same as NonrecursiveDFADecomp)
        Q = FSA()
        R = FSA()

        worklist = deque()
        visited = set()

        for i in dfa.start:
            worklist.append(i)
            visited.add(i)
            Q.add_start(i)
            R.add_start(i)

        while worklist:
            i = worklist.popleft()

            if dfa.is_final(i):
                if _is_universal_native(dfa, i, self.source_alphabet):
                    Q.add_stop(i)
                    continue
                else:
                    R.add_stop(i)

            for a, j in dfa.arcs(i):
                if j not in visited:
                    worklist.append(j)
                    visited.add(j)
                Q.add_arc(i, a, j)
                R.add_arc(i, a, j)

        self.quotient = Q
        self.remainder = R

    def __rshift__(self, y):
        return PyniniNonrecursiveDecomp(self.fst, self.target + (y,),
                                        backend=self._backend)

    def decompose_next(self):
        return {y: self >> y for y in self.target_alphabet}


def _is_universal_native(dfa, state, source_alphabet):
    """BFS universality check on a native FSA DFA state."""
    visited = set()
    worklist = deque()
    visited.add(state)
    worklist.append(state)
    while worklist:
        i = worklist.popleft()
        if not dfa.is_final(i):
            return False
        dest = dict(dfa.arcs(i))
        for a in source_alphabet:
            if a not in dest:
                return False
            j = dest[a]
            if j not in visited:
                visited.add(j)
                worklist.append(j)
    return True
