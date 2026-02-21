"""
Token-level decomposition for BPE-like FSTs, plus position-set-quotiented
peekaboo helpers for TransducedLM / FusedTransducedLM integration.

Two capabilities:

1. **Standalone TokenDecompose** — full-target Q/R decomposition using
   position-set DFA states.  Only works for FSTs satisfying
   ``all_input_universal`` (BPE-like hub topology).

2. **TokenPeekabooHelper** — FusedTransducedLM helper that builds a
   peekaboo DFA with position-set quotienting.  Works for any
   token-decomposable FST (BPE and PTB).  DFA states that share the
   same "position key" (buffer-length profile) are merged, yielding a
   much smaller DFA (e.g., 35× compression on PTB).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from transduction.base import DecompositionResult
from transduction.fsa import FSA, EPSILON
from transduction.universality import check_all_input_universal
from transduction.util import Integerizer

# ---------------------------------------------------------------------------
# ByteTrie — shared by standalone and peekaboo paths
# ---------------------------------------------------------------------------

class ByteTrie:
    """Trie over byte sequences for fast prefix matching."""

    def __init__(self):
        self.children: list[dict[Any, int]] = [{}]
        self.completions: list[list[tuple[Any, int]]] = [[]]

    def insert(self, token_id: Any, byte_seq: tuple) -> None:
        node = 0
        for b in byte_seq:
            if b in self.children[node]:
                node = self.children[node][b]
            else:
                next_id = len(self.children)
                self.children.append({})
                self.completions.append([])
                self.children[node][b] = next_id
                node = next_id
        self.completions[node].append((token_id, len(byte_seq)))

    def matches_at(self, target: tuple, pos: int) -> list[tuple[Any, int]]:
        """Collect tokens matching target[pos:], including partial (beyond-target) matches."""
        result = []
        node = 0
        target_len = len(target)
        for i in range(pos, target_len):
            if target[i] in self.children[node]:
                node = self.children[node][target[i]]
                result.extend(self.completions[node])
            else:
                return result
        advance_cap = target_len - pos
        self._collect_subtree(node, advance_cap, result)
        return result

    def _collect_subtree(self, node: int, advance_cap: int,
                         result: list[tuple[Any, int]]) -> None:
        for _byte, child in self.children[node].items():
            for (tid, _byte_len) in self.completions[child]:
                result.append((tid, advance_cap))
            self._collect_subtree(child, advance_cap, result)


def extract_token_bytes(fst) -> list[tuple[Any, tuple]]:
    """Extract (token_id, byte_sequence) pairs from a BPE-like FST."""
    start_set = set(fst.start)
    tokens = []
    for start in fst.start:
        for a, b, j in fst.arcs(start):
            if a == EPSILON:
                continue
            token_id = a
            bytes_out: list = []
            if b != EPSILON:
                bytes_out.append(b)
            current = j
            while current not in start_set:
                found = False
                for a2, b2, j2 in fst.arcs(current):
                    if a2 == EPSILON:
                        if b2 != EPSILON:
                            bytes_out.append(b2)
                        current = j2
                        found = True
                        break
                if not found:
                    break
            tokens.append((token_id, tuple(bytes_out)))
    return tokens


def build_trie(fst) -> ByteTrie:
    """Build a ByteTrie from an FST's extracted tokens."""
    token_list = extract_token_bytes(fst)
    trie = ByteTrie()
    for token_id, byte_seq in token_list:
        if byte_seq:
            trie.insert(token_id, byte_seq)
    return trie


# ---------------------------------------------------------------------------
# Standalone TokenDecompose (BPE-only, all_input_universal required)
# ---------------------------------------------------------------------------

class TokenDecompose(DecompositionResult):
    """Token-level decomposition using position-set DFA states.

    For BPE-like FSTs where ``all_input_universal`` holds.  DFA states are
    frozensets of positions {0..target_len} instead of the O(|fst_states| *
    target_len) NFA state space of the generic approach.
    """

    def __init__(self, fst, target):
        assert check_all_input_universal(fst), \
            "TokenDecompose requires all_input_universal"

        self.fst = fst
        self.target = tuple(target)
        target_alphabet = fst.B - {EPSILON}
        oov = set(target) - target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")
        target_len = len(self.target)

        token_list = extract_token_bytes(fst)
        trie = ByteTrie()
        for (token_id, byte_seq) in token_list:
            if byte_seq:
                trie.insert(token_id, byte_seq)

        matches = [trie.matches_at(self.target, p) for p in range(target_len)]
        zero_len_tokens = [tid for (tid, bs) in token_list if not bs]

        Q = FSA()
        R = FSA()
        worklist: deque = deque()
        visited: set = set()

        start_state = frozenset({0})
        worklist.append(start_state)
        visited.add(start_state)
        Q.add_start(start_state)
        R.add_start(start_state)

        while worklist:
            state = worklist.popleft()
            if target_len in state:
                Q.add_stop(state)
                continue

            by_token: dict[Any, set[int]] = defaultdict(set)
            for p in state:
                if p < target_len:
                    for (tid, advance) in matches[p]:
                        new_pos = p + advance
                        if new_pos <= target_len:
                            by_token[tid].add(new_pos)

            for tid in zero_len_tokens:
                Q.add_arc(state, tid, state)
                R.add_arc(state, tid, state)

            for token_id, succ_positions in by_token.items():
                if not succ_positions:
                    continue
                succ_state = frozenset(succ_positions)
                Q.add_arc(state, token_id, succ_state)
                R.add_arc(state, token_id, succ_state)
                if succ_state not in visited:
                    visited.add(succ_state)
                    worklist.append(succ_state)

        self.quotient = Q
        self.remainder = R


# ---------------------------------------------------------------------------
# Position-set key extraction
# ---------------------------------------------------------------------------

def _position_key(nfa_state_set, N: int) -> frozenset:
    """Compute position-set key from a frozenset of PeekabooLookaheadNFA elements.

    Each NFA element is ``(fst_state, buffer_tuple, truncated)``.
    The position key extracts ``(buf_len, extra_byte_or_None, truncated)``
    tuples, discarding the fst_state.

    For token-decomposable FSTs, all NFA elements sharing the same position
    key have identical finality and universality, so the quotienting is exact.
    """
    result = set()
    for (i, ys, truncated) in nfa_state_set:
        buf_len = len(ys)
        extra = ys[N] if buf_len > N else None
        result.add((buf_len, extra, truncated))
    return frozenset(result)


# ---------------------------------------------------------------------------
# TokenPeekabooHelper — FusedTransducedLM helper with position-set quotienting
# ---------------------------------------------------------------------------

class _PeekabooClassifyResult:
    __slots__ = ('quotient_sym', 'remainder_syms', 'is_preimage',
                 'has_truncated', 'trunc_output_syms')

    def __init__(self, quotient_sym, remainder_syms, is_preimage,
                 has_truncated, trunc_output_syms):
        self.quotient_sym = quotient_sym
        self.remainder_syms = remainder_syms
        self.is_preimage = is_preimage
        self.has_truncated = has_truncated
        self.trunc_output_syms = trunc_output_syms


class TokenPeekabooHelper:
    """FusedTransducedLM helper using position-set-quotiented peekaboo DFA.

    Builds the standard PeekabooLookaheadNFA (optionally with trie-dispatch),
    determinizes with position-set quotienting, and provides the
    ``classify``/``arcs``/``run`` interface expected by ``FusedTransducedLM``.

    For token-decomposable FSTs, the quotiented DFA is much smaller than the
    standard DFA.  For BPE (1,313 states): ~45 position-set states vs
    ~7,000 standard DFA states per step.
    """

    def __init__(self, fst):
        self.fst = fst
        self._sym_map = Integerizer()
        for s in (fst.A - {EPSILON}):
            self._sym_map(s)
        for s in (fst.B - {EPSILON}):
            self._sym_map(s)
        self._inv_sym = {v: k for k, v in self._sym_map.items()}

        from transduction.peekaboo_incremental import FstUniversality
        self._univ = FstUniversality(fst)

        self._dfa_states = None       # canonical_key -> representative frozenset
        self._dfa_arcs = None         # representative -> [(label, representative)]
        self._dfa_starts = None       # list of representative frozensets
        self._decomp = None
        self._resume_frontiers = None
        self._preimage_stops = None
        self._classify_cache = {}
        self._arcs_cache = {}
        self._target = ()

    def idx_to_sym_map(self):
        out = [s for s in self._sym_map if s in self.fst.B and s != EPSILON]
        return [self._sym_map[s] for s in out]

    def new_step(self, target_u32):
        target = tuple(self._inv_sym[u] for u in target_u32)
        self._target = target
        self._classify_cache.clear()
        self._arcs_cache.clear()

        N = len(target)

        # Build peekaboo NFA (with trie dispatch if available)
        from transduction.trie_dispatch import TrieDispatchPeekabooPrecover
        nfa = TrieDispatchPeekabooPrecover(self.fst, target)

        # Quotiented powerset determinization + classification in a single BFS.
        # `canonical` maps position keys to representative DFA states (frozensets).
        canonical: dict[frozenset, frozenset] = {}
        dfa_arcs: dict[frozenset, list[tuple[Any, frozenset]]] = {}

        # Compute NFA start states and epsilon closure
        raw_starts: set = set()
        for s in nfa.start():
            raw_starts.add(s)
        # Epsilon closure of start
        start_fs = self._eps_closure(nfa, frozenset(raw_starts))
        start_key = _position_key(start_fs, N)
        canonical[start_key] = start_fs

        worklist: deque = deque()
        worklist.append(start_key)

        decomp: dict[Any, DecompositionResult] = {}
        resume_frontiers: dict[Any, set] = {}
        preimage_stops: set = set()

        _all_input_universal = self._univ.all_input_universal
        _fst_is_final = self.fst.is_final

        if not _all_input_universal:
            from transduction.peekaboo_incremental import TruncatedDFA, FstUniversality
            truncated_dfas: dict = {}
            univ_filters: dict = {}

        def ensure_symbol(y):
            if y not in decomp:
                decomp[y] = DecompositionResult(set(), set())
                resume_frontiers[y] = set()
                if not _all_input_universal:
                    # Build the truncated DFA and universality filter for y
                    # We need a LazyDeterminize-compatible DFA object.
                    # Use the NFA for target + (y,)
                    from transduction.peekaboo_incremental import TruncatedDFA
                    nfa_y = TrieDispatchPeekabooPrecover(self.fst, target + (y,))
                    trunc_dfa = TruncatedDFA(
                        dfa=None, fst=self.fst, target=target + (y,),
                    )
                    # For non-aiu, store a reference for later universality checks.
                    # We use the _univ helper's make_filter, which needs a DFA-like
                    # object with arcs().  We pass None and rely on the state-level check.
                    truncated_dfas[y] = trunc_dfa
                    univ_filters[y] = self._univ.make_filter(
                        self.fst, target + (y,), trunc_dfa, self.fst.A - {EPSILON},
                    )

        expanded_keys: set = set()

        while worklist:
            key = worklist.popleft()
            if key in expanded_keys:
                continue
            expanded_keys.add(key)

            state = canonical[key]

            # --- Classify this state ---
            relevant_symbols: set = set()
            final_symbols: set = set()
            state_has_truncated = False
            state_is_preimage = False

            for (i, ys, truncated) in state:
                if len(ys) == N and _fst_is_final(i):
                    state_is_preimage = True
                if len(ys) > N:
                    y = ys[N]
                    relevant_symbols.add(y)
                    if ys[:N] == target and _fst_is_final(i):
                        final_symbols.add(y)
                state_has_truncated = state_has_truncated or truncated

            if state_is_preimage:
                preimage_stops.add(state)

            continuous = None
            for y in relevant_symbols:
                ensure_symbol(y)
                is_univ = (
                    y in final_symbols if _all_input_universal
                    else univ_filters[y].is_universal(state)
                )
                if is_univ:
                    if continuous is not None:
                        raise ValueError(
                            f"State is universal for both {continuous!r} and {y!r}"
                        )
                    decomp[y].quotient.add(state)
                    continuous = y
                    continue
                if y in final_symbols:
                    decomp[y].remainder.add(state)

            if continuous is not None:
                # Q-absorbed — don't expand further
                dfa_arcs[state] = []
                continue

            # --- Expand: compute NFA successors per input symbol ---
            by_label: dict[Any, set] = defaultdict(set)
            for nfa_elem in state:
                for (x, dest) in nfa.arcs(nfa_elem):
                    if x != EPSILON:
                        by_label[x].add(dest)

            # Epsilon closure of each successor set
            arcs_list: list[tuple[Any, frozenset]] = []
            for x, dest_set in by_label.items():
                dest_fs = self._eps_closure(nfa, frozenset(dest_set))
                dest_key = _position_key(dest_fs, N)
                if dest_key not in canonical:
                    canonical[dest_key] = dest_fs
                dest_rep = canonical[dest_key]
                arcs_list.append((x, dest_rep))

                if dest_key not in expanded_keys:
                    worklist.append(dest_key)

                # Truncation-boundary carry-forward detection
                if not state_has_truncated:
                    for (_, ys_d, trunc_d) in dest_rep:
                        if trunc_d:
                            y_d = ys_d[-1] if ys_d else None
                            if y_d is not None:
                                ensure_symbol(y_d)
                                resume_frontiers[y_d].add(state)

            dfa_arcs[state] = arcs_list

        # Add non-truncated Q/R states to resume frontiers
        for y in decomp:
            for st in decomp[y].quotient | decomp[y].remainder:
                if not any(truncated for _, _, truncated in st):
                    resume_frontiers.setdefault(y, set()).add(st)

        self._dfa_states = canonical
        self._dfa_arcs = dfa_arcs
        self._dfa_starts = [canonical[start_key]]
        self._decomp = decomp
        self._resume_frontiers = resume_frontiers
        self._preimage_stops = preimage_stops

    def _eps_closure(self, nfa, state_set: frozenset) -> frozenset:
        """Compute epsilon closure of a set of NFA states."""
        result = set(state_set)
        worklist = list(state_set)
        while worklist:
            s = worklist.pop()
            for (x, dest) in nfa.arcs(s):
                if x == EPSILON and dest not in result:
                    result.add(dest)
                    worklist.append(dest)
        return frozenset(result)

    def start_ids(self):
        return list(self._dfa_starts)

    def arcs(self, sid):
        cached = self._arcs_cache.get(id(sid))
        if cached is not None:
            return cached
        raw = self._dfa_arcs.get(sid, [])
        arcs = [(self._sym_map(x), dest) for x, dest in raw]
        self._arcs_cache[id(sid)] = arcs
        return arcs

    def run(self, source_path):
        path = [self._inv_sym[x] for x in source_path]
        states = self._dfa_starts
        if not states:
            return None
        state = states[0]
        for x in path:
            found = False
            for (lbl, dest) in self._dfa_arcs.get(state, []):
                if lbl == x:
                    state = dest
                    found = True
                    break
            if not found:
                return None
        return state

    def classify(self, sid):
        cached = self._classify_cache.get(id(sid))
        if cached is not None:
            return cached

        quotient_sym = None
        remainder_syms = []
        for y, d in self._decomp.items():
            if sid in d.quotient:
                if quotient_sym is not None and quotient_sym != y:
                    raise ValueError("Multiple quotient symbols for one state")
                quotient_sym = y
            if sid in d.remainder:
                remainder_syms.append(y)

        is_preimage = sid in self._preimage_stops
        has_truncated = any(truncated for _, _, truncated in sid)
        trunc_output_syms = [
            y for y, states in self._resume_frontiers.items()
            if sid in states
        ]

        result = _PeekabooClassifyResult(
            quotient_sym=(self._sym_map(quotient_sym) if quotient_sym is not None else None),
            remainder_syms=[self._sym_map(y) for y in remainder_syms],
            is_preimage=is_preimage,
            has_truncated=has_truncated,
            trunc_output_syms=[self._sym_map(y) for y in trunc_output_syms],
        )
        self._classify_cache[id(sid)] = result
        return result


# ---------------------------------------------------------------------------
# TokenPeekabooState — TransducedLM adapter
# ---------------------------------------------------------------------------

class TokenPeekabooState:
    """PeekabooState adapter backed by position-set-quotiented DFA.

    Drop-in replacement for ``RustPeekabooState`` when used with
    ``TransducedLM(decomp_state_cls=TokenPeekabooState)``.
    """

    _LAZY_BFS_ATTRS = frozenset({
        'decomp', 'dfa', 'resume_frontiers', 'preimage_stops',
    })

    def __init__(self, fst, target=(), parent=None, *, univ=None,
                 _helper=None):
        self.fst = fst
        self.target = tuple(target) if not isinstance(target, tuple) else target
        self.target_alphabet = fst.B - {EPSILON}
        self.source_alphabet = fst.A - {EPSILON}
        oov = set(self.target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        # Share helper across steps for stable sym_map
        if _helper is not None:
            self._helper = _helper
        else:
            self._helper = TokenPeekabooHelper(fst)

        self._bfs_done = False

    def __getattr__(self, name):
        if name in TokenPeekabooState._LAZY_BFS_ATTRS:
            self._ensure_bfs()
            return self.__dict__[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute {name!r}"
        )

    def _ensure_bfs(self):
        if self._bfs_done:
            return

        helper = self._helper
        sym_map = helper._sym_map
        inv_sym = helper._inv_sym

        target_u32 = [sym_map(y) for y in self.target]
        helper.new_step(target_u32)

        # Build decomp: {py_sym: DecompositionResult(quotient=set, remainder=set)}
        decomp = {}
        for y, d in helper._decomp.items():
            decomp[y] = DecompositionResult(set(d.quotient), set(d.remainder))

        resume_frontiers = {}
        for y, states in helper._resume_frontiers.items():
            resume_frontiers[y] = set(states)

        preimage_stops = set(helper._preimage_stops)

        # DFA adapter
        dfa = _TokenDFAAdapter(helper)

        self.decomp = decomp
        self.resume_frontiers = resume_frontiers
        self.preimage_stops = preimage_stops
        self.dfa = dfa
        self._bfs_done = True

    def __rshift__(self, y):
        return TokenPeekabooState(
            self.fst, self.target + (y,),
            _helper=self._helper,
        )


class _TokenDFAAdapter:
    """Wraps TokenPeekabooHelper as a DFA for TransducedLM beam search."""

    def __init__(self, helper: TokenPeekabooHelper):
        self._helper = helper

    def arcs(self, state):
        raw = self._helper._dfa_arcs.get(state, [])
        return list(raw)

    def run(self, source_path):
        states = self._helper._dfa_starts
        if not states:
            return None
        state = states[0]
        for x in source_path:
            found = False
            for (lbl, dest) in self._helper._dfa_arcs.get(state, []):
                if lbl == x:
                    state = dest
                    found = True
                    break
            if not found:
                return None
        return state

    def start(self):
        return list(self._helper._dfa_starts)
