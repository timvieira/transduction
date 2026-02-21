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

    Two modes depending on FST structure:

    **Trie mode** (``all_input_universal``): States are frozensets of
    ``(buf_len, extra_byte_or_None, truncated)`` descriptors tracking hub
    positions only.  Arcs are computed lazily via ByteTrie matching,
    bypassing NFA epsilon-closure entirely.

    **NFA mode** (non-AIU fallback): States are frozensets of NFA elements
    ``(fst_state, buffer, truncated)``, quotiented by position key.  Arcs
    are computed lazily via NFA expansion + epsilon closure on demand.

    Both modes provide the ``classify``/``arcs``/``run`` interface expected
    by ``FusedTransducedLM``.
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

        # Trie mode: all_input_universal AND all hubs (start states) are final
        self._use_trie = (
            self._univ.all_input_universal
            and all(fst.is_final(s) for s in fst.start)
        )
        if self._use_trie:
            token_list = extract_token_bytes(fst)
            self._trie = build_trie(fst)
            self._token_bytes = {tid: bs for tid, bs in token_list}
            self._all_tokens = [(tid, bs) for tid, bs in token_list if bs]
            self._zero_len_tokens = [tid for tid, bs in token_list if not bs]

        # Per-step state (reset in new_step)
        self._classify_cache: dict = {}
        self._arcs_cache: dict = {}
        self._target: tuple = ()
        self._N: int = 0
        self._dfa_starts: list = []
        self._lazy_arcs: dict = {}        # state -> [(label, dest_state)]
        self._canonical: dict = {}        # frozenset -> canonical frozenset
        self._resume_membership: dict = {}  # id(state) -> set of output symbols

    def idx_to_sym_map(self):
        out = [s for s in self._sym_map if s in self.fst.B and s != EPSILON]
        return [self._sym_map[s] for s in out]

    def new_step(self, target_u32):
        target = tuple(self._inv_sym[u] for u in target_u32)
        self._target = target
        N = len(target)
        self._N = N
        self._classify_cache.clear()
        self._arcs_cache.clear()
        self._lazy_arcs.clear()
        self._canonical.clear()
        self._resume_membership.clear()

        if self._use_trie:
            self._new_step_trie(target, N)
        else:
            self._new_step_nfa(target, N)

    def _new_step_trie(self, target, N):
        # Precompute trie matches for each target position
        self._matches = [self._trie.matches_at(target, p) for p in range(N)]

        # Start state: hub at position 0
        start = frozenset({(0, None, False)})
        self._canonical[start] = start
        self._dfa_starts = [start]

    def _new_step_nfa(self, target, N):
        from transduction.trie_dispatch import TrieDispatchPeekabooPrecover
        self._nfa = TrieDispatchPeekabooPrecover(self.fst, target)

        # Compute NFA start states and epsilon closure
        raw_starts = set(self._nfa.start())
        start_fs = self._eps_closure(self._nfa, frozenset(raw_starts))
        start_key = _position_key(start_fs, N)
        self._canonical[start_key] = start_fs
        self._dfa_starts = [start_fs]

        # Lazily built universality filters (non-AIU only)
        self._univ_filters: dict = {}

    # ---- Trie-based lazy expansion (AIU path) ----

    def _expand_trie(self, state):
        """Compute arcs from ``state`` using trie matching."""
        N = self._N
        by_token: dict = defaultdict(set)

        for (pos, extra, truncated) in state:
            if pos < N:
                # Hub within target: use precomputed trie matches
                for (tid, advance) in self._matches[pos]:
                    tlen = len(self._token_bytes[tid])
                    if tlen == advance:
                        # Token fits within (or exactly at) target boundary
                        by_token[tid].add((pos + advance, None, False))
                    else:
                        # Token extends beyond target
                        extra_byte = self._token_bytes[tid][N - pos]
                        trunc = tlen > (N - pos) + 1
                        by_token[tid].add((N + 1, extra_byte, trunc))
            elif pos == N and extra is None:
                # Hub at target end: all tokens fire
                for tid, bs in self._all_tokens:
                    by_token[tid].add((N + 1, bs[0], len(bs) > 1))
            elif pos == N + 1 and not truncated:
                # Non-truncated lookahead: all non-zero tokens -> truncated
                for tid, _bs in self._all_tokens:
                    by_token[tid].add((N + 1, extra, True))
            elif pos == N + 1 and truncated:
                # Already truncated: all tokens -> self-loop
                for tid, _bs in self._all_tokens:
                    by_token[tid].add((N + 1, extra, True))

        # Build arc list, canonicalize successor states
        arcs_list: list = []
        state_has_truncated = any(t for _, _, t in state)
        resume_syms: set = set()

        for tid, succ_descs in by_token.items():
            succ_state = frozenset(succ_descs)
            succ_state = self._canonicalize(succ_state)
            arcs_list.append((tid, succ_state))

            # Resume frontier detection: non-truncated source -> truncated dest
            if not state_has_truncated:
                for (_, extra_d, trunc_d) in succ_state:
                    if trunc_d and extra_d is not None:
                        resume_syms.add(extra_d)

        # Zero-length tokens: self-loops
        for tid in self._zero_len_tokens:
            arcs_list.append((tid, state))

        self._lazy_arcs[state] = arcs_list

        # Record resume frontier membership
        if resume_syms:
            self._resume_membership.setdefault(id(state), set()).update(
                resume_syms,
            )

    def _compute_classify_trie(self, state):
        """Compute classification for a trie-based state."""
        N = self._N

        relevant_symbols: set = set()
        has_truncated = False
        is_preimage = False

        for (pos, extra, truncated) in state:
            if pos == N and extra is None and not truncated:
                is_preimage = True
            if pos == N + 1 and extra is not None:
                relevant_symbols.add(extra)
            has_truncated = has_truncated or truncated

        # With all_input_universal + all hubs final: every relevant symbol
        # with a final hub is universal (Q-absorbed).
        quotient_sym = None
        remainder_syms: list = []
        for y in relevant_symbols:
            if quotient_sym is not None:
                raise ValueError(
                    f"State is universal for both {quotient_sym!r} and {y!r}"
                )
            quotient_sym = y

        # Trunc output syms
        trunc_output_syms_set: set = set()
        rm = self._resume_membership.get(id(state))
        if rm:
            trunc_output_syms_set.update(rm)
        # Non-truncated Q/R states are also resume frontiers
        if not has_truncated:
            if quotient_sym is not None:
                trunc_output_syms_set.add(quotient_sym)
            for y in remainder_syms:
                trunc_output_syms_set.add(y)

        return _PeekabooClassifyResult(
            quotient_sym=(
                self._sym_map(quotient_sym)
                if quotient_sym is not None else None
            ),
            remainder_syms=[self._sym_map(y) for y in remainder_syms],
            is_preimage=is_preimage,
            has_truncated=has_truncated,
            trunc_output_syms=[
                self._sym_map(y) for y in trunc_output_syms_set
            ],
        )

    # ---- NFA-based lazy expansion (non-AIU fallback) ----

    def _expand_nfa(self, state):
        """Compute arcs from ``state`` using NFA expansion + epsilon closure."""
        N = self._N
        nfa = self._nfa

        by_label: dict = defaultdict(set)
        for nfa_elem in state:
            for (x, dest) in nfa.arcs(nfa_elem):
                if x != EPSILON:
                    by_label[x].add(dest)

        arcs_list: list = []
        closure_cache: dict = {}
        state_has_truncated = any(t for _, _, t in state)
        resume_syms: set = set()

        for x, dest_set in by_label.items():
            dest_key_raw = frozenset(dest_set)
            if dest_key_raw not in closure_cache:
                closure_cache[dest_key_raw] = self._eps_closure(
                    nfa, dest_key_raw,
                )
            dest_fs = closure_cache[dest_key_raw]
            dest_key = _position_key(dest_fs, N)
            if dest_key not in self._canonical:
                self._canonical[dest_key] = dest_fs
            dest_rep = self._canonical[dest_key]
            arcs_list.append((x, dest_rep))

            # Resume frontier detection
            if not state_has_truncated:
                for (_, ys_d, trunc_d) in dest_rep:
                    if trunc_d:
                        y_d = ys_d[-1] if ys_d else None
                        if y_d is not None:
                            resume_syms.add(y_d)

        self._lazy_arcs[state] = arcs_list

        if resume_syms:
            self._resume_membership.setdefault(id(state), set()).update(
                resume_syms,
            )

    def _ensure_univ_filter(self, y):
        """Lazily build universality filter for output symbol ``y``."""
        if y not in self._univ_filters:
            from transduction.peekaboo_incremental import TruncatedDFA
            target = self._target
            trunc_dfa = TruncatedDFA(
                dfa=None, fst=self.fst, target=target + (y,),
            )
            self._univ_filters[y] = self._univ.make_filter(
                self.fst, target + (y,), trunc_dfa,
                self.fst.A - {EPSILON},
            )

    def _compute_classify_nfa(self, state):
        """Compute classification for an NFA-based state."""
        N = self._N
        target = self._target
        _fst_is_final = self.fst.is_final
        _all_input_universal = self._univ.all_input_universal

        relevant_symbols: set = set()
        final_symbols: set = set()
        has_truncated = False
        is_preimage = False

        for (i, ys, truncated) in state:
            if len(ys) == N and _fst_is_final(i):
                is_preimage = True
            if len(ys) > N:
                y = ys[N]
                relevant_symbols.add(y)
                if ys[:N] == target and _fst_is_final(i):
                    final_symbols.add(y)
            has_truncated = has_truncated or truncated

        quotient_sym = None
        remainder_syms: list = []
        for y in relevant_symbols:
            if _all_input_universal:
                is_univ = y in final_symbols
            else:
                self._ensure_univ_filter(y)
                is_univ = self._univ_filters[y].is_universal(state)
            if is_univ:
                if quotient_sym is not None:
                    raise ValueError(
                        f"State is universal for both {quotient_sym!r} "
                        f"and {y!r}"
                    )
                quotient_sym = y
                continue
            if y in final_symbols:
                remainder_syms.append(y)

        # Trunc output syms
        trunc_output_syms_set: set = set()
        rm = self._resume_membership.get(id(state))
        if rm:
            trunc_output_syms_set.update(rm)
        if not has_truncated:
            if quotient_sym is not None:
                trunc_output_syms_set.add(quotient_sym)
            for y in remainder_syms:
                trunc_output_syms_set.add(y)

        return _PeekabooClassifyResult(
            quotient_sym=(
                self._sym_map(quotient_sym)
                if quotient_sym is not None else None
            ),
            remainder_syms=[self._sym_map(y) for y in remainder_syms],
            is_preimage=is_preimage,
            has_truncated=has_truncated,
            trunc_output_syms=[
                self._sym_map(y) for y in trunc_output_syms_set
            ],
        )

    # ---- Shared helpers ----

    def _canonicalize(self, state):
        if state in self._canonical:
            return self._canonical[state]
        self._canonical[state] = state
        return state

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

    def _ensure_expanded(self, state):
        """Ensure arcs for ``state`` have been computed."""
        if state not in self._lazy_arcs:
            if self._use_trie:
                self._expand_trie(state)
            else:
                self._expand_nfa(state)

    # ---- Public interface ----

    def start_ids(self):
        return list(self._dfa_starts)

    def arcs(self, sid):
        cached = self._arcs_cache.get(id(sid))
        if cached is not None:
            return cached
        self._ensure_expanded(sid)
        raw = self._lazy_arcs.get(sid, [])
        arcs = [(self._sym_map(x), dest) for x, dest in raw]
        self._arcs_cache[id(sid)] = arcs
        return arcs

    def run(self, source_path):
        path = [self._inv_sym[x] for x in source_path]
        if not self._dfa_starts:
            return None
        state = self._dfa_starts[0]
        for x in path:
            self._ensure_expanded(state)
            found = False
            for (lbl, dest) in self._lazy_arcs.get(state, []):
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
        if self._use_trie:
            result = self._compute_classify_trie(sid)
        else:
            result = self._compute_classify_nfa(sid)
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

        target_u32 = [sym_map(y) for y in self.target]
        helper.new_step(target_u32)

        # Materialize all reachable states to extract decomp/resume/preimage.
        worklist: deque = deque(helper._dfa_starts)
        visited: set = {id(s) for s in helper._dfa_starts}

        decomp: dict = {}
        resume_frontiers: dict = {}
        preimage_stops: set = set()
        inv = helper._inv_sym

        while worklist:
            state = worklist.popleft()
            cls = helper.classify(state)

            if cls.is_preimage:
                preimage_stops.add(state)

            q_sym = (
                inv[cls.quotient_sym] if cls.quotient_sym is not None
                else None
            )
            r_syms = [inv[s] for s in cls.remainder_syms]
            trunc_syms = [inv[s] for s in cls.trunc_output_syms]

            for y in trunc_syms:
                resume_frontiers.setdefault(y, set()).add(state)

            if q_sym is not None:
                decomp.setdefault(
                    q_sym, DecompositionResult(set(), set()),
                )
                decomp[q_sym].quotient.add(state)
                # Q-absorbed: don't expand arcs
                continue

            for y in r_syms:
                decomp.setdefault(
                    y, DecompositionResult(set(), set()),
                )
                decomp[y].remainder.add(state)

            # Expand
            helper._ensure_expanded(state)
            for (_, dest) in helper._lazy_arcs.get(state, []):
                if id(dest) not in visited:
                    visited.add(id(dest))
                    worklist.append(dest)

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
        self._helper._ensure_expanded(state)
        return list(self._helper._lazy_arcs.get(state, []))

    def run(self, source_path):
        helper = self._helper
        if not helper._dfa_starts:
            return None
        state = helper._dfa_starts[0]
        for x in source_path:
            helper._ensure_expanded(state)
            found = False
            for (lbl, dest) in helper._lazy_arcs.get(state, []):
                if lbl == x:
                    state = dest
                    found = True
                    break
            if not found:
                return None
        return state

    def start(self):
        return list(self._helper._dfa_starts)
