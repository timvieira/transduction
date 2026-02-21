"""
Python port of the Rust bitset and hash-consing optimizations for lazy precover DFA.

Faithfully reproduces the key data structures and algorithms from:
  - crates/transduction-core/src/powerset.rs  → PowersetArena
  - crates/transduction-core/src/precover.rs  → PackedPrecoverNFA
  - crates/transduction-core/src/lazy_precover.rs → LazyPrecoverDFA

Three optimizations work together:

1. **Integer packing** (PackedPrecoverNFA): NFA states ``(fst_state, buf_pos)``
   are packed into a single int via ``fst_state * stride + buf_pos``.  Since
   the output buffer is always a prefix of the target, we only need its
   *length*, not the string itself.  This makes state hashing O(1) — a single
   int hash — instead of O(tuple_size) for the tuple ``(fst_state, ys)``
   representation in the original PrecoverNFA.

2. **Hash-consing (PowersetArena)**: DFA states (sets of NFA states) are interned
   as sorted tuples and assigned integer IDs.  Identical NFA-state sets always
   get the same ID.  A **singleton fast path** hashes just the bare int for
   single-element sets (which are ~99% of states in BPE FSTs), avoiding tuple
   construction and hashing overhead.

3. **Epsilon closure caching with productivity filtering** (PackedPrecoverNFA):
   Each NFA state's epsilon closure is computed once via BFS and cached.  Only
   **productive** states are kept in the closure — those whose FST state has
   non-epsilon input arcs (contribute DFA transitions) or that are NFA-final
   (affect powerset finality).  Transit-only states in epsilon chains are
   filtered out, which for BPE FSTs reduces powerset sizes from ~7000 to ~100.

Usage::

    from transduction.lazy_precover_dfa import LazyPrecoverDFA

    dfa = LazyPrecoverDFA(fst, target)

    # Drop-in replacement for PrecoverNFA(fst, target).det()
    [start] = dfa.start()
    for symbol, dest in dfa.arcs(start):
        print(f"{symbol} -> {dest} (final={dfa.is_final(dest)})")

    # Inspect internals:
    print(dfa.stats())
"""

from __future__ import annotations

from collections import deque
from typing import Any

from transduction.lazy import Lazy, EPSILON
from transduction.util import Integerizer, State


# ═══════════════════════════════════════════════════════════════════════
# PowersetArena — hash-consing for DFA states
# Mirrors: crates/transduction-core/src/powerset.rs
# ═══════════════════════════════════════════════════════════════════════

class PowersetArena:
    """Intern sorted tuples of packed NFA states as integer DFA state IDs.

    After interning, all DFA operations use cheap integer IDs rather than
    hashing large frozensets or tuples of NFA states.

    **Singleton fast path**: When the powerset contains exactly one NFA
    state, we hash just the bare int rather than a 1-tuple.  For BPE FSTs,
    ~99% of powerset states are singletons (each FST state has a unique
    epsilon-closed successor per input symbol), so this avoids tuple
    construction and hashing on the hot path.

    **Finality update on cache hit**: ``is_final`` is always overwritten
    when a set is re-interned.  This is essential for incremental
    decomposition where the same NFA state set can be final at one target
    length but not another.
    """

    def __init__(self):
        # Two separate maps: one for singletons (int → id), one for
        # multi-element sets (tuple → id).  This is the singleton fast path.
        self._single_map: dict[int, int] = {}
        self._map: dict[tuple[int, ...], int] = {}

        # Parallel arrays indexed by state ID.
        self.sets: list[tuple[int, ...]] = []
        self.is_final: list[bool] = []

    def intern(self, sorted_set: tuple[int, ...], any_final: bool) -> int:
        """Intern a sorted set of NFA states.  Returns the integer state ID.

        On cache hit, ``is_final[id]`` is **always updated** to *any_final*.
        """
        if len(sorted_set) == 1:
            # ── Singleton fast path ──
            # Hash a single int instead of a 1-tuple.
            key = sorted_set[0]
            sid = self._single_map.get(key)
            if sid is not None:
                self.is_final[sid] = any_final
                return sid
            sid = len(self.sets)
            self.sets.append(sorted_set)
            self.is_final.append(any_final)
            self._single_map[key] = sid
            return sid

        # ── General path ──
        sid = self._map.get(sorted_set)
        if sid is not None:
            self.is_final[sid] = any_final
            return sid
        sid = len(self.sets)
        self.sets.append(sorted_set)
        self.is_final.append(any_final)
        self._map[sorted_set] = sid
        return sid

    def lookup(self, sorted_set: tuple[int, ...]) -> int | None:
        """Look up an existing set.  Returns ``None`` if not interned."""
        if len(sorted_set) == 1:
            return self._single_map.get(sorted_set[0])
        return self._map.get(sorted_set)

    def __len__(self) -> int:
        return len(self.sets)


# ═══════════════════════════════════════════════════════════════════════
# PackedPrecoverNFA — NFA with packed integer states and cached closures
# Mirrors: crates/transduction-core/src/precover.rs
# ═══════════════════════════════════════════════════════════════════════

class PackedPrecoverNFA:
    """Precover NFA with packed integer states and cached epsilon closures.

    State packing
    -------------
    Each NFA state ``(fst_state, buf_pos)`` is packed into a single int::

        packed = fst_state * stride + buf_pos

    where ``stride = target_len + 1``.  Since the output buffer is always
    a prefix of the target string, we only need to track its length
    (``buf_pos``), not the actual string.  Packing makes state hashing
    and comparison O(1).

    Epsilon closure caching
    -----------------------
    The closure of each NFA state is computed once (BFS over epsilon arcs)
    and cached as a sorted tuple.  Subsequent lookups are O(1) dict hits.

    Productivity filtering
    ----------------------
    Only *productive* states are kept in the closure:

    - States whose FST state has at least one non-epsilon input arc
      (they contribute DFA transitions via ``compute_all_arcs``).
    - States that are NFA-final (they affect powerset finality).

    Transit-only states (epsilon-input-only, non-final) are dropped.  For
    BPE FSTs with multi-byte tokens, this collapses long epsilon chains
    (one FST state per byte) to just the endpoints.

    Two-phase NFA transitions
    -------------------------
    Mirrors the Rust ``PrecoverNFA::arcs`` and ``arcs_x`` methods:

    **Boundary** (``buf_pos == target_len``): the entire target has been
    produced.  All FST arcs are available — further output is absorbed.

    **Growing** (``buf_pos < target_len``): only arcs whose output matches
    the next target symbol (advancing ``buf_pos``) or EPSILON (keeping
    ``buf_pos``) are followed.  Other outputs are incompatible.
    """

    def __init__(self, fst, target, *, stride: int | None = None):
        self.fst = fst
        self.target = tuple(target)
        self.target_len = len(self.target)
        self.stride = stride if stride is not None else self.target_len + 1
        assert self.stride >= self.target_len + 1

        fst.ensure_trie_index()

        # ── Integerize FST states ──
        # The Rust code assumes contiguous u32 state IDs.  Python FSTs can
        # use arbitrary hashable objects as states (strings, tuples, etc.).
        # We map them to contiguous ints so they can be packed into the
        # packed = fst_state * stride + buf_pos integer representation.
        self._state_map: Integerizer = Integerizer()
        for s in fst.states:
            self._state_map(s)
        self._inv_state_map: list = list(self._state_map)   # int → original state

        # ── Pre-build epsilon-input arc indexes ──
        # In the Rust code these are Fst methods: eps_input_arcs(i),
        # eps_input_arcs_by_output(i, y).  We build equivalent dicts here,
        # keyed by integerized state IDs for cheap packing.

        # _eps_arcs[int_state] → tuple of (output, int_dest) for input=ε arcs.
        # Used in boundary-phase closure (all outputs are OK).
        self._eps_arcs: dict[int, tuple] = {}

        # _eps_by_output[(int_state, output)] → tuple of int_dest for
        # eps-input arcs with that specific output.
        # Used in growing-phase closure (only ε or target[buf_pos] OK).
        self._eps_by_output: dict[tuple, tuple] = {}

        # _has_non_eps_input: set of int state IDs with ≥1 non-epsilon input arc.
        # Determines productivity for closure filtering.
        self._has_non_eps_input: set[int] = set()

        # _arcs_all_int[int_state] → tuple of (input_sym, int_dest).
        # _arcs_by_output_int[(int_state, output)] → tuple of (input_sym, int_dest).
        # These mirror the FST's trie index but with integerized state IDs.
        self._arcs_all_int: dict[int, tuple] = {}
        self._arcs_by_output_int: dict[tuple, tuple] = {}

        # _is_final_int[int_state] → bool.
        sm = self._state_map
        self._is_final_int: list[bool] = [
            fst.is_final(self._inv_state_map[i])
            for i in range(len(sm))
        ]

        # _start_states_int: integerized start state IDs.
        self._start_states_int: list[int] = [sm(s) for s in fst.start]

        for orig_state in fst.states:
            si = sm(orig_state)

            # Build epsilon-input arc index (integerized)
            eps = fst._arcs_by_input.get((orig_state, EPSILON), ())
            if eps:
                int_eps = tuple((y, sm(j)) for y, j in eps)
                self._eps_arcs[si] = int_eps
                by_y: dict = {}
                for y, j_int in int_eps:
                    by_y.setdefault(y, []).append(j_int)
                for y, js in by_y.items():
                    self._eps_by_output[(si, y)] = tuple(js)

            # Build all-arcs index (integerized)
            all_arcs = fst._arcs_all.get(orig_state, ())
            if all_arcs:
                self._arcs_all_int[si] = tuple((x, sm(j)) for x, j in all_arcs)

            # Build arcs-by-output index (integerized)
            for y in fst.B:
                arcs_y = fst._arcs_by_output.get((orig_state, y), ())
                if arcs_y:
                    self._arcs_by_output_int[(si, y)] = tuple(
                        (x, sm(j)) for x, j in arcs_y
                    )

            # Non-epsilon input arc check
            for x, _j in all_arcs:
                if x != EPSILON:
                    self._has_non_eps_input.add(si)
                    break

        # ── Epsilon closure cache ──
        # packed_state → (sorted_tuple_of_productive_states, max_buf_pos)
        #
        # max_buf_pos is the maximum buf_pos in the closure; used for
        # fast cache eviction in incremental mode (not used here, but
        # stored for Rust fidelity).
        self._eps_cache: dict[int, tuple[tuple[int, ...], int]] = {}
        self._eps_hits = 0
        self._eps_misses = 0

    # ── State packing ──────────────────────────────────────────────

    def pack(self, fst_state: int, buf_pos: int) -> int:
        """Pack (fst_state, buf_pos) into a single integer."""
        return fst_state * self.stride + buf_pos

    def unpack(self, packed: int) -> tuple[int, int]:
        """Unpack to (fst_state, buf_pos)."""
        return divmod(packed, self.stride)

    # ── NFA queries ────────────────────────────────────────────────

    def is_final(self, packed: int) -> bool:
        """NFA state is final iff FST state is final AND buf_pos == target_len."""
        int_state, buf_pos = self.unpack(packed)
        return self._is_final_int[int_state] and buf_pos == self.target_len

    def is_productive(self, packed: int) -> bool:
        """Whether the state contributes to DFA transitions or finality."""
        int_state, buf_pos = self.unpack(packed)
        return (
            int_state in self._has_non_eps_input
            or (self._is_final_int[int_state] and buf_pos == self.target_len)
        )

    def start_states(self) -> list[int]:
        """Packed start states of the NFA."""
        return [self.pack(s, 0) for s in self._start_states_int]

    def arcs(self, packed: int):
        """All arcs: yields ``(input_symbol, dest_packed)``.

        Includes epsilon-input arcs; ``compute_all_arcs`` filters them out.
        """
        int_state, buf_pos = self.unpack(packed)
        if buf_pos == self.target_len:
            # Boundary phase: all arcs, buffer stays full.
            for x, j in self._arcs_all_int.get(int_state, ()):
                yield x, self.pack(j, self.target_len)
        else:
            # Growing phase: only ε-output and target[buf_pos]-output arcs.
            for x, j in self._arcs_by_output_int.get((int_state, EPSILON), ()):
                yield x, self.pack(j, buf_pos)
            for x, j in self._arcs_by_output_int.get((int_state, self.target[buf_pos]), ()):
                yield x, self.pack(j, buf_pos + 1)

    def arcs_x_epsilon(self, packed: int):
        """Epsilon-input successors (for closure BFS).

        Mirrors Rust's ``arcs_x(packed, EPSILON)``.
        """
        int_state, buf_pos = self.unpack(packed)
        if buf_pos == self.target_len:
            # Boundary: all epsilon-input arcs (any output OK).
            for _y, j in self._eps_arcs.get(int_state, ()):
                yield self.pack(j, self.target_len)
        else:
            # Growing: eps-input with output=ε → buf_pos unchanged.
            for j in self._eps_by_output.get((int_state, EPSILON), ()):
                yield self.pack(j, buf_pos)
            # Growing: eps-input with output=target[buf_pos] → advance.
            for j in self._eps_by_output.get((int_state, self.target[buf_pos]), ()):
                yield self.pack(j, buf_pos + 1)

    # ── Epsilon closure ────────────────────────────────────────────

    def eps_closure_single(self, state: int) -> tuple[int, ...]:
        """Cached epsilon closure of a single NFA state.

        Mirrors Rust's ``eps_closure_single_cached``:

        1. BFS from *state* following epsilon-input arcs.
        2. Filter to productive states only.
        3. Sort for deterministic hashing (sorted tuples are the
           canonical representation for PowersetArena).
        4. Cache for O(1) retrieval on subsequent calls.

        Returns:
            Sorted tuple of productive packed NFA states.
        """
        cached = self._eps_cache.get(state)
        if cached is not None:
            self._eps_hits += 1
            return cached[0]
        self._eps_misses += 1

        # BFS over epsilon arcs
        visited = {state}
        worklist = deque([state])
        while worklist:
            s = worklist.popleft()
            for dest in self.arcs_x_epsilon(s):
                if dest not in visited:
                    visited.add(dest)
                    worklist.append(dest)

        # Productivity filter: keep only states that contribute to DFA
        # transitions or finality.  For BPE, a token like "cat" = [c,a,t]
        # creates 3 intermediary epsilon-only states; filtering collapses
        # the chain to just the endpoints with real input arcs.
        result = tuple(sorted(s for s in visited if self.is_productive(s)))

        # max_buf_pos: used by Rust for incremental cache eviction.
        max_bp = max((s % self.stride for s in result), default=0)
        self._eps_cache[state] = (result, max_bp)
        return result

    def eps_closure_set(self, states) -> tuple[int, ...]:
        """Epsilon closure of a set of states (sorted, deduplicated)."""
        all_states: list[int] = []
        for s in states:
            all_states.extend(self.eps_closure_single(s))
        return tuple(sorted(set(all_states)))

    def eps_cache_stats(self) -> tuple[int, int]:
        """Return ``(hits, misses)`` for the epsilon closure cache."""
        return self._eps_hits, self._eps_misses

    # ── Batch arc computation ──────────────────────────────────────

    def compute_all_arcs(
        self,
        states: tuple[int, ...],
        by_symbol: dict | None = None,
    ) -> list[tuple[Any, tuple[int, ...]]]:
        """Batch-compute all non-epsilon arcs from a powerset state.

        Mirrors Rust's ``compute_all_arcs_into``.

        For each NFA state in the powerset, iterates its arcs (skipping
        epsilon-input arcs), epsilon-closes each destination, and groups
        results by input symbol.

        Complexity: O(total arcs from the powerset) — NOT O(|alphabet|).
        Only symbols that actually appear on arcs are processed.

        Args:
            states: Sorted tuple of packed NFA states (the powerset).
            by_symbol: Optional reusable dict for buffer reuse.  List values
                are cleared but the dict structure is preserved across calls,
                reducing allocation overhead (mirrors the Rust ``arcs_buf``
                pattern).

        Returns:
            List of ``(input_symbol, successor_tuple)`` pairs.
        """
        if by_symbol is None:
            by_symbol = {}
        else:
            # Clear values but keep allocated list objects.
            for v in by_symbol.values():
                v.clear()

        for packed in states:
            for x, dest in self.arcs(packed):
                if x != EPSILON:
                    # Epsilon-close the destination and add to bucket.
                    closure = self.eps_closure_single(dest)
                    bucket = by_symbol.get(x)
                    if bucket is None:
                        by_symbol[x] = list(closure)
                    else:
                        bucket.extend(closure)

        # Collect: sort + dedup each bucket, then clear for reuse.
        result = []
        for sym, bucket in by_symbol.items():
            if bucket:
                result.append((sym, tuple(sorted(set(bucket)))))
                bucket.clear()   # keep list object allocated for next call

        return result


# ═══════════════════════════════════════════════════════════════════════
# LazyPrecoverDFA — lazy DFA with hash-consed powerset states
# Mirrors: crates/transduction-core/src/lazy_precover.rs
# ═══════════════════════════════════════════════════════════════════════

class LazyPrecoverDFA(Lazy):
    """Lazy DFA over the precover NFA, using hash-consed powerset states.

    Drop-in replacement for ``PrecoverNFA(fst, target).det()`` that fuses
    the three Rust optimizations: integer packing, hash-consing, and
    epsilon closure caching with productivity filtering.

    Architecture::

        PrecoverNFA (lazy, tuple states)
            │
            ▼  replaced by
        PackedPrecoverNFA (packed int states, cached closures)
            │
            ▼  wrapped by
        PowersetArena (hash-consing: NFA state sets → int IDs)
            │
            ▼  exposed as
        LazyPrecoverDFA (Lazy interface with int state IDs)

    States are integer IDs assigned by PowersetArena.  Arcs are computed
    lazily on first access and cached.  The epsilon closure cache is
    shared across all DFA expansion steps — each NFA state's closure is
    computed at most once.

    Note: In Rust, the NFA is re-created as a temporary for each expansion
    step (transferring the epsilon cache in/out) due to lifetime constraints.
    In Python we simply keep one NFA instance alive — the effect is the
    same since the FST and target don't change.
    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = tuple(target)
        self.stride = len(self.target) + 1

        # Single NFA instance (in Rust, temporaries are created per-step
        # and the eps_cache is transferred; here we just keep one alive).
        self._nfa = PackedPrecoverNFA(fst, self.target)

        self.arena = PowersetArena()

        # Compute epsilon-closed initial powerset state.
        raw_starts = self._nfa.start_states()
        init_closed = self._nfa.eps_closure_set(raw_starts)
        any_final = any(self._nfa.is_final(s) for s in init_closed)
        self._start_id = self.arena.intern(init_closed, any_final)

        # Per-state arc cache: _arcs_cache[sid] = [...] once expanded, None otherwise.
        self._arcs_cache: list[list[tuple[Any, int]] | None] = [None] * len(self.arena)

        # Reusable buffer for compute_all_arcs (Rust's arcs_buf pattern).
        self._arcs_buf: dict = {}

    # ── Lazy interface ─────────────────────────────────────────────

    def start(self):
        yield self._start_id

    def is_final(self, sid: int) -> bool:
        return self.arena.is_final[sid]

    def arcs(self, sid: int) -> list[tuple[Any, int]]:
        """Lazily compute and return arcs from DFA state *sid*.

        On first call for a given state:

        1. Look up the NFA state set from the arena.
        2. Batch-compute all non-epsilon arcs via ``compute_all_arcs``.
        3. Intern each successor powerset state.
        4. Cache the result.

        Subsequent calls return the cached arc list directly.
        """
        self._ensure_arcs(sid)
        return self._arcs_cache[sid]

    def arcs_x(self, sid: int, x) -> Any:
        self._ensure_arcs(sid)
        for lbl, dest in self._arcs_cache[sid]:
            if lbl == x:
                yield dest

    def epsremove(self):
        """No-op: already a DFA (no epsilon arcs)."""
        return self

    # ── Core expansion logic ───────────────────────────────────────

    def _ensure_arcs(self, sid: int) -> None:
        """Ensure arcs for DFA state *sid* are computed and cached.

        Mirrors ``LazyPrecoverDFA::ensure_arcs_for`` in lazy_precover.rs.
        """
        if sid < len(self._arcs_cache) and self._arcs_cache[sid] is not None:
            return

        nfa = self._nfa
        states = self.arena.sets[sid]

        # Batch-compute all non-epsilon arcs from this powerset state.
        all_arcs = nfa.compute_all_arcs(states, self._arcs_buf)

        # Intern each successor powerset state in the arena.
        result: list[tuple[Any, int]] = []
        for sym, successor in all_arcs:
            any_final = any(nfa.is_final(s) for s in successor)
            dest_id = self.arena.intern(successor, any_final)
            result.append((sym, dest_id))

        # Grow arc cache if arena created new states during interning.
        while len(self._arcs_cache) < len(self.arena):
            self._arcs_cache.append(None)

        self._arcs_cache[sid] = result

    # ── Traversal ──────────────────────────────────────────────────

    def run(self, path) -> int | None:
        """Traverse a source path.  Returns reached DFA state ID or None."""
        state = self._start_id
        for sym in path:
            self._ensure_arcs(state)
            arcs = self._arcs_cache[state]
            found = False
            for lbl, dest in arcs:
                if lbl == sym:
                    state = dest
                    found = True
                    break
            if not found:
                return None
        return state

    # ── Introspection ──────────────────────────────────────────────

    def powerset_size(self, sid: int) -> int:
        """Number of NFA states in the powerset for DFA state *sid*."""
        return len(self.arena.sets[sid])

    def num_states(self) -> int:
        """Total number of interned DFA states so far."""
        return len(self.arena)

    def nfa_states(self, sid: int) -> tuple[int, ...]:
        """Packed NFA state set for DFA state *sid*."""
        return self.arena.sets[sid]

    def unpack_nfa_state(self, packed: int) -> tuple[int, int]:
        """Unpack a packed NFA state to ``(fst_state, buf_pos)``."""
        return divmod(packed, self.stride)

    def stats(self) -> dict:
        """Statistics about the DFA construction so far.

        Useful for understanding the impact of each optimization::

            dfa = LazyPrecoverDFA(fst, target)
            dfa.run(some_path)       # trigger lazy expansion
            print(dfa.stats())
            # {'num_dfa_states': 42,
            #  'num_expanded': 15,
            #  'singleton_fraction': 0.95,   ← singleton fast path helps 95%
            #  'eps_cache_hits': 1200,
            #  'eps_cache_misses': 80,       ← each closure computed once
            #  ...}
        """
        n = len(self.arena)
        sizes = [len(self.arena.sets[i]) for i in range(n)]
        singletons = sum(1 for s in sizes if s == 1)
        hits, misses = self._nfa.eps_cache_stats()
        return {
            'num_dfa_states': n,
            'num_expanded': sum(1 for c in self._arcs_cache[:n] if c is not None),
            'avg_powerset_size': sum(sizes) / n if n else 0,
            'max_powerset_size': max(sizes) if sizes else 0,
            'singleton_fraction': singletons / n if n else 0,
            'eps_cache_size': len(self._nfa._eps_cache),
            'eps_cache_hits': hits,
            'eps_cache_misses': misses,
        }
