"""
Factored decomposition: uses a factored state representation for the
powerset DFA, with position-based universality caching.

Standard representation:
    frozenset({(q1, buf1), (q2, buf2), (q3, buf1), ...})

Factored representation (used for universality caching):
    FState({pos1: frozenset({q1, q3}), pos2: frozenset({q2})})

Hybrid approach:
  - Arc computation uses the standard PrecoverNFA.det() pipeline
    (fast, benefits from EpsilonRemove caching).
  - Universality checking uses factored representation for
    frontier caching: states at position N are cached by their
    FST state set, avoiding redundant BFS checks.
  - Incremental (>> operator): dirty-state reuse across target
    extensions, following TruncatedIncrementalDFADecomp's pattern.
"""

from collections import defaultdict, deque
from functools import cached_property

from transduction import EPSILON
from transduction.fsa import FSA
from transduction.base import DecompositionResult, IncrementalDecomposition
from transduction.precover_nfa import PrecoverNFA
from transduction.universality import (
    UniversalityFilter,
    check_all_input_universal,
    compute_ip_universal_states,
)
from transduction.lazy import EpsilonRemove


class FState:
    """Factored DFA state: maps positions to sets of FST states.

    Internally stored as a tuple of (position, frozenset_of_fst_states) pairs,
    sorted by position for canonical ordering and fast hashing.
    """
    __slots__ = ('_items', '_hash')

    def __init__(self, mapping):
        if isinstance(mapping, dict):
            items = tuple(sorted(
                ((p, qs) for p, qs in mapping.items() if qs),
                key=lambda x: x[0]
            ))
        else:
            items = tuple(sorted(mapping, key=lambda x: x[0]))
        self._items = items
        self._hash = hash(items)

    @classmethod
    def from_nfa_states(cls, nfa_states):
        """Build from an iterable of (fst_state, buffer) NFA states."""
        mapping = defaultdict(set)
        for (q, buf) in nfa_states:
            mapping[len(buf)].add(q)
        return cls({p: frozenset(qs) for p, qs in mapping.items()})

    def to_frozenset_with_target(self, target):
        """Convert to flat frozenset using target to reconstruct buffers."""
        result = set()
        for (pos, qs) in self._items:
            buf = target[:pos]
            for q in qs:
                result.add((q, buf))
        return frozenset(result)

    @property
    def max_pos(self):
        return self._items[-1][0] if self._items else -1

    @property
    def is_position_uniform(self):
        return len(self._items) == 1

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, FState):
            return NotImplemented
        return self._items == other._items

    def __repr__(self):
        parts = [f'{pos}: {qs}' for pos, qs in self._items]
        return f'FState({{{", ".join(parts)}}})'

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


# ---------------------------------------------------------------------------
# Factored universality: caches frontier universality by FST state set
# ---------------------------------------------------------------------------

class FactoredUniversalityFilter:
    """Wraps UniversalityFilter with frontier caching by FST state set.

    For DFA states where all NFA elements are at position N (pure frontier),
    universality depends only on the FST state set. This cache avoids
    redundant BFS universality checks across different targets that share
    the same frontier FST state sets.
    """

    def __init__(self, fst, target, dfa, source_alphabet, *,
                 all_input_universal=None, ip_universal_states=None):
        self.all_input_universal = (
            check_all_input_universal(fst) if all_input_universal is None
            else all_input_universal
        )
        self.ip_universal_states = (
            frozenset() if self.all_input_universal
            else (ip_universal_states if ip_universal_states is not None
                  else compute_ip_universal_states(fst))
        )
        witnesses = frozenset((q, target) for q in self.ip_universal_states)
        self._inner = UniversalityFilter(
            fst, target, dfa, source_alphabet,
            all_input_universal=self.all_input_universal,
            witnesses=witnesses,
        )
        # Cache: frozenset_of_fst_states -> bool (for pure-frontier states)
        self._frontier_cache = {}

    @property
    def dfa(self):
        return self._inner.dfa

    @dfa.setter
    def dfa(self, value):
        self._inner.dfa = value

    def is_universal(self, flat_state, fstate, N):
        """Check universality with frontier caching.

        Args:
            flat_state: frozenset of (fst_state, buffer) NFA states
            fstate: FState factored representation of the same state
            N: target length
        """
        if not self._inner.dfa.is_final(flat_state):
            return False

        if self.all_input_universal:
            return True

        # For pure-frontier states: cache by FST state set
        if fstate.is_position_uniform and fstate.max_pos == N:
            fst_key = fstate._items[0][1]
            cached = self._frontier_cache.get(fst_key)
            if cached is not None:
                return cached
            result = self._inner.is_universal(flat_state)
            self._frontier_cache[fst_key] = result
            return result

        return self._inner.is_universal(flat_state)

    def evict_frontier(self, old_depth, new_target):
        """Evict stale entries for incremental target extension."""
        new_witnesses = frozenset(
            (q, new_target) for q in self.ip_universal_states
        )
        self._inner.evict_frontier(old_depth, new_target, new_witnesses)


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
STATUS_NEW = 0
STATUS_INTERIOR = 1
STATUS_QSTOP = 2
STATUS_RSTOP = 3


def _trimmed_fsa(start_states, stop_states, get_incoming):
    """Build a trimmed FSA by backward BFS from stop states."""
    if not stop_states:
        return FSA()
    backward_reachable = set()
    worklist = deque(stop_states)
    while worklist:
        state = worklist.popleft()
        if state in backward_reachable:
            continue
        backward_reachable.add(state)
        for _, pred in get_incoming(state):
            if pred not in backward_reachable:
                worklist.append(pred)
    arcs = [
        (pred, x, state)
        for state in backward_reachable
        for x, pred in get_incoming(state)
        if pred in backward_reachable
    ]
    return FSA(
        start={s for s in start_states if s in backward_reachable},
        arcs=arcs,
        stop=stop_states,
    )


# ---------------------------------------------------------------------------
# FactoredDecomp: hybrid approach
# ---------------------------------------------------------------------------

class FactoredDecomp(IncrementalDecomposition):
    """Factored decomposition with frontier universality caching.

    Uses the standard PrecoverNFA.det() pipeline for arc computation
    (fast, benefits from EpsilonRemove caching) while using FState
    for universality caching. Supports incremental >> and decompose_next.

    DFA states are standard frozensets internally; FState is used as a
    lightweight projection for universality caching only.
    """

    def __init__(self, fst, target=(), parent=None):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        target = tuple(target)
        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")
        self.target = target
        self._has_eps = EPSILON in fst.A

        if parent is not None:
            assert fst is parent.fst
            self._all_input_universal = parent._all_input_universal
            self._ip_universal_states = parent._ip_universal_states
            self._frontier_cache = parent._frontier_cache
            self._eps_cache = parent._eps_cache
            self._build_incremental(parent)
        else:
            self._all_input_universal = check_all_input_universal(fst)
            self._ip_universal_states = (
                frozenset() if self._all_input_universal
                else compute_ip_universal_states(fst)
            )
            self._frontier_cache = {}
            self._eps_cache = {}
            self._build_fresh()

    def _make_filter(self, dfa):
        """Create a FactoredUniversalityFilter for the current target."""
        return FactoredUniversalityFilter(
            self.fst, self.target, dfa, self.source_alphabet,
            all_input_universal=self._all_input_universal,
            ip_universal_states=self._ip_universal_states,
        )

    def _bfs_expand(self, dfa, filt, worklist, seen, N):
        """BFS expansion of new DFA states with factored universality caching."""
        frontier = set()
        q_stops = set()
        r_stops = set()

        while worklist:
            i = worklist.popleft()
            if self._status.get(i, STATUS_NEW) != STATUS_NEW:
                continue

            # Compute FState for universality caching
            fstate = FState.from_nfa_states(i)
            mp = fstate.max_pos
            self._max_bufpos[i] = mp

            if mp >= N:
                frontier.add(i)

            if dfa.is_final(i):
                # Use factored universality with frontier caching
                if filt.is_universal(i, fstate, N):
                    self._status[i] = STATUS_QSTOP
                    self._trans[i] = {}
                    q_stops.add(i)
                    continue

            arcs = {}
            for a, j in dfa.arcs(i):
                arcs[a] = j
                self._incoming.setdefault(j, set()).add((a, i))
                if j not in seen and self._status.get(j, STATUS_NEW) == STATUS_NEW:
                    worklist.append(j)
                    seen.add(j)

            self._trans[i] = arcs
            if dfa.is_final(i):
                self._status[i] = STATUS_RSTOP
                r_stops.add(i)
            else:
                self._status[i] = STATUS_INTERIOR

        return frontier, q_stops, r_stops

    def _build_fresh(self):
        dfa = PrecoverNFA(self.fst, self.target).det()
        filt = self._make_filter(dfa)
        # Share frontier cache across filter instances
        filt._frontier_cache = self._frontier_cache

        self._trans = {}
        self._status = {}
        self._incoming = {}
        self._max_bufpos = {}
        self._filt = filt

        N = len(self.target)
        worklist = deque()
        seen = set()
        start_states = []

        for i in dfa.start():
            worklist.append(i)
            seen.add(i)
            start_states.append(i)
            self._incoming.setdefault(i, set())

        frontier, q_stops, r_stops = self._bfs_expand(dfa, filt, worklist, seen, N)

        # Persist eps closure cache
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            self._eps_cache = dfa.fsa._closure_cache

        self._start_states = start_states
        self._frontier = frontier
        self._q_stops = q_stops
        self._r_stops = r_stops

    def _build_incremental(self, parent):
        """Build from parent state by re-expanding only dirty+border states."""
        old_depth = len(parent.target)

        # Take ownership of parent's dicts
        self._trans = parent._trans
        self._status = parent._status
        self._incoming = parent._incoming
        self._max_bufpos = parent._max_bufpos

        # Transfer and evict stale eps closure cache
        if self._has_eps:
            eps_cache = parent._eps_cache
            stale = [k for k, v in eps_cache.items()
                     if len(k[1]) >= old_depth
                     or any(len(s[1]) >= old_depth for s in v)]
            for k in stale:
                del eps_cache[k]
            self._eps_cache = eps_cache
        else:
            self._eps_cache = {}

        dirty = parent._frontier
        border = set()
        for D in dirty:
            for (_, src) in self._incoming.get(D, ()):
                if src not in dirty:
                    border.add(src)

        invalidated = dirty | border
        for s in invalidated:
            old_arcs = self._trans.get(s, {})
            for label, dest in old_arcs.items():
                inc = self._incoming.get(dest)
                if inc is not None:
                    inc.discard((label, s))
            self._trans[s] = {}
            self._status[s] = STATUS_NEW

        q_stops = parent._q_stops - invalidated
        r_stops = parent._r_stops - invalidated

        dfa = PrecoverNFA(self.fst, self.target).det()

        # Inject persisted eps closure cache
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            dfa.fsa._closure_cache = self._eps_cache

        self._start_states = list(dfa.start())

        filt = self._make_filter(dfa)
        filt._frontier_cache = self._frontier_cache
        # Evict stale universality cache entries
        filt.evict_frontier(old_depth, self.target)
        self._filt = filt

        N = len(self.target)
        worklist = deque(invalidated)
        seen = set(invalidated)
        for s in self._start_states:
            self._incoming.setdefault(s, set())
            if s not in seen and self._status.get(s, STATUS_NEW) == STATUS_NEW:
                worklist.append(s)
                seen.add(s)

        frontier, new_q, new_r = self._bfs_expand(dfa, filt, worklist, seen, N)

        # Persist eps closure cache
        if self._has_eps and isinstance(dfa.fsa, EpsilonRemove):
            self._eps_cache = dfa.fsa._closure_cache

        self._frontier = frontier
        self._q_stops = q_stops | new_q
        self._r_stops = r_stops | new_r

    def _get_incoming(self, state):
        return self._incoming.get(state, ())

    @cached_property
    def quotient(self):
        return _trimmed_fsa(self._start_states, self._q_stops, self._get_incoming)

    @cached_property
    def remainder(self):
        return _trimmed_fsa(self._start_states, self._r_stops, self._get_incoming)

    def __rshift__(self, y):
        return FactoredDecomp(self.fst, self.target + (y,), parent=self)

    def decompose_next(self):
        """Decompose for every next target symbol.

        Each branch is built independently (non-incremental) to avoid
        mutating shared parent state. The >> operator remains available
        for sequential single-branch extension.
        """
        return {y: FactoredDecomp(self.fst, self.target + (y,))
                for y in sorted(self.target_alphabet)}
