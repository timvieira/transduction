"""Compute safe states and safe powerstates for finite decomposition analysis.

Implements Algorithms 1 and 2 from the revised Finite Decomposition Lemma
(reports/finite_decomposition_lemma.tex):

- ``compute_safe_states(fst)`` — Level 1: individual-state safety via SCC
  analysis on the state graph.  O(|S| + |T|), precomputable once per transducer.

- ``compute_safe_powersets(fst, seeds, budget)`` — Level 2: powerset safety via
  lazy BFS + SCC analysis on the powerset graph.  Bounded by ``budget``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from transduction.fsa import EPSILON
from transduction.universality import compute_ip_universal_states


# ============================================================
# Generic graph helpers
# ============================================================

def _cyclic_states(nodes, succs: dict) -> set:
    """Return nodes that lie on cycles (Tarjan's SCC on an arbitrary graph).

    Args:
        nodes: iterable of graph nodes.
        succs: adjacency dict mapping node -> iterable of successor nodes.

    Returns:
        Set of nodes in non-trivial SCCs or with self-loops.
    """
    index_counter = [0]
    stack: list = []
    on_stack: set = set()
    idx: dict = {}
    low: dict = {}
    cyclic: set = set()

    def strongconnect(v):
        idx[v] = low[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in succs.get(v, ()):
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                cyclic.update(scc)
            elif scc[0] in succs.get(scc[0], ()):
                cyclic.add(scc[0])

    for s in nodes:
        if s not in idx:
            strongconnect(s)

    return cyclic


def _backward_reachable(seeds, preds: dict) -> set:
    """BFS backward reachability from ``seeds`` through ``preds``.

    Args:
        seeds: iterable of starting nodes.
        preds: adjacency dict mapping node -> iterable of predecessor nodes.

    Returns:
        Set of all nodes reachable backward from seeds (including seeds).
    """
    reached = set(seeds)
    queue = deque(seeds)
    while queue:
        s = queue.popleft()
        for p in preds.get(s, ()):
            if p not in reached:
                reached.add(p)
                queue.append(p)
    return reached


def _build_preds(nodes, succs: dict) -> dict:
    """Build predecessor adjacency from successor adjacency.

    Args:
        nodes: iterable of graph nodes.
        succs: adjacency dict mapping node -> iterable of successor nodes.

    Returns:
        dict mapping node -> set of predecessor nodes.
    """
    preds: dict[Any, set] = defaultdict(set)
    for s in nodes:
        for j in succs.get(s, ()):
            preds[j].add(s)
    return preds


# ============================================================
# Level 1: individual-state safety
# ============================================================

def _compute_finite_closure_states(fst) -> frozenset:
    """Return the set of states with finite closure (no reachable cycle).

    Uses ``fst.strongly_connected_components()`` to identify cyclic states,
    then backward BFS to find all states that can reach a cycle.
    """
    sccs = fst.strongly_connected_components()

    # Build state-level adjacency (for self-loop detection + backward reach)
    succs: dict[Any, set] = defaultdict(set)
    for s in fst.states:
        for _a, _b, j in fst.arcs(s):
            succs[s].add(j)

    # Identify cyclic states from SCCs
    cyclic_states = set()
    for scc in sccs:
        if len(scc) > 1:
            cyclic_states.update(scc)
        elif scc[0] in succs.get(scc[0], ()):  # self-loop
            cyclic_states.add(scc[0])

    # States that can reach a cyclic state have infinite closure
    preds = _build_preds(fst.states, succs)
    can_reach_cycle = _backward_reachable(cyclic_states, preds)

    return frozenset(fst.states - can_reach_cycle)


def compute_safe_states(fst) -> frozenset:
    """Compute the set of safe states (Level 1: individual-state safety).

    A state is safe if it is:
    (a) ip-universal, or
    (b) has finite closure (no reachable cycle), or
    (c) all successors are safe.

    Equivalently, a state is safe iff it is a base case or cannot reach any
    cycle in the subgraph induced by non-base-case states.

    Returns a frozenset of safe states.  Runs in O(|S| + |T|) time.
    """
    # Step 1: base cases
    ip_universal = compute_ip_universal_states(fst)
    finite_closure = _compute_finite_closure_states(fst)
    base = ip_universal | finite_closure

    # Step 2: restricted graph (non-base-case states)
    non_base = fst.states - base
    if not non_base:
        return frozenset(fst.states)

    # Build adjacency for restricted graph
    restricted_succs: dict[Any, set] = defaultdict(set)
    for s in non_base:
        for _a, _b, j in fst.arcs(s):
            if j in non_base:
                restricted_succs[s].add(j)

    # Step 3: SCC + backward propagation on restricted graph
    cyclic = _cyclic_states(non_base, restricted_succs)
    restricted_preds = _build_preds(non_base, restricted_succs)
    unsafe = _backward_reachable(cyclic, restricted_preds)

    return frozenset(fst.states - unsafe)


# ============================================================
# Level 2: powerset safety
# ============================================================

def _is_collectively_ip_universal(fst, powerstate: frozenset, ip_universal_states: frozenset, source_alphabet: set) -> bool:
    """Check if a powerstate is collectively ip-universal.

    Uses a BFS over powerstates (same as is_cylinder in the paper):
    returns True iff every reachable powerstate is non-empty and contains
    at least one accepting state, and has successors on all input symbols.
    """
    # Fast path: any individual state is ip-universal
    if not powerstate.isdisjoint(ip_universal_states):
        return True

    visited = set()
    queue = deque()
    visited.add(powerstate)
    queue.append(powerstate)

    while queue:
        ps = queue.popleft()
        # Must contain an accepting state
        if not any(fst.is_final(s) for s in ps):
            return False
        # Must have successors on all source symbols
        for x in source_alphabet:
            next_ps = frozenset(
                j for s in ps for _b, j in fst.arcs(s, x)
            )
            if not next_ps:
                return False
            if next_ps not in visited:
                visited.add(next_ps)
                queue.append(next_ps)
    return True


def compute_safe_powersets(fst, seeds: list[frozenset], budget: int = 2**20) -> set[frozenset]:
    """Compute safe powerstates (Level 2: powerset safety).

    Performs a lazy BFS over the powerset graph starting from ``seeds``,
    then runs SCC analysis to identify safe powerstates.

    Args:
        fst: The transducer.
        seeds: Powerstates to start exploration from.
        budget: Maximum number of powerstates to explore.

    Returns:
        A set of frozensets (powerstates) that are certified safe.
    """
    source_alphabet = fst.A - {EPSILON}
    ip_universal = compute_ip_universal_states(fst)
    finite_closure = _compute_finite_closure_states(fst)

    explored: dict[frozenset, dict] = {}  # powerstate -> {x: successor_ps}
    base_cases: set[frozenset] = set()
    queue = deque(seeds)
    queued = set(seeds)

    while queue and len(explored) < budget:
        ps = queue.popleft()
        if ps in explored or not ps:
            continue
        explored[ps] = {}

        # Check base cases
        if _is_collectively_ip_universal(fst, ps, ip_universal, source_alphabet):
            base_cases.add(ps)
            continue
        if ps <= finite_closure:  # all states in ps have finite closure
            base_cases.add(ps)
            continue

        # Expand successors
        for x in source_alphabet:
            next_ps = frozenset(
                j for s in ps for _b, j in fst.arcs(s, x)
            )
            explored[ps][x] = next_ps
            if next_ps and next_ps not in explored and next_ps not in queued:
                queue.append(next_ps)
                queued.add(next_ps)

    # SCC analysis on explored non-base-case powerstates
    non_base = set(explored.keys()) - base_cases

    # Build adjacency
    ps_succs: dict[frozenset, set] = defaultdict(set)
    for ps in non_base:
        for _x, nps in explored[ps].items():
            if nps in non_base:
                ps_succs[ps].add(nps)

    # Find cyclic powerstates
    cyclic = _cyclic_states(non_base, ps_succs)

    # Initial unsafe seeds: cyclic + has unexplored successors
    unsafe_seeds: set[frozenset] = set(cyclic)
    for ps in non_base:
        for _x, nps in explored[ps].items():
            if nps and nps not in explored:
                unsafe_seeds.add(ps)
                break

    # Backward propagation from all unsafe seeds
    ps_preds = _build_preds(non_base, ps_succs)
    unsafe = _backward_reachable(unsafe_seeds, ps_preds)

    return (set(explored.keys()) | base_cases) - unsafe


# ============================================================
# Frontier computation
# ============================================================

def compute_frontier(fst, target: tuple) -> frozenset:
    """Compute the frontier F(t): states reachable along paths emitting target.

    For an input-determinized transducer, this is the set of states reachable
    from initial states along arcs whose output labels concatenate to ``target``.
    """
    fst.ensure_trie_index()
    N = len(target)

    initial = {(s, 0) for s in fst.start}

    visited = set(initial)
    queue = deque(initial)
    frontier_states = set()

    while queue:
        s, pos = queue.popleft()

        if pos == N:
            frontier_states.add(s)
            # Follow eps:eps arcs (don't advance position)
            for a, b, j in fst.arcs(s):
                if b == EPSILON or b == '':
                    new = (j, pos)
                    if a == EPSILON or a == '':
                        if new not in visited:
                            visited.add(new)
                            queue.append(new)
                            frontier_states.add(j)

        if pos < N:
            for a, b, j in fst.arcs(s):
                if b == EPSILON or b == '':
                    # eps-output: advance state but not position
                    new = (j, pos)
                    if new not in visited:
                        visited.add(new)
                        queue.append(new)
                elif b == target[pos]:
                    # matching output: advance position
                    new = (j, pos + 1)
                    if new not in visited:
                        visited.add(new)
                        queue.append(new)

    return frozenset(frontier_states)
