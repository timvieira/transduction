"""Compiled beam search via static FST region decomposition.

Statically analyzes an FST to partition states into *regions* — connected
subgraphs with uniform computational character — then dispatches hypotheses
to specialized scoring code per region type.  At decode time, hypotheses
are grouped by region and processed in batches ("horizontal fusion"), while
multi-step deterministic chains are collapsed into table lookups ("vertical
fusion").

Region types (from most to least specialized):

- **HubRegion**: IP-universal + final + output-deterministic, all paths return
  to the hub.  Scoring via sparse matvec over a precomputed OutputTrie.
  No runtime BFS.

- **CorridorRegion**: Deterministic chain of states — each has exactly one
  non-epsilon arc per source symbol, powerset stays size 1.  Multi-step
  chains are collapsed into a single table lookup (vertical fusion).

- **UniversalPlateauRegion**: Subgraph where every reachable DFA state is
  IP-universal.  Skips universality checks, reducing per-state cost.

- **WildRegion**: General case.  Falls back to particle-based best-first
  search with lazy DFA (same as FusedTransducedLM).

Usage::

    from transduction.lm.compiled_beam import CompiledBeam
    from transduction.lm.ngram import CharNgramLM
    from transduction import examples

    inner = CharNgramLM.train("hello world", n=2)
    fst = examples.lowercase()
    cb = CompiledBeam(inner, fst, K=10)
    state = cb.initial()
    print(state.logp_next)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from functools import cached_property
from typing import Any

from transduction.fst import FST, EPSILON
from transduction.lm.base import LM, LMState, Token
from transduction.lm.transduced import Particle, _select_top_k
from transduction.lm.fused_transduced import _FusedSearch
from transduction.lm.generalized_beam import (
    OutputTrie, _compute_hub_vocab, _FusedSearchAdapter,
)
from transduction.universality import compute_ip_universal_states
from transduction.util import LogDistr, LogVector, Str, logsumexp


# ---------------------------------------------------------------------------
# Hyp — a single hypothesis
# ---------------------------------------------------------------------------

class Hyp:
    """A single hypothesis: LM state + weight + region-specific state."""
    __slots__ = ('lm_state', 'weight', 'region_state')

    def __init__(self, lm_state: LMState, weight: float, region_state: Any) -> None:
        self.lm_state = lm_state
        self.weight = weight
        self.region_state = region_state     # HubRegion: trie node (int)
                                             # WildRegion: dfa_state


class ScoredOutput:
    """Output of Region.score(): symbol scores + carry-forward."""
    __slots__ = ('scores', 'eos_score', 'carry')

    def __init__(self) -> None:
        self.scores: LogVector[Token] = LogVector()
        self.eos_score: float = float('-inf')
        self.carry: dict[Token, list[tuple[Region, Any]]] = defaultdict(list)

    def _logaddexp_eos(self, v: float) -> None:
        a, b = self.eos_score, v
        if a < b: a, b = b, a
        self.eos_score = a + math.log1p(math.exp(b - a)) if b > float('-inf') else a

    def merge(self, other: ScoredOutput) -> None:
        for sym, w in other.scores.items():
            self.scores.logaddexp(sym, w)
        self._logaddexp_eos(other.eos_score)
        for sym, items in other.carry.items():
            self.carry[sym].extend(items)


# ---------------------------------------------------------------------------
# Region — abstract interface
# ---------------------------------------------------------------------------

class Region(ABC):
    """A region of the FST with uniform computational character.

    - **score**: Given hypotheses, compute per-output-symbol scores and
      carry-forward data.  All hypotheses in the batch are processed together
      (horizontal fusion).

    - **advance**: Given carry-forward data for a chosen output symbol,
      produce new (region, Hyp) pairs for the next step.
    """

    @abstractmethod
    def score(self, hyps: list[Hyp], target: Str[Token],
              compiled: CompiledBeam) -> ScoredOutput:
        ...

    @abstractmethod
    def advance(self, carry_items: list, y: Token,
                compiled: CompiledBeam) -> list[tuple[Region, Hyp]]:
        ...


# ---------------------------------------------------------------------------
# HubRegion — trie-mass scoring with horizontal batching
# ---------------------------------------------------------------------------

class HubRegion(Region):
    """Region at an IP-universal accepting hub with deterministic output.

    Scoring: for each hyp k at trie node n_k with weight w_k and log-mass
    vector m_k, the contribution to output symbol y is::

        score[y] += exp(w_k + m_k[child(n_k, y)] - m_k[n_k])

    Horizontal batching: when multiple hyps share the same LM state,
    log_mass_sum is computed once and reused.  Prefetch batches forward
    passes across distinct LM states.
    """

    def __init__(self, hub_state: Any, trie: OutputTrie,
                 member_states: set | None = None) -> None:
        self.hub_state = hub_state
        self.trie = trie
        # All FST states belonging to this hub (hub + epsilon-reachable)
        self.member_states = member_states or {hub_state}

    def __repr__(self) -> str:
        return f'HubRegion(hub={self.hub_state})'

    def score(self, hyps: list[Hyp], target: Str[Token],
              compiled: CompiledBeam) -> ScoredOutput:
        out = ScoredOutput()
        trie = self.trie

        # Compute log_mass for each hyp (batched across shared LM states)
        mass_vectors = self._batched_log_mass(hyps, compiled)

        # EOT extension: hyps at trie nodes with a None child have completed
        # a source token.  Extend them to hub, corridor/plateau, or particle
        # destinations.
        extended_hub_hyps, exit_particles, non_hub_exits = self._extend_eot(
            hyps, mass_vectors, compiled,
        )

        # Collect all items to score: original hyps + newly extended hub hyps.
        # Each item is (trie_node, weight, log_mass_vector, lm_state).
        all_items: list[tuple[int, float, Any, LMState]] = []
        for k, h in enumerate(hyps):
            all_items.append((h.region_state, h.weight, mass_vectors[k], h.lm_state))
        all_items.extend(extended_hub_hyps)

        # Score trie children (horizontal: all items together)
        for node, w, log_mass, lm_state in all_items:
            node_mass = float(log_mass[node])
            for y, child_node in trie.children[node].items():
                if y is None:
                    continue
                child_mass = float(log_mass[child_node])
                out.scores.logaddexp(y, w + child_mass - node_mass)

            # EOS: hyps at trie root contribute P_inner(EOS)
            if node == trie.root:
                eos_lp = lm_state.logp_next[compiled.inner_lm.eos]
                if eos_lp > float('-inf'):
                    out._logaddexp_eos(w + eos_lp)

        # Carry-forward: for each item, record data for advancing on each y.
        # Store (parent_node, child_node, weight, log_mass, lm_state) so
        # advance() can recompute the correct weight.
        for node, w, log_mass, lm_state in all_items:
            for y, child_node in trie.children[node].items():
                if y is None:
                    continue
                out.carry[y].append((
                    self,
                    (lm_state, node, child_node, w, log_mass),
                ))

        # Particles from hub exits go to wild region
        if exit_particles:
            for p in exit_particles:
                out.carry['__particles__'].append((None, p))

        # Non-hub exits (corridor/plateau): score inline
        for new_lm, dest_state, w_prime, dest_region in non_hub_exits:
            _score_from_state(new_lm, dest_state, w_prime, compiled, out)

        return out

    def _batched_log_mass(self, hyps: list[Hyp],
                          compiled: CompiledBeam) -> list:
        """Compute trie log-mass vectors, deduplicating by LM state."""
        seen: dict[int, int] = {}
        unique_states: list[LMState] = []
        hyp_to_unique: list[int] = []

        for h in hyps:
            key = h.lm_state.context_key
            if key not in seen:
                seen[key] = len(unique_states)
                unique_states.append(h.lm_state)
            hyp_to_unique.append(seen[key])

        if len(unique_states) > 1:
            compiled.inner_lm.prefetch(unique_states)

        unique_masses = self.trie.log_mass_sum_batch(
            [s.logp_next for s in unique_states])
        return [unique_masses[hyp_to_unique[k]] for k in range(len(hyps))]

    def _extend_eot(self, hyps: list[Hyp], mass_vectors: list,
                    compiled: CompiledBeam,
                    ) -> tuple[list[tuple], list[Particle]]:
        """Extend hyps at end-of-token boundaries.

        3-phase prepare/prefetch/complete pattern for batched forward passes.
        """
        trie = self.trie

        # Phase 1: Prepare
        hub_pending: list[tuple[LMState, Any, OutputTrie, float]] = []
        non_hub_pending: list[tuple[LMState, Any, float, Region]] = []
        exit_particles: list[Particle] = []
        needs_logp: list[LMState] = []

        for k, h in enumerate(hyps):
            children = trie.children[h.region_state]
            if None not in children:
                continue

            eot_node = children[None]
            src_sym, dest_state = trie.leaf_info[eot_node]
            node_mass = float(mass_vectors[k][h.region_state])
            eot_mass = float(mass_vectors[k][eot_node])
            w_prime = h.weight + eot_mass - node_mass
            if w_prime <= float('-inf'):
                continue

            new_lm_state = h.lm_state >> src_sym

            dest_region = compiled._region_map.region_for(dest_state)
            if isinstance(dest_region, HubRegion):
                hub_pending.append((new_lm_state, dest_state, dest_region.trie, w_prime))
                needs_logp.append(new_lm_state)
            elif isinstance(dest_region, (CorridorRegion, UniversalPlateauRegion)):
                # Route to corridor/plateau region
                non_hub_pending.append((new_lm_state, dest_state, w_prime, dest_region))
                needs_logp.append(new_lm_state)
            elif compiled._wild_region is not None:
                exit_particles.append(
                    Particle(None, new_lm_state, w_prime, (src_sym,))
                )

        # Phase 2: Prefetch
        if needs_logp:
            compiled.inner_lm.prefetch(needs_logp)

        # Phase 3: Complete — batch log_mass_sum by dest_trie
        hub_items = []
        by_trie: dict[int, list[int]] = defaultdict(list)
        for idx, (new_lm_state, dest_state, dest_trie, w_prime) in enumerate(hub_pending):
            by_trie[id(dest_trie)].append(idx)

        mass_results: dict[int, Any] = {}
        for trie_id, indices in by_trie.items():
            dest_trie = hub_pending[indices[0]][2]
            logp_nexts = [hub_pending[i][0].logp_next for i in indices]
            batch_masses = dest_trie.log_mass_sum_batch(logp_nexts)
            for k, idx in enumerate(indices):
                mass_results[idx] = batch_masses[k]

        for idx, (new_lm_state, dest_state, dest_trie, w_prime) in enumerate(hub_pending):
            hub_items.append((dest_trie.root, w_prime,
                              mass_results[idx], new_lm_state))

        # Non-hub exits: (lm_state, fst_state, weight, region)
        non_hub_exits = []
        for new_lm_state, dest_state, w_prime, region in non_hub_pending:
            non_hub_exits.append((new_lm_state, dest_state, w_prime, region))

        return hub_items, exit_particles, non_hub_exits

    def advance(self, carry_items: list, y: Token,
                compiled: CompiledBeam) -> list[tuple[Region, Hyp]]:
        """Advance hub hyps by descending the trie on symbol y."""
        result = []
        for _region, (lm_state, parent_node, child_node, w, log_mass) in carry_items:
            # Weight for the advanced hyp: parent weight + transition score
            new_w = w + float(log_mass[child_node]) - float(log_mass[parent_node])
            result.append((self, Hyp(lm_state, new_w, child_node)))
        return result


# ---------------------------------------------------------------------------
# WildRegion — particle-based best-first search (general fallback)
# ---------------------------------------------------------------------------

class WildRegion(Region):
    """General-case region: particle expansion via lazy DFA.

    Delegates to _FusedSearch.  No horizontal vectorization within this
    region (best-first search is inherently sequential).
    """

    def __repr__(self) -> str:
        return 'WildRegion()'

    def score(self, hyps: list[Hyp], target: Str[Token],
              compiled: CompiledBeam) -> ScoredOutput:
        out = ScoredOutput()

        # region_state is a Particle (preserves dfa_state + source_path)
        particles = [
            Particle(h.region_state.dfa_state, h.lm_state, h.weight,
                     h.region_state.source_path)
            for h in hyps
        ]

        if not particles or compiled._fused_adapter is None:
            return out

        search = _FusedSearch(compiled._fused_adapter, target, particles)
        p_scores, p_eos, p_carry = search.search()

        for sym, w in p_scores.items():
            out.scores.logaddexp(sym, w)
        out.eos_score = p_eos
        for sym, ps in p_carry.items():
            for p in ps:
                out.carry[sym].append((self, p))

        return out

    def advance(self, carry_items: list, y: Token,
                compiled: CompiledBeam) -> list[tuple[Region, Hyp]]:
        """Advance particles from carry-forward."""
        result = []
        for _region, p in carry_items:
            if isinstance(p, Particle):
                # Store full Particle as region_state to preserve source_path
                result.append((self, Hyp(p.lm_state, p.log_weight, p)))
        return result


# ---------------------------------------------------------------------------
# CorridorRegion — deterministic chain with table-lookup transitions
# ---------------------------------------------------------------------------

class CorridorRegion(Region):
    """Deterministic corridor: output-deterministic chain of states.

    Each FST state in the corridor has at most one non-epsilon output per
    source symbol, so the powerset stays size 1.

    Two kinds of corridor arcs:

    1. **Non-epsilon input** (src_sym → (output, dest)):
       Consume src_sym from the LM, produce output.  Weight is
       P_LM(σ) * ReachableMass(dest, LM>>σ) to account for the total
       probability mass reachable from the destination.

    2. **Epsilon input** (EPSILON → (output, dest)):
       Produce output deterministically without consuming input (probability 1).
       These arise in epsilon chains like newspeak2 states 5→6→7→0.

    Attributes:
        states: set of FST states in this corridor.
        table: {state: {src_sym_or_EPSILON: (output_or_None, dest_state)}}.
    """

    def __init__(self, states: set, table: dict) -> None:
        self.states = states
        self.table = table   # {state: {src_sym: (output, dest_state)}}

    def __repr__(self) -> str:
        return f'CorridorRegion(states={self.states})'

    def score(self, hyps: list[Hyp], target: Str[Token],
              compiled: CompiledBeam) -> ScoredOutput:
        out = ScoredOutput()

        for h in hyps:
            fst_state = h.region_state
            transitions = self.table.get(fst_state, {})

            if EPSILON in transitions:
                # Epsilon-input corridor: produce output deterministically
                output_sym, dest_state = transitions[EPSILON]
                if output_sym is not None:
                    out.scores.logaddexp(output_sym, h.weight)
                    out.carry[output_sym].append((
                        self, (h.lm_state, EPSILON, dest_state, h.weight),
                    ))
                continue

            # Non-epsilon input arcs: weight by dest reachable mass
            lm_logp = h.lm_state.logp_next

            for src_sym, (output_sym, dest_state) in transitions.items():
                logp = lm_logp[src_sym]
                if logp <= float('-inf'):
                    continue

                new_lm = h.lm_state >> src_sym
                dest_mass = _reachable_log_mass(
                    dest_state, new_lm, compiled)
                if dest_mass <= float('-inf'):
                    continue

                # Raw weight (without dest mass) for carry-forward
                w_raw = h.weight + logp
                # Mass-weighted weight for scoring
                w_scored = w_raw + dest_mass

                if output_sym is not None:
                    out.scores.logaddexp(output_sym, w_scored)
                    # Carry uses raw weight so next step doesn't double-count
                    out.carry[output_sym].append((
                        self, (h.lm_state, src_sym, dest_state, w_raw),
                    ))
                else:
                    # Epsilon output: source consumed, no output yet.
                    # Re-score from destination with raw weight.
                    _score_from_state(new_lm, dest_state, w_raw, compiled, out)

            # EOS contribution
            eos_lp = lm_logp[compiled.inner_lm.eos]
            if eos_lp > float('-inf') and compiled.fst.is_final(fst_state):
                out._logaddexp_eos(h.weight + eos_lp)

        return out

    def advance(self, carry_items: list, y: Token,
                compiled: CompiledBeam) -> list[tuple[Region, Hyp]]:
        result = []
        for _region, (lm_state, src_sym, dest_state, w) in carry_items:
            if src_sym == EPSILON:
                new_lm = lm_state
            else:
                new_lm = lm_state >> src_sym
            result.extend(_route_hyp(new_lm, w, dest_state, compiled))
        return result


# ---------------------------------------------------------------------------
# UniversalPlateauRegion — all-universal subgraph, skip universality checks
# ---------------------------------------------------------------------------

def _reachable_log_mass(fst_state: Any, lm_state: LMState,
                        compiled: CompiledBeam,
                        depth: int = 0,
                        _visited: set | None = None) -> float:
    """Compute log of total reachable mass from (fst_state, lm_state).

    ReachableMass(q, s) = P_s(EOS)*[q final] + Σ_σ P_s(σ)*RM(dest(σ), s>>σ)

    For hub states, RM ≈ 1.0 (IP-universal + full source alphabet).
    For corridors, computed by recursion with cycle detection.
    For wild/unknown states, approximated as 1.0 (conservative).
    """
    if depth > 15:
        # Depth limit: approximate remaining mass as what we've seen
        return 0.0 if compiled.fst.is_final(fst_state) else float('-inf')

    if _visited is None:
        _visited = set()

    # Hub: total reachable mass ≈ 1.0
    if fst_state in compiled._hub_regions:
        return 0.0  # log(1.0)

    # Cycle detection for corridors
    state_key = lm_state.context_key, fst_state
    if state_key in _visited:
        # Fixpoint approximation: P(EOS) from this state if final, else 0
        if compiled.fst.is_final(fst_state):
            return lm_state.logp_next[compiled.inner_lm.eos]
        return float('-inf')
    _visited.add(state_key)

    fst = compiled.fst
    lm_logp = lm_state.logp_next

    terms = []

    # EOS contribution
    if fst.is_final(fst_state):
        eos_lp = lm_logp[compiled.inner_lm.eos]
        if eos_lp > float('-inf'):
            terms.append(eos_lp)

    # Arc contributions
    for a, b, j in fst.arcs(fst_state):
        if a == EPSILON:
            # Epsilon-input arc: reachable mass = reachable mass at dest
            rm = _reachable_log_mass(j, lm_state, compiled, depth + 1, _visited)
            if rm > float('-inf'):
                terms.append(rm)
        else:
            logp = lm_logp[a]
            if logp <= float('-inf'):
                continue
            new_lm = lm_state >> a
            rm = _reachable_log_mass(j, new_lm, compiled, depth + 1, _visited)
            if rm > float('-inf'):
                terms.append(logp + rm)

    _visited.discard(state_key)

    if not terms:
        return float('-inf')
    return logsumexp(terms)


def _score_from_state(lm_state: LMState, fst_state: Any, w: float,
                      compiled: CompiledBeam, out: ScoredOutput) -> None:
    """Score next output from a given FST state (used for epsilon-output chase)."""
    dest_region = compiled._region_map.region_for(fst_state)
    if isinstance(dest_region, HubRegion):
        hub_hyp = Hyp(lm_state, w, dest_region.trie.root)
        out.merge(dest_region.score([hub_hyp], (), compiled))
    elif isinstance(dest_region, (CorridorRegion, UniversalPlateauRegion)):
        hyp = Hyp(lm_state, w, fst_state)
        out.merge(dest_region.score([hyp], (), compiled))
    elif compiled._wild_region is not None:
        p = Particle(None, lm_state, w, ())
        wild_hyp = Hyp(lm_state, w, p)
        out.merge(compiled._wild_region.score([wild_hyp], (), compiled))


def _route_hyp(lm_state: LMState, w: float, dest_state: Any,
               compiled: CompiledBeam) -> list[tuple[Region, Hyp]]:
    """Route a hypothesis to its destination region."""
    dest_region = compiled._region_map.region_for(dest_state)
    if isinstance(dest_region, HubRegion):
        return [(dest_region, Hyp(lm_state, w, dest_region.trie.root))]
    elif isinstance(dest_region, (CorridorRegion, UniversalPlateauRegion)):
        return [(dest_region, Hyp(lm_state, w, dest_state))]
    elif compiled._wild_region is not None:
        p = Particle(None, lm_state, w, ())
        return [(compiled._wild_region, Hyp(lm_state, w, p))]
    return []


class UniversalPlateauRegion(Region):
    """Subgraph where every reachable DFA state is IP-universal.

    Within this region, every non-epsilon arc produces a quotient symbol.
    The universality check is skipped entirely (statically guaranteed).
    Scoring: for each source symbol σ with arc (σ, y, dest), contribute
    p(σ|LM) to score[y].  Epsilon-output arcs are chased to the next
    output-producing state.

    Attributes:
        states: set of FST states in this plateau.
        ip_universal: set of IP-universal FST states.
    """

    def __init__(self, states: set, ip_universal: set) -> None:
        self.states = states
        self.ip_universal = ip_universal

    def __repr__(self) -> str:
        return f'UniversalPlateauRegion(states={len(self.states)})'

    def score(self, hyps: list[Hyp], target: Str[Token],
              compiled: CompiledBeam) -> ScoredOutput:
        out = ScoredOutput()
        fst = compiled.fst

        for h in hyps:
            fst_state = h.region_state
            lm_logp = h.lm_state.logp_next

            # All arcs from a universal state contribute to quotient
            for a, b, j in fst.arcs(fst_state):
                if a == EPSILON:
                    continue
                logp = lm_logp[a]
                if logp <= float('-inf'):
                    continue

                new_lm = h.lm_state >> a
                dest_mass = _reachable_log_mass(j, new_lm, compiled)
                if dest_mass <= float('-inf'):
                    continue

                # Raw weight (without dest mass) for carry-forward
                w_raw = h.weight + logp
                # Mass-weighted weight for scoring
                w_scored = w_raw + dest_mass

                if b != EPSILON:
                    out.scores.logaddexp(b, w_scored)
                    # Carry uses raw weight so next step doesn't double-count
                    out.carry[b].append((
                        self, (h.lm_state, a, j, w_raw),
                    ))
                else:
                    # Epsilon output: chase with raw weight
                    _score_from_state(new_lm, j, w_raw, compiled, out)

            # EOS contribution
            if fst.is_final(fst_state):
                eos_lp = lm_logp[compiled.inner_lm.eos]
                if eos_lp > float('-inf'):
                    out._logaddexp_eos(h.weight + eos_lp)

        return out

    def advance(self, carry_items: list, y: Token,
                compiled: CompiledBeam) -> list[tuple[Region, Hyp]]:
        result = []
        for _region, (lm_state, src_sym, dest_state, w) in carry_items:
            new_lm = lm_state >> src_sym
            result.extend(_route_hyp(new_lm, w, dest_state, compiled))
        return result


def _hub_member_states(fst: FST, hub: Any) -> set:
    """BFS from hub following epsilon-input arcs; return all visited states."""
    from collections import deque
    visited = {hub}
    queue = deque([hub])
    while queue:
        state = queue.popleft()
        for a, b, j in fst.arcs(state):
            if a == EPSILON and j not in visited:
                visited.add(j)
                queue.append(j)
    return visited


# ---------------------------------------------------------------------------
# RegionAnalyzer — static FST → region map
# ---------------------------------------------------------------------------

class RegionAnalyzer:
    """Statically analyze an FST to identify regions.

    Identifies:
    - Hub regions (IP-universal + final + deterministic output)
    - Corridor regions (output-deterministic chains)
    - Universal plateau regions (all-IP-universal subgraphs)
    - Wild region (everything else)
    """

    def __init__(self, fst: FST, inner_lm: LM) -> None:
        self.fst = fst
        self.inner_lm = inner_lm

    def analyze(self) -> RegionMap:
        fst = self.fst
        hub_regions: dict[Any, HubRegion] = {}
        corridor_regions: dict[Any, CorridorRegion] = {}
        plateau_regions: dict[Any, UniversalPlateauRegion] = {}

        # Fast path for BPE-like FSTs: single start=stop state
        if len(fst.start) == 1 and set(fst.start) == set(fst.stop):
            hub = next(iter(fst.start))
            entries = _compute_hub_vocab(fst, hub)
            if entries is not None:
                entry_syms = {src for src, _, _ in entries}
                source_alpha = fst.A - {EPSILON}
                all_return = all(dest == hub for _, _, dest in entries)
                if entry_syms == source_alpha and all_return:
                    trie = OutputTrie(entries, self.inner_lm.eos)
                    if not trie.is_empty:
                        members = _hub_member_states(fst, hub)
                        hub_regions[hub] = HubRegion(hub, trie, members)
                        return RegionMap(
                            hub_regions, corridor_regions,
                            plateau_regions, None,
                        )

        # General path: compute IP-universal states
        try:
            import transduction_core
            from transduction.rust_bridge import to_rust_fst
            rust_fst, sym_map, state_map = to_rust_fst(fst)
            inv_sym = {v: k for k, v in sym_map.items()}
            inv_state = {v: k for k, v in state_map.items()}

            ip_univ_vec = transduction_core.rust_compute_ip_universal_states(rust_fst)
            ip_univ = {inv_state[i] for i, v in enumerate(ip_univ_vec) if v}
        except ImportError:
            ip_univ = compute_ip_universal_states(fst)

        # --- Hub detection ---
        hubs = {q for q in ip_univ if fst.is_final(q)}
        for hub in hubs:
            entries = _compute_hub_vocab(fst, hub)
            if entries is not None:
                trie = OutputTrie(entries, self.inner_lm.eos)
                if not trie.is_empty:
                    members = _hub_member_states(fst, hub)
                    hub_regions[hub] = HubRegion(hub, trie, members)

        # All states belonging to any hub region
        classified: set = set()
        for hr in hub_regions.values():
            classified |= hr.member_states

        # --- Corridor detection ---
        # A state is a corridor state if:
        # 1. It's not already a hub
        # 2. It has output-deterministic arcs (each source sym -> unique output)
        # 3. Each source sym has exactly one destination
        source_alpha = fst.A - {EPSILON}
        for q in fst.states:
            if q in classified:
                continue
            arcs = fst.arcs(q)
            # Check output determinism: group non-epsilon-input arcs by input
            by_input: dict[Any, list[tuple]] = defaultdict(list)
            for a, b, j in arcs:
                if a != EPSILON:
                    by_input[a].append((b, j))
            is_deterministic = all(len(v) == 1 for v in by_input.values())
            # Also detect epsilon-only states (deterministic epsilon chains)
            eps_arcs = [(b, j) for a, b, j in arcs if a == EPSILON]
            is_eps_only = len(by_input) == 0 and len(eps_arcs) == 1
            if is_deterministic and (by_input or is_eps_only):
                table = {}
                for src_sym, [(out, dest)] in by_input.items():
                    table[src_sym] = (out if out != EPSILON else None, dest)
                if is_eps_only:
                    # Store epsilon transition in the table with a sentinel key
                    out, dest = eps_arcs[0]
                    table[EPSILON] = (out if out != EPSILON else None, dest)
                corridor = CorridorRegion({q}, {q: table})
                corridor_regions[q] = corridor
                classified.add(q)

        # --- Universal plateau detection ---
        # States that are IP-universal but not hubs or corridors
        for q in ip_univ:
            if q not in classified:
                plateau_regions[q] = UniversalPlateauRegion({q}, ip_univ)
                classified.add(q)

        # Determine if wild region is needed — only for states that are
        # not covered by hub, corridor, or plateau regions.
        wild_needed = any(q not in classified for q in fst.states)

        wild_region = WildRegion() if wild_needed else None
        return RegionMap(hub_regions, corridor_regions,
                         plateau_regions, wild_region)


class RegionMap:
    """Maps FST states to regions."""

    def __init__(self, hub_regions: dict[Any, HubRegion],
                 corridor_regions: dict[Any, CorridorRegion],
                 plateau_regions: dict[Any, UniversalPlateauRegion],
                 wild_region: WildRegion | None) -> None:
        self.hub_regions = hub_regions
        self.corridor_regions = corridor_regions
        self.plateau_regions = plateau_regions
        self.wild_region = wild_region

    def region_for(self, fst_state: Any) -> Region:
        if fst_state in self.hub_regions:
            return self.hub_regions[fst_state]
        if fst_state in self.corridor_regions:
            return self.corridor_regions[fst_state]
        if fst_state in self.plateau_regions:
            return self.plateau_regions[fst_state]
        if self.wild_region is not None:
            return self.wild_region
        raise ValueError(f"No region for state {fst_state}")

    def summary(self) -> str:
        parts = []
        if self.hub_regions:
            parts.append(f"{len(self.hub_regions)} hub(s)")
        if self.corridor_regions:
            parts.append(f"{len(self.corridor_regions)} corridor(s)")
        if self.plateau_regions:
            parts.append(f"{len(self.plateau_regions)} plateau(s)")
        if self.wild_region:
            parts.append("wild")
        return ', '.join(parts) if parts else 'empty'

    def detailed_summary(self) -> str:
        """Multi-line summary with per-region details."""
        lines = [f"RegionMap: {self.summary()}"]
        for state, r in self.hub_regions.items():
            n = len(r.trie._source_syms)
            n_nodes = len(r.trie.children)
            lines.append(f"  HubRegion({state}): {n} tokens, {n_nodes} trie nodes")
        for state, r in self.corridor_regions.items():
            n_arcs = sum(len(t) for t in r.table.values())
            lines.append(f"  CorridorRegion({state}): {n_arcs} arcs")
        for state, r in self.plateau_regions.items():
            lines.append(f"  UniversalPlateauRegion({state})")
        if self.wild_region:
            lines.append("  WildRegion (fallback)")
        return '\n'.join(lines)

    def state_classification(self) -> dict[Any, str]:
        """Return dict mapping each classified state to its region type name."""
        result = {}
        for hr in self.hub_regions.values():
            for s in hr.member_states:
                result[s] = 'hub'
        for s in self.corridor_regions:
            result[s] = 'corridor'
        for s in self.plateau_regions:
            result[s] = 'plateau'
        return result

    def _repr_html_(self) -> str:
        """Jupyter visualization of the region map."""
        return render_region_map_html(self)


# ---------------------------------------------------------------------------
# _CompiledBundle — orchestrator: groups hyps by region, scores, advances
# ---------------------------------------------------------------------------

class _CompiledBundle:
    """Holds hypotheses across regions, scores uniformly, advances."""

    def __init__(self, compiled: CompiledBeam,
                 hyps_by_region: dict[Region, list[Hyp]],
                 target: Str[Token]) -> None:
        self.compiled = compiled
        self.hyps_by_region = hyps_by_region
        self._target = target

    @classmethod
    def initial(cls, compiled: CompiledBeam) -> _CompiledBundle:
        return cls(compiled, dict(compiled._initial_hyps_by_region), ())

    @cached_property
    def _scored(self) -> tuple[LogDistr[Token], dict[Token, list[tuple[Region, Any]]]]:
        merged = ScoredOutput()

        # Score each region's hypotheses
        for region, hyps in self.hyps_by_region.items():
            if hyps:
                result = region.score(hyps, self._target, self.compiled)
                merged.merge(result)

        # Handle particles from hub EOT exits → score via wild region
        particle_items = merged.carry.pop('__particles__', [])
        if particle_items and self.compiled._wild_region is not None:
            particles_as_hyps = [
                Hyp(p.lm_state, p.log_weight, p)
                for _, p in particle_items
            ]
            wild = self.compiled._wild_region
            wild_result = wild.score(particles_as_hyps, self._target, self.compiled)
            merged.merge(wild_result)

        merged.scores[self.compiled.eos] = merged.eos_score
        logp_next = merged.scores.normalize()
        return logp_next, merged.carry

    @property
    def logp_next(self) -> LogDistr[Token]:
        return self._scored[0]

    @property
    def carry_forward(self) -> dict[Token, list[tuple[Region, Any]]]:
        return self._scored[1]

    def advance(self, y: Token) -> _CompiledBundle:
        carry = self.carry_forward.get(y, [])

        # Group carry by region
        by_region: dict[Region, list] = defaultdict(list)
        for item in carry:
            region = item[0]
            if region is not None:
                by_region[region].append(item)

        # Dispatch advance to each region
        new_hyps: list[tuple[Region, Hyp]] = []
        for region, items in by_region.items():
            new_hyps.extend(region.advance(items, y, self.compiled))

        # Global pruning across regions
        K = self.compiled.K
        if K is not None and len(new_hyps) > K:
            new_hyps.sort(key=lambda rh: -rh[1].weight)
            new_hyps = new_hyps[:K]

        # Also prune wild particles separately
        wild = self.compiled._wild_region
        if wild is not None:
            wild_hyps = [h for r, h in new_hyps if r is wild]
            if len(wild_hyps) > self.compiled.max_beam:
                wild_hyps.sort(key=lambda h: -h.weight)
                wild_hyps = set(id(h) for h in wild_hyps[:self.compiled.max_beam])
                new_hyps = [(r, h) for r, h in new_hyps
                            if r is not wild or id(h) in wild_hyps]

        # Group into hyps_by_region, merging hub hyps with same context_key.
        grouped: dict[Region, list[Hyp]] = defaultdict(list)
        for region, hyp in new_hyps:
            if isinstance(region, HubRegion):
                # Merge hyps at same (trie_node, lm_context) via logaddexp.
                merged_list = grouped[region]
                key = (hyp.region_state, hyp.lm_state.context_key)
                found = False
                for i, existing in enumerate(merged_list):
                    if (existing.region_state, existing.lm_state.context_key) == key:
                        a, b = existing.weight, hyp.weight
                        if a < b: a, b = b, a
                        new_w = a + math.log1p(math.exp(b - a)) if b > float('-inf') else a
                        merged_list[i] = Hyp(existing.lm_state, new_w, existing.region_state)
                        found = True
                        break
                if not found:
                    merged_list.append(hyp)
            else:
                grouped[region].append(hyp)

        return _CompiledBundle(self.compiled, grouped, self._target + (y,))


# ---------------------------------------------------------------------------
# CompiledBeamState
# ---------------------------------------------------------------------------

class CompiledBeamState(LMState):
    """LM state for CompiledBeam."""

    def __init__(self, cb: CompiledBeam, target: Str[Token],
                 logprefix: float, bundle: _CompiledBundle) -> None:
        self.cb = cb
        self.eos = cb.eos
        self._target = target
        self.logprefix = logprefix
        self._bundle = bundle

    @property
    def context_key(self):
        return self._target

    @property
    def logp_next(self) -> LogDistr[Token]:
        return self._bundle.logp_next

    def __rshift__(self, y: Token) -> CompiledBeamState:
        logp_delta = self.logp_next[y]
        new_bundle = self._bundle.advance(y)
        return CompiledBeamState(
            self.cb, self._target + (y,),
            self.logprefix + logp_delta, new_bundle,
        )

    def __repr__(self) -> str:
        total = sum(len(hs) for hs in self._bundle.hyps_by_region.values())
        regions = len(self._bundle.hyps_by_region)
        return (f'CompiledBeamState(target={self._target!r},'
                f' regions={regions}, hyps={total})')


# ---------------------------------------------------------------------------
# CompiledBeam — main entry point
# ---------------------------------------------------------------------------

class CompiledBeam(LM):
    """Compiled beam search over transduced language models.

    Statically analyzes the FST to identify regions, then dispatches
    hypotheses to specialized scoring code per region type.

    Args:
        inner_lm: Inner language model.
        fst: Finite-state transducer mapping source -> target.
        K: Beam width (max hypotheses across all regions).
        max_beam: Maximum particles for wild-region search.
        max_steps: Search budget for wild-region particle expansion.
        eos: Outer EOS token.
        helper: DFA helper backend for wild region ('python' or 'rust').
        top_k: If set, only expand top-k source symbols per particle.
    """

    def __init__(self, inner_lm: LM, fst: FST[Any, Any],
                 K: int = 10,
                 max_beam: int = 100,
                 max_steps: int = 1000,
                 eos: Token = None,     # type: ignore[assignment]
                 helper: str = "python",
                 top_k: int | None = None) -> None:
        self.inner_lm = inner_lm
        self.fst = fst
        self.K = K
        self.max_beam = max_beam
        self.max_steps = max_steps
        self.eos = eos
        self._top_k = top_k

        # Static analysis
        analyzer = RegionAnalyzer(fst, inner_lm)
        self._region_map = analyzer.analyze()
        self._hub_regions = self._region_map.hub_regions
        self._wild_region = self._region_map.wild_region

        # Wild-region DFA helper
        self._fused_helper = None
        self._fused_adapter = None
        self._sym_map: dict[Any, int] = {}
        self._inv_sym_map: dict[int, Any] = {}

        if self._wild_region is not None:
            self._setup_helper(fst, helper)
            self._fused_adapter = _FusedSearchAdapter(self)

        # Initial hypotheses
        self._initial_hyps_by_region: dict[Region, list[Hyp]] = defaultdict(list)
        self._precompute_initial()

    def _setup_helper(self, fst: FST, helper: str) -> None:
        if helper == "rust":
            import transduction_core
            from transduction.rust_bridge import to_rust_fst
            rust_fst, sym_map, _ = to_rust_fst(fst)
            self._fused_helper = transduction_core.RustLazyPeekabooDFA(rust_fst, True)
            self._sym_map = {k: v for k, v in sym_map.items()}
            self._inv_sym_map = {v: k for k, v in sym_map.items()}
        elif helper == "python":
            from transduction.python_lazy_peekaboo_dfa import PythonLazyPeekabooDFAHelper
            helper_obj = PythonLazyPeekabooDFAHelper(fst)
            self._fused_helper = helper_obj
            self._sym_map = {k: v for k, v in helper_obj._sym_map.items()}
            self._inv_sym_map = {v: k for k, v in helper_obj._sym_map.items()}
        else:
            raise ValueError(f"helper must be 'python' or 'rust', got {helper!r}")

    def _precompute_initial(self) -> None:
        initial_lm = self.inner_lm.initial()

        for q0 in self.fst.start:
            if q0 in self._hub_regions:
                region = self._hub_regions[q0]
                self._initial_hyps_by_region[region].append(
                    Hyp(initial_lm, 0.0, region.trie.root)
                )
            elif q0 in self._region_map.corridor_regions:
                region = self._region_map.corridor_regions[q0]
                self._initial_hyps_by_region[region].append(
                    Hyp(initial_lm, 0.0, q0)
                )
            elif q0 in self._region_map.plateau_regions:
                region = self._region_map.plateau_regions[q0]
                self._initial_hyps_by_region[region].append(
                    Hyp(initial_lm, 0.0, q0)
                )
            elif self._wild_region is not None:
                self._fused_helper.new_step([])
                for sid in self._fused_helper.start_ids():
                    self._initial_hyps_by_region[self._wild_region].append(
                        Hyp(initial_lm, 0.0, Particle(sid, initial_lm, 0.0, ()))
                    )

    def run(self, source_path: Str[Token]) -> int | None:
        path_u32 = [self._sym_map[x] for x in source_path]
        return self._fused_helper.run(path_u32)

    def initial(self) -> CompiledBeamState:
        return CompiledBeamState(
            self, (), 0.0, _CompiledBundle.initial(self),
        )

    def __repr__(self) -> str:
        return f'CompiledBeam(inner={self.inner_lm!r}, {self._region_map.summary()})'


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Color scheme for region types
_REGION_COLORS = {
    'hub': '#4CAF50',       # green
    'corridor': '#2196F3',  # blue
    'plateau': '#FF9800',   # orange
    'wild': '#F44336',      # red
    'unclassified': '#9E9E9E',  # gray
}


def render_region_map_html(rmap: RegionMap, fst: FST | None = None) -> str:
    """Render a RegionMap as an HTML summary table.

    If fst is provided, includes state-level classification with color coding.
    """
    lines = ['<div style="font-family: monospace; font-size: 13px;">']
    lines.append('<h3 style="margin: 4px 0;">Region Map</h3>')

    # Summary bar
    counts = []
    if rmap.hub_regions:
        c = _REGION_COLORS['hub']
        counts.append(f'<span style="color:{c}; font-weight:bold;">'
                      f'{len(rmap.hub_regions)} hub(s)</span>')
    if rmap.corridor_regions:
        c = _REGION_COLORS['corridor']
        counts.append(f'<span style="color:{c}; font-weight:bold;">'
                      f'{len(rmap.corridor_regions)} corridor(s)</span>')
    if rmap.plateau_regions:
        c = _REGION_COLORS['plateau']
        counts.append(f'<span style="color:{c}; font-weight:bold;">'
                      f'{len(rmap.plateau_regions)} plateau(s)</span>')
    if rmap.wild_region:
        c = _REGION_COLORS['wild']
        counts.append(f'<span style="color:{c}; font-weight:bold;">wild</span>')
    lines.append('<p>' + ' | '.join(counts) + '</p>')

    # Detail table
    lines.append('<table style="border-collapse:collapse; margin:4px 0;">')
    lines.append('<tr style="border-bottom:2px solid #333;">'
                 '<th style="padding:2px 8px;">State</th>'
                 '<th style="padding:2px 8px;">Region</th>'
                 '<th style="padding:2px 8px;">Details</th></tr>')

    classification = rmap.state_classification()

    # Hubs
    for state, r in sorted(rmap.hub_regions.items(), key=lambda x: str(x[0])):
        c = _REGION_COLORS['hub']
        n = len(r.trie._source_syms)
        n_nodes = len(r.trie.children)
        lines.append(
            f'<tr><td style="padding:2px 8px;">{state}</td>'
            f'<td style="padding:2px 8px; color:{c}; font-weight:bold;">Hub</td>'
            f'<td style="padding:2px 8px;">{n} tokens, {n_nodes} trie nodes</td></tr>'
        )

    # Corridors
    for state, r in sorted(rmap.corridor_regions.items(), key=lambda x: str(x[0])):
        c = _REGION_COLORS['corridor']
        n_arcs = sum(len(t) for t in r.table.values())
        lines.append(
            f'<tr><td style="padding:2px 8px;">{state}</td>'
            f'<td style="padding:2px 8px; color:{c}; font-weight:bold;">Corridor</td>'
            f'<td style="padding:2px 8px;">{n_arcs} deterministic arcs</td></tr>'
        )

    # Plateaus
    for state, r in sorted(rmap.plateau_regions.items(), key=lambda x: str(x[0])):
        c = _REGION_COLORS['plateau']
        lines.append(
            f'<tr><td style="padding:2px 8px;">{state}</td>'
            f'<td style="padding:2px 8px; color:{c}; font-weight:bold;">Plateau</td>'
            f'<td style="padding:2px 8px;">IP-universal</td></tr>'
        )

    lines.append('</table>')
    lines.append('</div>')
    return '\n'.join(lines)


def render_region_graphviz(fst: FST, rmap: RegionMap, coalesce_chains: bool = False):
    """Render FST with region-colored states using visualize_automaton.

    Parallel edges are merged with compact labels.  Chain coalescing is
    off by default so that region coloring per-state is visible.

    Returns an InteractiveGraph (renders as SVG in Jupyter).
    """
    from transduction.viz import visualize_automaton
    classification = rmap.state_classification()

    def sty_node(q):
        rtype = classification.get(q, 'wild' if rmap.wild_region else 'unclassified')
        color = _REGION_COLORS.get(rtype, '#9E9E9E')
        return {'style': 'filled,rounded', 'fillcolor': color + '80',
                'tooltip': rtype}

    return visualize_automaton(fst, sty_node=sty_node,
                               coalesce_chains=coalesce_chains)
