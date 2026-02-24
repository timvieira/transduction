"""Generalized beam search over transduced language models.

Unifies CharacterBeam (trie-mass scoring at IP-universal accepting hubs)
with FusedTransducedLM (particle expansion via DFA classification).  At hub
states, hypotheses use the fast trie-mass path; elsewhere, they fall back to
the general particle-expansion path with Q/R/preimage classification.

The algorithm maintains two hypothesis types:

- **HubHyp**: at an IP-universal accepting hub, with a per-hub OutputTrie
  for O(1) next-symbol scoring via trie mass.
- **Particle**: at a non-hub DFA state, scored via best-first search with
  quotient/remainder/preimage classification (same as FusedTransducedLM).

Usage::

    from transduction.lm.generalized_beam import GeneralizedBeam
    from transduction.lm.ngram import CharNgramLM
    from transduction import examples

    inner = CharNgramLM.train("hello world", n=2)
    fst = examples.lowercase()
    gb = GeneralizedBeam(inner, fst, K=10)
    state = gb.initial()
    print(state.logp_next)
"""

from __future__ import annotations

import heapq
import numpy as np
from collections import defaultdict, deque
from functools import cached_property
from typing import Any

from transduction.fst import FST, EPSILON
from transduction.lm.base import LM, LMState, Token
from transduction.lm.transduced import Particle, _select_top_k
from transduction.lm.fused_transduced import _FusedSearch, FusedTransducedLM
from transduction.universality import compute_ip_universal_states
from transduction.util import LogDistr, LogVector, State, Str, logsumexp


# ---------------------------------------------------------------------------
# OutputTrie — per-hub trie mapping output sequences to (source_sym, dest)
# ---------------------------------------------------------------------------

class OutputTrie:
    """Trie over output-symbol sequences reachable from a hub state.

    Generalizes TokenCharacterTrie from character_beam.py.  Each leaf stores
    ``(source_symbol, dest_state)`` — the source token consumed and the FST
    state reached.  Precomputes a COO reachability index for vectorized
    ``log_mass_sum`` via ``np.logaddexp.at``.
    """

    def __init__(self, entries: list[tuple[Any, tuple, Any]],
                 inner_eos: Any) -> None:
        """Build trie from entries.

        Args:
            entries: list of (source_symbol, output_tuple, dest_state).
            inner_eos: the inner LM's EOS token.
        """
        self._source_syms: list[Any] = []
        src_to_idx: dict[Any, int] = {}

        self.root = 0
        children: list[dict[Any, int]] = [{}]
        leaf_info: dict[int, tuple[Any, Any]] = {}   # leaf_node -> (src_sym, dest)
        coo_nodes: list[int] = []
        coo_src_ids: list[int] = []

        for src_sym, output_seq, dest_state in entries:
            if src_sym not in src_to_idx:
                src_to_idx[src_sym] = len(self._source_syms)
                self._source_syms.append(src_sym)
            idx = src_to_idx[src_sym]

            # Walk root -> leaf, creating trie nodes along the path.
            curr = 0
            path = [0]
            for out_sym in output_seq:
                if out_sym not in children[curr]:
                    nid = len(children)
                    children[curr][out_sym] = nid
                    children.append({})
                curr = children[curr][out_sym]
                path.append(curr)

            # End-of-token sentinel leaf.
            sentinel = len(children)
            children[curr][None] = sentinel
            children.append({})
            leaf_info[sentinel] = (src_sym, dest_state)

            # COO: every node on the path (incl. sentinel) maps to this source sym.
            for node in path:
                coo_nodes.append(node)
                coo_src_ids.append(idx)
            coo_nodes.append(sentinel)
            coo_src_ids.append(idx)

        self.children = children
        self.leaf_info = leaf_info
        self._coo_nodes = np.array(coo_nodes, dtype=np.intp)
        self._coo_src_ids = np.array(coo_src_ids, dtype=np.intp)
        self._inner_eos = inner_eos

    def log_mass_sum(self, logp_next: LogDistr) -> np.ndarray:
        """Bottom-up log-probability mass at each trie node via logaddexp scatter."""
        logp = np.array([logp_next[tok] for tok in self._source_syms])
        log_mass = np.full(len(self.children), -np.inf, dtype=np.float64)
        np.logaddexp.at(log_mass, self._coo_nodes, logp[self._coo_src_ids])
        return log_mass

    @property
    def is_empty(self) -> bool:
        return len(self._source_syms) == 0


# ---------------------------------------------------------------------------
# HubHyp — hypothesis at an IP-universal accepting hub
# ---------------------------------------------------------------------------

class HubHyp:
    """A single hypothesis at a hub state: LM state + trie position + mass.

    Scoring uses trie mass: ``log_mass[child] - log_mass[node]``.
    """
    __slots__ = ('lm_state', 'hub', 'trie', 'node', 'log_mass', 'weight')

    def __init__(self, lm_state: LMState, hub: Any, trie: OutputTrie,
                 node: int, log_mass: np.ndarray, weight: float) -> None:
        self.lm_state = lm_state
        self.hub = hub
        self.trie = trie
        self.node = node
        self.log_mass = log_mass
        self.weight = weight

    def __repr__(self) -> str:
        return f'HubHyp(w={self.weight:.2f}, hub={self.hub})'

    def __lt__(self, other: HubHyp) -> bool:
        return self.weight < other.weight

    @property
    def actions(self) -> dict[Any, int]:
        return self.trie.children[self.node]

    def has_EOT(self) -> bool:
        return None in self.trie.children[self.node]


# ---------------------------------------------------------------------------
# Hub vocab computation
# ---------------------------------------------------------------------------

def _compute_hub_vocab(fst: FST, hub: Any) -> list[tuple[Any, tuple, Any]] | None:
    """BFS from hub following epsilon-input arcs, collecting source arcs.

    Returns list of (source_symbol, output_tuple, dest_state) or None
    if the hub's vocab is non-deterministic (a source symbol has multiple
    output sequences).
    """
    # BFS: track (fst_state, accumulated_output_tuple)
    visited: set[Any] = set()
    queue: deque[tuple[Any, tuple]] = deque()
    queue.append((hub, ()))
    visited.add(hub)

    entries: list[tuple[Any, tuple, Any]] = []
    seen: dict[Any, tuple[tuple, Any]] = {}   # source_sym -> (output, dest)

    while queue:
        state, output_so_far = queue.popleft()

        for a, b, j in fst.arcs(state):
            out_ext = output_so_far + ((b,) if b != EPSILON else ())

            if a == EPSILON:
                # Follow epsilon-input arcs (output accumulates)
                if j not in visited:
                    visited.add(j)
                    queue.append((j, out_ext))
            else:
                # Non-epsilon source symbol: this completes a "token"
                if a in seen:
                    prev_out, prev_dest = seen[a]
                    if prev_out != out_ext or prev_dest != j:
                        return None  # Non-deterministic
                else:
                    seen[a] = (out_ext, j)
                    entries.append((a, out_ext, j))

    return entries


# ---------------------------------------------------------------------------
# _HybridBundle — collection of HubHyps + Particles
# ---------------------------------------------------------------------------

class _HybridBundle:
    """Holds hub hypotheses and particles, implements scoring and advance."""

    def __init__(self, alg: GeneralizedBeam,
                 hub_hyps: list[HubHyp],
                 particles: list[Particle],
                 target: Str[Token]) -> None:
        self.alg = alg
        self.hub_hyps = hub_hyps
        self.particles = particles
        self._target = target

    @classmethod
    def initial(cls, alg: GeneralizedBeam) -> _HybridBundle:
        return cls(alg, list(alg._initial_hub_hyps),
                   list(alg._initial_particles), ())

    def _extend_hub_hyps(self) -> tuple[list[HubHyp], list[Particle]]:
        """Extend hub hyps at end-of-token.

        Uses a 3-phase prepare/prefetch/complete pattern to enable batched
        forward passes when the inner LM supports prefetch (e.g. HuggingFaceLM).

        Returns (new_hub_hyps_from_extension, new_particles_from_hub_exit).
        """
        new_hub_hyps: list[HubHyp] = []
        new_particles: list[Particle] = []

        # Phase 1: Prepare — create child LM states (cheap: >> is lazy).
        # Collect items that go to hubs (need logp_next) vs non-hubs (particles).
        hub_pending: list[tuple[LMState, Any, OutputTrie, float]] = []
        needs_logp: list[LMState] = []

        for h in self.hub_hyps:
            if not h.has_EOT():
                continue

            for eot_child_node, (src_sym, dest_state) in self._eot_leaves(h):
                w_prime = h.weight + h.log_mass[eot_child_node] - h.log_mass[h.node]
                if w_prime <= -np.inf:
                    continue

                new_lm_state = h.lm_state >> src_sym

                if dest_state in self.alg._hub_tries:
                    dest_trie = self.alg._hub_tries[dest_state]
                    hub_pending.append((new_lm_state, dest_state, dest_trie, w_prime))
                    needs_logp.append(new_lm_state)
                else:
                    new_particles.append(Particle(
                        None, new_lm_state, w_prime, (src_sym,),
                    ))

        # Phase 2: Prefetch — batch forward passes for hub-bound states.
        if needs_logp:
            self.alg.inner_lm.prefetch(needs_logp)

        # Phase 3: Complete — consume (now-cached) logp_next via log_mass_sum.
        for new_lm_state, dest_state, dest_trie, w_prime in hub_pending:
            new_mass = dest_trie.log_mass_sum(new_lm_state.logp_next)
            new_hub_hyps.append(HubHyp(
                new_lm_state, dest_state, dest_trie,
                dest_trie.root, new_mass, w_prime,
            ))

        return new_hub_hyps, new_particles

    def _eot_leaves(self, h: HubHyp):
        """Yield (eot_child_node, (src_sym, dest_state)) for EOT transitions."""
        for child_node in self._collect_eot_nodes(h.trie, h.node):
            yield child_node, h.trie.leaf_info[child_node]

    def _collect_eot_nodes(self, trie: OutputTrie, node: int) -> list[int]:
        """Collect all EOT sentinel leaf nodes reachable from the given node."""
        result = []
        children = trie.children[node]
        if None in children:
            result.append(children[None])
        return result

    @cached_property
    def _scored(self) -> tuple[LogDistr[Token], dict[Token, list]]:
        """Compute logp_next and carry_forward."""
        scores: LogVector[Token] = LogVector()
        eos_score = -np.inf
        carry_forward: dict[Token, list] = defaultdict(list)

        # Phase 1: Hub hyp scoring + extension
        extended_hub_hyps, hub_exit_particles = self._extend_hub_hyps()

        all_hub_hyps = self.hub_hyps + extended_hub_hyps

        for h in all_hub_hyps:
            # Score output symbols via trie mass
            node_mass = h.log_mass[h.node]
            for y, child_node in h.trie.children[h.node].items():
                if y is None:
                    continue  # EOT sentinel, not a real output symbol
                child_mass = h.log_mass[child_node]
                scores.logaddexp(y, h.weight + child_mass - node_mass)

            # EOS contribution: hub hyps at root contribute P_inner(EOS)
            if h.node == h.trie.root:
                eos_lp = h.lm_state.logp_next[self.alg.inner_lm.eos]
                if eos_lp > -np.inf:
                    eos_score = np.logaddexp(eos_score, h.weight + eos_lp)

        # Phase 2: Particle search
        all_particles = list(self.particles) + hub_exit_particles

        if all_particles and self.alg._fused_helper is not None:
            p_scores, p_eos, p_carry = self._particle_search(all_particles)
            for sym, w in p_scores.items():
                scores.logaddexp(sym, w)
            eos_score = np.logaddexp(eos_score, p_eos)
            for sym, ps in p_carry.items():
                carry_forward[sym].extend(ps)

        # Hub hyp carry-forward: advance trie on each symbol
        for h in all_hub_hyps:
            node_mass = h.log_mass[h.node]
            for y, child_node in h.trie.children[h.node].items():
                if y is None:
                    continue
                carry_forward[y].append(('hub', h, child_node))

        scores[self.alg.eos] = eos_score
        logp_next = scores.normalize()
        return logp_next, carry_forward

    def _particle_search(self, particles: list[Particle]
                         ) -> tuple[LogVector[Token], float, dict[Token, list[Particle]]]:
        """Run FusedTransducedLM-style best-first search over particles."""
        adapter = self.alg._fused_adapter
        search = _FusedSearch(adapter, self._target, particles)
        return search.search()

    @property
    def logp_next(self) -> LogDistr[Token]:
        return self._scored[0]

    @property
    def carry_forward(self) -> dict[Token, list]:
        return self._scored[1]

    def advance(self, y: Token) -> _HybridBundle:
        """Advance by symbol y: prune, then advance all hyps."""
        carry = self.carry_forward.get(y, [])

        new_hub_hyps: list[HubHyp] = []
        new_particles: list[Particle] = []

        for item in carry:
            if isinstance(item, tuple) and item[0] == 'hub':
                _, h, child_node = item
                new_w = h.weight + h.log_mass[child_node] - h.log_mass[h.node]
                new_hub_hyps.append(HubHyp(
                    h.lm_state, h.hub, h.trie,
                    child_node, h.log_mass, new_w,
                ))
            else:
                # Particle carry-forward
                new_particles.append(item)

        # Prune particles
        new_particles = _select_top_k(new_particles, self.alg.max_beam)

        # Prune hub hyps
        if self.alg.K is not None and len(new_hub_hyps) > self.alg.K:
            new_hub_hyps.sort(key=lambda h: -h.weight)
            new_hub_hyps = new_hub_hyps[:self.alg.K]

        new_target = self._target + (y,)
        return _HybridBundle(self.alg, new_hub_hyps, new_particles, new_target)


# ---------------------------------------------------------------------------
# GeneralizedBeamState
# ---------------------------------------------------------------------------

class GeneralizedBeamState(LMState):
    """LM state for GeneralizedBeam: wraps a _HybridBundle."""

    def __init__(self, gb: GeneralizedBeam, target: Str[Token],
                 logp: float, bundle: _HybridBundle) -> None:
        self.gb = gb
        self.eos = gb.eos
        self._target = target
        self.logp = logp
        self._bundle = bundle

    @property
    def logp_next(self) -> LogDistr[Token]:
        return self._bundle.logp_next

    def __rshift__(self, y: Token) -> GeneralizedBeamState:
        logp_delta = self.logp_next[y]
        new_bundle = self._bundle.advance(y)
        return GeneralizedBeamState(
            self.gb, self._target + (y,),
            self.logp + logp_delta, new_bundle,
        )

    def __repr__(self) -> str:
        return (f'GeneralizedBeamState(target={self._target!r},'
                f' hubs={len(self._bundle.hub_hyps)},'
                f' particles={len(self._bundle.particles)})')


# ---------------------------------------------------------------------------
# _FusedSearchAdapter — wraps GeneralizedBeam as FusedTransducedLM for search
# ---------------------------------------------------------------------------

class _FusedSearchAdapter:
    """Adapter making GeneralizedBeam look like FusedTransducedLM for _FusedSearch."""

    def __init__(self, gb: GeneralizedBeam) -> None:
        self._gb = gb
        self.inner_lm = gb.inner_lm
        self.max_steps = gb.max_steps
        self.eos = gb.eos
        self._top_k = gb._top_k
        self._rust_helper = gb._fused_helper
        self._sym_map = gb._sym_map
        self._inv_sym_map = gb._inv_sym_map

    def run(self, source_path: Str[Token]) -> int | None:
        path_u32 = [self._sym_map[x] for x in source_path]
        return self._rust_helper.run(path_u32)


# ---------------------------------------------------------------------------
# GeneralizedBeam (main entry point)
# ---------------------------------------------------------------------------

class GeneralizedBeam(LM):
    """Generalized beam search over transduced language models.

    Combines trie-mass scoring at IP-universal accepting hubs with
    particle-based DFA expansion elsewhere.

    Args:
        inner_lm: Inner language model.
        fst: Finite-state transducer mapping source -> target.
        K: Beam width for hub hypotheses.
        max_beam: Maximum particles carried forward.
        max_steps: Search budget for particle expansion.
        eos: Outer EOS token.
        helper: DFA helper backend ('python' or 'rust').
        top_k: If set, only expand top-k source symbols per particle.
    """

    def __init__(self, inner_lm: LM, fst: FST[Any, Any],
                 K: int = 10,
                 max_beam: int = 100,
                 max_steps: int = 1000,
                 eos: Token = '<EOS>',   # type: ignore[assignment]
                 helper: str = "python",
                 top_k: int | None = None) -> None:
        self.inner_lm = inner_lm
        self.fst = fst
        self.K = K
        self.max_beam = max_beam
        self.max_steps = max_steps
        self.eos = eos
        self._top_k = top_k

        self._hub_tries: dict[Any, OutputTrie] = {}
        self._deterministic_hubs: set[Any] = set()

        # Fast path: BPE-like FSTs with a single start/stop state whose hub
        # vocab covers the entire source alphabet.  Replaces the expensive
        # compute_ip_universal_states fixpoint (O(|Q|^2 * |Sigma|)) with a
        # single O(|arcs|) BFS via _compute_hub_vocab.  If the hub vocab is
        # deterministic and complete, the hub is IP-universal by construction.
        rust_fst = None
        _rust_sym_map = None
        _used_fast_path = False

        if len(fst.start) == 1 and set(fst.start) == set(fst.stop):
            hub = next(iter(fst.start))
            entries = _compute_hub_vocab(fst, hub)
            if entries is not None:
                entry_syms = {src for src, _, _ in entries}
                source_alpha = fst.A - {EPSILON}
                all_return_to_hub = all(dest == hub for _, _, dest in entries)
                if entry_syms == source_alpha and all_return_to_hub:
                    trie = OutputTrie(entries, inner_lm.eos)
                    if not trie.is_empty:
                        self._hub_tries[hub] = trie
                        self._deterministic_hubs.add(hub)
                    _used_fast_path = True

        if not _used_fast_path:
            # Slow path: Rust fixpoint (or Python fallback)
            try:
                import transduction_core
                from transduction.rust_bridge import to_rust_fst
                rust_fst, sym_map, state_map = to_rust_fst(fst)
                inv_sym_map = {v: k for k, v in sym_map.items()}
                inv_state_map = {v: k for k, v in state_map.items()}

                _rust_sym_map = sym_map

                ip_univ_vec = transduction_core.rust_compute_ip_universal_states(rust_fst)
                ip_univ = {inv_state_map[i] for i, v in enumerate(ip_univ_vec) if v}
                hubs = {q for q in ip_univ if fst.is_final(q)}

                for hub in hubs:
                    rust_entries = transduction_core.rust_compute_hub_vocab(
                        rust_fst, state_map(hub)
                    )
                    if rust_entries is not None:
                        entries = [
                            (inv_sym_map[src], tuple(inv_sym_map[o] for o in out_seq), inv_state_map[dest])
                            for src, out_seq, dest in rust_entries
                        ]
                        trie = OutputTrie(entries, inner_lm.eos)
                        if not trie.is_empty:
                            self._hub_tries[hub] = trie
                            self._deterministic_hubs.add(hub)
            except ImportError:
                # Fall back to Python
                rust_fst = None
                ip_univ = compute_ip_universal_states(fst)
                hubs = {q for q in ip_univ if fst.is_final(q)}

                for hub in hubs:
                    entries = _compute_hub_vocab(fst, hub)
                    if entries is not None:
                        trie = OutputTrie(entries, inner_lm.eos)
                        if not trie.is_empty:
                            self._hub_tries[hub] = trie
                            self._deterministic_hubs.add(hub)

        # Set up DFA helper for particle expansion
        self._fused_helper = None
        self._fused_adapter = None
        self._sym_map: dict[Any, int] = {}
        self._inv_sym_map: dict[int, Any] = {}

        need_particles = self._needs_particle_support(fst)
        if need_particles:
            self._setup_helper(fst, helper, rust_fst=rust_fst,
                               sym_map=_rust_sym_map)
            self._fused_adapter = _FusedSearchAdapter(self)

        # Precompute initial hypotheses
        self._initial_hub_hyps: list[HubHyp] = []
        self._initial_particles: list[Particle] = []
        self._precompute_initial()

    def _needs_particle_support(self, fst: FST) -> bool:
        """Check if any non-hub states exist or hub exits go to non-hubs."""
        # If there are start states that aren't hubs, need particles
        for q in fst.start:
            if q not in self._deterministic_hubs:
                return True

        # If any hub's vocab entries go to non-hub destinations, need particles
        for hub, trie in self._hub_tries.items():
            for leaf_node, (src_sym, dest) in trie.leaf_info.items():
                if dest not in self._deterministic_hubs:
                    return True

        return False

    def _setup_helper(self, fst: FST, helper: str,
                      rust_fst=None, sym_map=None) -> None:
        """Initialize the DFA helper for particle expansion."""
        if helper == "rust":
            import transduction_core
            if rust_fst is None or sym_map is None:
                from transduction.rust_bridge import to_rust_fst
                rust_fst, sym_map, _ = to_rust_fst(fst)
            self._fused_helper = transduction_core.RustLazyPeekabooDFA(rust_fst, True)
            self._sym_map = {k: v for k, v in sym_map.items()}
            self._inv_sym_map = {v: k for k, v in sym_map.items()}
        elif helper == "python":
            from transduction.trie_dispatch import PythonLazyPeekabooDFAHelper
            helper_obj = PythonLazyPeekabooDFAHelper(fst)
            self._fused_helper = helper_obj
            self._sym_map = {k: v for k, v in helper_obj._sym_map.items()}
            self._inv_sym_map = {v: k for k, v in helper_obj._sym_map.items()}
        else:
            raise ValueError(f"helper must be 'python' or 'rust', got {helper!r}")

    def _precompute_initial(self) -> None:
        """Create initial hub hyps and particles."""
        initial_lm = self.inner_lm.initial()

        for q0 in self.fst.start:
            if q0 in self._deterministic_hubs:
                trie = self._hub_tries[q0]
                mass = trie.log_mass_sum(initial_lm.logp_next)
                self._initial_hub_hyps.append(HubHyp(
                    initial_lm, q0, trie, trie.root, mass, 0.0,
                ))
            elif self._fused_helper is not None:
                # Create particle at DFA start state
                self._fused_helper.new_step([])
                for sid in self._fused_helper.start_ids():
                    self._initial_particles.append(Particle(
                        sid, initial_lm, 0.0, (),
                    ))

    def initial(self) -> GeneralizedBeamState:
        return GeneralizedBeamState(
            self, (), 0.0, _HybridBundle.initial(self),
        )

    def __repr__(self) -> str:
        n_hubs = len(self._hub_tries)
        return f'GeneralizedBeam(inner={self.inner_lm!r}, hubs={n_hubs})'
