"""
Incremental transduced language model with particle-based approximate inference.

Given an inner LM P_inner(source) and an FST mapping source -> target,
TransducedLM estimates P(y | target_so_far) by maintaining a beam of K
particles (source-side hypotheses) that evolve as each target symbol is added.
Each particle tracks a DFA state (powerset of precover NFA states) and an
inner LM state; at quotient states, exact marginalization sums over infinitely
many source continuations, providing dramatic variance reduction.

Per-step computation uses best-first search through the decomposition's DFA.
The search pops the highest-weight particle and classifies its DFA state:
  - Quotient for symbol y: full weight contributes to y's score (exact
    marginalization over all continuations).  Particle is absorbed.
  - Remainder for symbol y: weight * P_inner(EOS) contributes to y's score.
  - Preimage: weight * P_inner(EOS) contributes to outer EOS score.
  - Otherwise: expand by source symbols weighted by P_inner.
Carry-forward passes particles at Q/R/resume-frontier states to the next
target step; top-K pruning controls approximation quality.

Usage:
    from transduction import examples
    from transduction.lm.ngram import CharNgramLM
    from transduction.lm.transduced import TransducedLM

    inner = CharNgramLM.train("hello world", n=2)
    fst = examples.lowercase()
    tlm = TransducedLM(inner, fst)

    state = tlm >> 'h'
    print(state.logp_next['e'])
    print(state.logp)
"""

from __future__ import annotations

import heapq
import numpy as np
from collections import defaultdict
from typing import Any

from transduction.lm.base import LM, LMState, Token
from transduction.fst import FST
from transduction.util import LogDistr, LogVector, State, Str
from transduction.viz import _format_nfa_set, render_particles_html

# Default incremental decomposition (Rust backend).
from transduction.rust_bridge import RustPeekabooState as _DefaultDecompState
from transduction.rust_bridge import _RustPeekabooUniv as _DefaultUniv


# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------

class Particle:
    """A weighted source-prefix hypothesis.

    Tracks a DFA state (powerset of precover NFA states) that captures the
    transducer's progress, the inner LM state, and a log importance weight.
    """
    __slots__ = ('dfa_state', 'lm_state', 'log_weight', 'source_path')

    def __init__(self, dfa_state: State, lm_state: LMState, log_weight: float,
                 source_path: Str[Token]) -> None:
        self.dfa_state = dfa_state
        self.lm_state = lm_state
        self.log_weight = log_weight
        self.source_path = source_path

    def __lt__(self, other: Particle) -> bool:
        # max-heap: higher weight = higher priority
        return self.log_weight > other.log_weight

    def __repr__(self) -> str:
        return f'Particle(w={self.log_weight:.3f}, {self.source_path})'


class _RhoExpander:
    """Lazy enumerator for rho-class children at a complete DFA state.

    Instead of eagerly pushing O(|V|) children into the heap, this holds a
    sorted list of (weight, symbol) pairs and yields one child at a time when
    popped.  Only the top item's weight is visible to the heap, so the
    expander is ordered correctly among regular Particles.
    """
    __slots__ = ('rho_dest', 'lm_state', 'source_path',
                 '_sorted', '_index')

    def __init__(self, rho_dest: State, lm_state: LMState,
                 source_path: Str[Token],
                 sorted_children: list[tuple[float, Token]]) -> None:
        self.rho_dest = rho_dest
        self.lm_state = lm_state
        self.source_path = source_path
        self._sorted = sorted_children   # [(weight, symbol), ...] desc
        self._index = 0

    @property
    def log_weight(self) -> float:
        return self._sorted[self._index][0]

    def __lt__(self, other: _RhoExpander | Particle) -> bool:
        # max-heap: higher weight = higher priority (matches Particle)
        return self.log_weight > other.log_weight

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._sorted)

    def pop_next(self) -> Particle:
        """Materialize the next highest-weight child Particle."""
        w, x = self._sorted[self._index]
        self._index += 1
        return Particle(self.rho_dest, self.lm_state >> x, w,
                        self.source_path + (x,))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _select_top_k(particles: list[Particle], k: int) -> list[Particle]:
    """Select the k highest-weight particles in O(n) expected time."""
    n = len(particles)
    if n <= k:
        return list(particles)
    weights = np.array([p.log_weight for p in particles])
    # argpartition: after partitioning, the k largest are at indices[n-k:]
    indices = np.argpartition(weights, n - k)[n - k:]
    return [particles[i] for i in indices]



# ---------------------------------------------------------------------------
# TransducedState
# ---------------------------------------------------------------------------

class TransducedState(LMState[Token]):
    """Immutable state for the TransducedLM.

    Supports:
        state >> y         -> new state (advance by target symbol y)
        state.logp_next[y] -> log P(y | target_so_far)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token

    Each state holds K particles (source-prefix hypotheses).  Computing
    logp_next runs best-first search through the decomposition's DFA:
    particles are popped by weight, classified (Q/R/preimage), scored,
    and expanded if not absorbed by a quotient state.  Carry-forward
    passes particles at Q/R/resume-frontier states to the next target step.
    """

    def __init__(self, tlm: TransducedLM, peekaboo_state: State,
                 particles: list[Particle], logp: float,
                 path: Str[Token] = ()) -> None:
        self.tlm = tlm
        self.eos = tlm.eos
        self._peekaboo_state = peekaboo_state
        self._particles = particles
        self.logp = logp
        self.path = path
        self._logp_next_cache: LogDistr[Token] | None = None
        self._carry_forward: dict[Token, list[Particle]] | None = None

    def _ensure_computed(self) -> None:
        if self._logp_next_cache is None:
            self._compute_logp_next()

    @property
    def logp_next(self) -> LogDistr[Token]:
        """Log-probability distribution over next target symbols."""
        self._ensure_computed()
        return self._logp_next_cache  # type: ignore[return-value]

    def __rshift__(self, y: Token) -> TransducedState:
        """Advance by target symbol ``y``, returning a new TransducedState."""
        if y not in self.tlm.fst.B:
            raise ValueError(f"Out of vocabulary: {y!r}")
        self._ensure_computed()

        # Advance peekaboo decomposition state
        new_peekaboo = self._peekaboo_state >> y

        # Translate carry-forward particles into the new DFA.
        # Resume-frontier states can be kept directly when the Rust arena
        # hasn't been reset (same generation); otherwise all particles must
        # be replayed through run() because DFA state IDs are stale.
        carry = self._carry_forward.get(y, [])
        new_dfa = new_peekaboo.dfa
        parent_gen = getattr(self._peekaboo_state, 'generation', None)
        child_gen = getattr(new_peekaboo, 'generation', None)
        ids_valid = (parent_gen is not None and parent_gen == child_gen)

        new_particles = []
        if ids_valid:
            resume = self._peekaboo_state.resume_frontiers.get(y, set())
            for p in carry:
                if p.dfa_state in resume:
                    new_particles.append(p)
                else:
                    s = new_dfa.run(p.source_path)
                    assert s is not None
                    new_particles.append(Particle(s, p.lm_state, p.log_weight, p.source_path))
        else:
            for p in carry:
                s = new_dfa.run(p.source_path)
                assert s is not None
                new_particles.append(Particle(s, p.lm_state, p.log_weight, p.source_path))

        return TransducedState(
            self.tlm,
            peekaboo_state = new_peekaboo,
            particles = _select_top_k(new_particles, self.tlm.K),
            logp = self.logp + self._logp_next_cache[y],
            path = self.path + (y,),
        )

    def _compute_logp_next(self) -> None:
        """Best-first search through the DFA to score next target symbols.

        Pops the highest-weight particle, classifies its DFA state, scores
        Q/R/preimage contributions, and expands non-quotient particles by
        source symbols.  After the expansion budget is exhausted, remaining
        queued items are scored without expansion.
        """
        decomp = self._peekaboo_state.decomp
        dfa = self._peekaboo_state.dfa

        # --- Per-state classification lookups ---
        q_of = {}       # dfa_state -> set of y where state in Q(y)
        r_of = {}       # dfa_state -> set of y where state in R(y)
        resume_of = {}  # dfa_state -> set of y where state in resume(y)

        for y, d in decomp.items():
            for state in d.quotient:
                q_of.setdefault(state, set()).add(y)
            for state in d.remainder:
                r_of.setdefault(state, set()).add(y)
        for y, states in self._peekaboo_state.resume_frontiers.items():
            for state in states:
                resume_of.setdefault(state, set()).add(y)

        # --- Accumulators ---
        scores: LogVector[Token] = LogVector()
        carry_forward: defaultdict[Token, list[Particle]] = defaultdict(list)
        cf_paths: defaultdict[Token, set[Str[Token]]] = defaultdict(set)

        def _score(particle: Particle) -> tuple[set[Token], set[Token]]:
            """Classify particle and accumulate scores.

            Returns (q_syms, r_syms) for the particle's DFA state.
            """
            d = particle.dfa_state
            w = particle.log_weight

            q_syms = q_of.get(d, set())
            r_syms = r_of.get(d, set())

            # Quotient: prefix probability p_X→(source) = P(source).
            for y in q_syms:
                scores.logaddexp(y, w)

            eos_lp = particle.lm_state.logp_next[self.tlm.inner_lm.eos]

            if eos_lp > -np.inf:
                eos_w = w + eos_lp

                # preimage (EOS): skip when particle is Q-absorbed.
                # A Q-classified particle's full prefix probability already
                # accounts for ALL continuations (including EOS) going to
                # the Q symbol.  Adding the preimage P_X(source) to EOS
                # would double-count: once in the Q's prefix probability
                # and once in the preimage.
                if d in self._peekaboo_state.preimage_stops and not q_syms:
                    scores.logaddexp(self.eos, eos_w)

                # Remainder
                for y in r_syms - q_syms:
                    scores.logaddexp(y, eos_w)

            return q_syms, r_syms

        def _add_carry_checked(y: Token, particle: Particle) -> None:
            """Add particle to carry_forward[y], skipping duplicates and prefix-dominated paths.

            If the exact source_path or a shorter prefix is already in cf_paths[y],
            this particle is redundant — either it's a duplicate or a shallower
            particle will re-discover it.  Skipping eagerly avoids accumulating
            O(budget) entries in carry_forward for cyclic FSTs.
            """
            path = particle.source_path
            if path in cf_paths[y] or any(path[:k] in cf_paths[y] for k in range(len(path))):
                return
            carry_forward[y].append(particle)
            cf_paths[y].add(path)

        def _carry_forward(particle: Particle, q_syms: set[Token],
                           r_syms: set[Token]) -> None:
            """Add particle to carry-forward sets for relevant symbols.

            Q-absorbed particles are not expanded, so they cannot create
            prefix-overlapping descendants — added directly without the
            prefix check.  Non-Q carry-forward (R-only or resume-frontier
            for a non-Q symbol) uses the prefix-domination check because
            the particle will be expanded and its descendants may also
            be carried forward for the same symbol.
            """
            d = particle.dfa_state
            for y in q_syms:
                carry_forward[y].append(particle)
                cf_paths[y].add(particle.source_path)
            for y in (r_syms | resume_of.get(d, set())) - q_syms:
                _add_carry_checked(y, particle)

        # --- Best-first search ---
        queue = list(self._particles)
        heapq.heapify(queue)

        expansions = 0
        while queue and expansions < self.tlm.max_expansions:
            expansions += 1

            item = heapq.heappop(queue)
            if isinstance(item, _RhoExpander):
                particle = item.pop_next()
                if not item.exhausted:
                    heapq.heappush(queue, item)
            else:
                particle = item

            q_syms, r_syms = _score(particle)
            _carry_forward(particle, q_syms, r_syms)

            if q_syms:
                # Q-absorbed: not expanded (exact marginalization).
                continue

            # Non-Q: expand by source symbols.
            # Use rho-aware expansion when available to reduce per-particle
            # cost at complete DFA states (O(D) explicit + O(|V|) rho class
            # with a single shared classify() call for the rho destination).
            lm_logp = particle.lm_state.logp_next
            if hasattr(dfa, 'rho_arcs'):
                has_rho, rho_dest, explicit_arcs, rho_symbols = dfa.rho_arcs(particle.dfa_state)
                for x, next_dfa_state in explicit_arcs:
                    child_w = particle.log_weight + lm_logp[x]
                    if child_w > -np.inf:
                        heapq.heappush(queue, Particle(
                            next_dfa_state,
                            particle.lm_state >> x,
                            child_w,
                            particle.source_path + (x,),
                        ))
                if has_rho and rho_dest is not None:
                    rho_children = []
                    for x in rho_symbols:
                        child_w = particle.log_weight + lm_logp[x]
                        if child_w > -np.inf:
                            rho_children.append((child_w, x))
                    if rho_children:
                        rho_children.sort(reverse=True)
                        heapq.heappush(queue, _RhoExpander(
                            rho_dest, particle.lm_state,
                            particle.source_path, rho_children))
            else:
                for x, next_dfa_state in dfa.arcs(particle.dfa_state):
                    child_w = particle.log_weight + lm_logp[x]
                    if child_w > -np.inf:
                        heapq.heappush(queue, Particle(
                            next_dfa_state,
                            particle.lm_state >> x,
                            child_w,
                            particle.source_path + (x,),
                        ))

        # Budget exhausted — score and carry forward remaining items
        # without expanding.  Drain _RhoExpanders fully so all rho
        # children contribute to the logp_next distribution.
        while queue:
            item = heapq.heappop(queue)
            if isinstance(item, _RhoExpander):
                particle = item.pop_next()
                if not item.exhausted:
                    heapq.heappush(queue, item)
            else:
                particle = item
            q_syms, r_syms = _score(particle)
            _carry_forward(particle, q_syms, r_syms)

        self._logp_next_cache = scores.normalize()
        self._carry_forward = carry_forward

    def _repr_html_(self) -> str:
        decomp = self._peekaboo_state.decomp
        q_states, r_states = set(), set()
        for d in decomp.values():
            q_states.update(d.quotient)
            r_states.update(d.remainder)

        ps = self._peekaboo_state
        can_decode = hasattr(ps, 'decode_dfa_state')
        decode_cache: dict[State, str] = {}
        def decode_fn(dfa_state: State) -> str:
            if not can_decode:
                return str(dfa_state)
            if dfa_state not in decode_cache:
                try:
                    decoded = ps.decode_dfa_state(dfa_state)
                    decode_cache[dfa_state] = _format_nfa_set(decoded)
                except Exception:
                    decode_cache[dfa_state] = str(dfa_state)
            return decode_cache[dfa_state]

        qr_builder = ps.build_qr_fsa if can_decode and hasattr(ps, 'build_qr_fsa') else None

        result = render_particles_html(
            'TransducedState', self._particles, self.path, self.logp,
            decode_fn=decode_fn,
            q_states=q_states, r_states=r_states,
            qr_builder=qr_builder, decomp=decomp,
        )

        from transduction.viz import render_logp_next_html
        result += render_logp_next_html(
            'TransducedState', self.path, self.logp, self.logp_next,
        )
        return result

    def __repr__(self) -> str:
        return (f'TransducedState(target={self._peekaboo_state.target!r},'
                f' K={len(self._particles)})')


# ---------------------------------------------------------------------------
# TransducedLM
# ---------------------------------------------------------------------------

class TransducedLM(LM[Token]):
    """Incremental transduced language model with particle-based inference.

    Computes the pushforward of an inner LM through an FST using a beam of
    K particles.  Each particle tracks a DFA state (powerset of precover NFA
    states) and an inner LM state.  At quotient states, exact marginalization
    sums over infinitely many source continuations.

    Per-step computation uses best-first search through the decomposition's
    DFA with an expansion budget.  Each popped particle is classified
    (Q/R/preimage) and scored; non-quotient particles are expanded by source
    symbols.  Quotient particles are absorbed (exact marginalization) without
    counting against the budget.

    Args:
        inner_lm: LM with StateLM interface (has .initial(), state >> x,
            state.logp_next).
        fst: FST instance mapping source -> target.
        K: Carry-forward budget -- maximum number of particles retained across
            target steps.  Larger K improves accuracy.
        max_expansions: Best-first search budget -- maximum number of non-
            quotient particles expanded per target step.  Larger values
            explore more of the source space per step.
        eos: Outer EOS token.
        decomp_state_cls: Incremental decomposition class (default: RustPeekabooState).
        univ_cls: Universality precomputation class (default: _RustPeekabooUniv).

    Note:
        For beam-sum consistency, both ``K`` and ``max_expansions`` must grow
        to infinity.  The defaults (K=100, max_expansions=1000) work for most
        FSTs, but high-branching-factor FSTs may need a larger ratio of
        ``max_expansions`` to ``K``.
    """

    def __init__(self, inner_lm: LM, fst: FST[Any, Any], K: int,
                 max_expansions: int = 1000, eos: Token = '<EOS>',  # type: ignore[assignment]
                 decomp_state_cls: type | None = None,
                 univ_cls: type = _DefaultUniv) -> None:
        self.inner_lm = inner_lm
        self.fst = fst
        self.max_expansions = max_expansions
        self.K = K
        self.eos = eos
        self._decomp_state_cls = decomp_state_cls or _DefaultDecompState
        self._univ = univ_cls(fst)

    def initial(self) -> TransducedState:
        """Return the initial TransducedState (empty target prefix)."""
        peekaboo = self._decomp_state_cls(self.fst, '', parent=None, univ=self._univ)
        inner_initial = self.inner_lm.initial()
        return TransducedState(
            self,
            peekaboo_state = peekaboo,
            logp = 0.0,
            particles = [
                Particle(s, inner_initial, 0.0, ())
                for s in peekaboo.dfa.start()
            ],
        )

    def __repr__(self) -> str:
        return f'TransducedLM(inner={self.inner_lm!r})'
