"""Pynini-backed transduced language model with particle-based inference.

Uses pynini/OpenFST for DFA construction instead of the Rust powerset backend.
Classifies particles by tracking their state in per-symbol precover DFAs
incrementally: when a particle is expanded by source symbol x, each z-state
is advanced by one transition-table lookup rather than re-running the full
source path from scratch.

Usage:
    from transduction import examples
    from transduction.lm.ngram import CharNgramLM
    from transduction.lm.pynini_transduced import PyniniTransducedLM

    inner = CharNgramLM.train("hello world", n=2)
    fst = examples.lowercase()
    tlm = PyniniTransducedLM(inner, fst, K=100)

    state = tlm >> 'h'
    print(state.logp_next['e'])
"""

import heapq
import numpy as np
from collections import defaultdict

import pynini

from transduction.fst import EPSILON
from transduction.lm.base import LM, LMState
from transduction.util import LogVector
from transduction.lm.transduced import Particle, _select_top_k
from transduction.pynini_ops import (
    PyniniPrecover, _universal_states,
)


def _build_exact_filter(target, out_label_map, out_sym_table):
    """Build a pynini acceptor for exactly the target sequence (no Sigma* tail).

    Used for preimage: compose(fst, exact_filter).project('input') gives the
    set of source strings whose transduction is exactly `target`.
    """
    n = len(target)
    filt = pynini.Fst()
    for _ in range(n + 1):
        filt.add_state()
    filt.set_start(0)
    filt.set_final(n)
    for i, sym in enumerate(target):
        label = out_label_map[sym]
        filt.add_arc(i, pynini.Arc(label, label, 0, i + 1))
    filt.set_input_symbols(out_sym_table)
    filt.set_output_symbols(out_sym_table)
    return filt


def _build_transition_table(dfa):
    """Pre-build transition table and final-state set from a pynini DFA.

    Returns (trans, finals) where:
      trans: dict[state, dict[label, next_state]]
      finals: frozenset of final state IDs
    """
    zero = pynini.Weight.zero(dfa.weight_type())
    trans = {}
    finals = set()
    for state in dfa.states():
        arcs_map = {}
        for arc in dfa.arcs(state):
            arcs_map[arc.ilabel] = arc.nextstate
        trans[state] = arcs_map
        if dfa.final(state) != zero:
            finals.add(state)
    return trans, frozenset(finals)


# Sentinel: particle's z-state is dead (unreachable) in a given DFA.
_DEAD = None


class PyniniTransducedState(LMState):
    """Immutable state for PyniniTransducedLM.

    Each state holds a beam of particles (source-prefix hypotheses).
    Computing logp_next runs best-first search through the precover DFA,
    classifying particles via incremental state tracking in per-symbol
    pynini precover DFAs.
    """

    def __init__(self, tlm, particles, target, logp, path=()):
        self.tlm = tlm
        self.eos = tlm.eos
        self._particles = particles
        self._target = target
        self.logp = logp
        self.path = path
        self._logp_next_cache = None
        self._carry_forward = None

    def _ensure_computed(self):
        if self._logp_next_cache is None:
            self._compute_logp_next()

    @property
    def logp_next(self):
        self._ensure_computed()
        return self._logp_next_cache

    def __rshift__(self, y):
        self._ensure_computed()
        if y not in self._logp_next_cache:
            raise ValueError(f"Out of vocabulary: {y!r}")

        carry = self._carry_forward.get(y, [])

        # Build new precover DFA for target + (y,) and replay source paths
        new_target = self._target + (y,)
        new_dfa = self.tlm._backend.build_dfa(new_target)

        new_particles = []
        for p in carry:
            reached = new_dfa.run(p.source_path)
            if reached is not None:
                # run() returns a set of states; for DFAs it's a singleton
                if isinstance(reached, set):
                    for st in reached:
                        new_particles.append(Particle(st, p.lm_state, p.log_weight, p.source_path))
                else:
                    new_particles.append(Particle(reached, p.lm_state, p.log_weight, p.source_path))

        new_particles = _select_top_k(new_particles, self.tlm.K)

        return PyniniTransducedState(
            self.tlm, new_particles, new_target,
            self.logp + self._logp_next_cache[y],
            path=self.path + (y,),
        )

    def _compute_logp_next(self):
        """Best-first search with incremental pynini-based Q/R classification.

        For each candidate next-symbol z, builds precover(target+(z,)) via
        pynini, pre-computes transition tables and universal/final state sets.
        Particles track their position in each z-DFA incrementally: when
        expanded by source symbol x, child z-states are advanced from parent
        z-states via O(1) dict lookups rather than re-running the full path.
        """
        target = self._target
        tlm = self.tlm
        pd = tlm._pd

        # Build base precover DFA (native FSA) for expansion
        base_dfa = tlm._backend.build_dfa(target)

        # Build preimage DFA (pynini): source strings with output == target
        exact_filter = _build_exact_filter(target, pd.out_label_map, pd.out_sym_table)
        preimage_pynini = pynini.compose(pd.pfst, exact_filter).project('input').optimize()

        # Per-symbol precover DFAs: transition tables, universal states, final states
        target_alphabet = tlm.fst.B - {EPSILON}
        reachable_z = []        # z values with non-empty precover
        trans_z = {}            # z -> {state: {label: next_state}}
        universal_z = {}        # z -> frozenset of universal state IDs
        final_z = {}            # z -> frozenset of final state IDs
        start_z = {}            # z -> start state ID

        in_label_map = pd.in_label_map
        input_alphabet_ids = pd.input_alphabet_ids

        for z in target_alphabet:
            extended = target + (z,)
            dfa_z = pd.precover(extended)
            s = dfa_z.start()
            if s == pynini.NO_STATE_ID:
                continue
            reachable_z.append(z)
            start_z[z] = s
            universal_z[z] = frozenset(_universal_states(dfa_z, input_alphabet_ids))
            trans_z[z], final_z[z] = _build_transition_table(dfa_z)

        # Preimage: transition table and final states
        pi_start = preimage_pynini.start()
        if pi_start != pynini.NO_STATE_ID:
            trans_pi, pi_finals = _build_transition_table(preimage_pynini)
        else:
            trans_pi, pi_finals = {}, frozenset()

        # --- Incremental z-state tracking ---
        # z_state_cache: source_path -> (z_states, pi_state)
        #   z_states: dict[z, state_in_precover_z]  (only live z's)
        #   pi_state: state in preimage DFA or _DEAD
        z_state_cache = {}

        def _init_z_states(source_path):
            """Compute z-states for a source_path from scratch."""
            if source_path in z_state_cache:
                return
            source_labels = [in_label_map[x] for x in source_path]

            z_states = {}
            for z in reachable_z:
                state = start_z[z]
                alive = True
                for label in source_labels:
                    next_s = trans_z[z].get(state, {}).get(label)
                    if next_s is None:
                        alive = False
                        break
                    state = next_s
                if alive:
                    z_states[z] = state

            # Preimage
            pi_state = _DEAD
            if pi_start != pynini.NO_STATE_ID:
                pi_state = pi_start
                for label in source_labels:
                    next_s = trans_pi.get(pi_state, {}).get(label)
                    if next_s is None:
                        pi_state = _DEAD
                        break
                    pi_state = next_s

            z_state_cache[source_path] = (z_states, pi_state)

        def _advance_z_states(parent_path, x):
            """Advance parent's z-states by one symbol x. O(|live_z|)."""
            child_path = parent_path + (x,)
            if child_path in z_state_cache:
                return
            parent_z, parent_pi = z_state_cache[parent_path]
            x_label = in_label_map[x]

            child_z = {}
            for z, state in parent_z.items():
                next_s = trans_z[z].get(state, {}).get(x_label)
                if next_s is not None:
                    child_z[z] = next_s

            child_pi = _DEAD
            if parent_pi is not _DEAD:
                next_s = trans_pi.get(parent_pi, {}).get(x_label)
                if next_s is not None:
                    child_pi = next_s

            z_state_cache[child_path] = (child_z, child_pi)

        # Classification from cached z-states (pure Python set lookups)
        _classify_cache = {}   # source_path -> (q_syms, r_syms, is_preimage)

        def _classify(source_path):
            """Classify a source_path using pre-computed z-states."""
            if source_path in _classify_cache:
                return _classify_cache[source_path]
            z_states, pi_state = z_state_cache[source_path]

            q_syms = set()
            r_syms = set()
            for z, state in z_states.items():
                if state in universal_z[z]:
                    q_syms.add(z)
                elif state in final_z[z]:
                    r_syms.add(z)

            is_preimage = pi_state is not _DEAD and pi_state in pi_finals

            _classify_cache[source_path] = (q_syms, r_syms, is_preimage)
            return q_syms, r_syms, is_preimage

        # Initialize z-states for incoming particles
        for p in self._particles:
            _init_z_states(p.source_path)

        # --- Accumulators ---
        scores = LogVector()
        carry_forward = defaultdict(list)
        cf_paths = defaultdict(set)

        def _score(particle):
            """Classify particle and accumulate scores. Returns (q_syms, r_syms)."""
            w = particle.log_weight

            q_syms, r_syms, is_preimage = _classify(particle.source_path)

            for y in q_syms:
                scores.logaddexp(y, w)

            eos_lp = particle.lm_state.logp_next[tlm.inner_lm.eos]
            if eos_lp > -np.inf:
                eos_w = w + eos_lp
                # Preimage EOS: skip when Q-absorbed
                if is_preimage and not q_syms:
                    scores.logaddexp(self.eos, eos_w)
                # Remainder
                for y in r_syms - q_syms:
                    scores.logaddexp(y, eos_w)

            return q_syms, r_syms

        def _add_carry_checked(y, particle):
            """Add to carry_forward[y] with prefix-domination check."""
            path = particle.source_path
            if path in cf_paths[y] or any(path[:k] in cf_paths[y] for k in range(len(path))):
                return
            carry_forward[y].append(particle)
            cf_paths[y].add(path)

        def _carry(particle, q_syms, r_syms):
            """Add particle to carry-forward sets."""
            for y in q_syms:
                carry_forward[y].append(particle)
                cf_paths[y].add(particle.source_path)
            for y in r_syms - q_syms:
                _add_carry_checked(y, particle)

        # --- Best-first search ---
        queue = list(self._particles)
        heapq.heapify(queue)

        expansions = 0
        while queue and expansions < tlm.max_expansions:
            expansions += 1
            particle = heapq.heappop(queue)

            q_syms, r_syms = _score(particle)
            _carry(particle, q_syms, r_syms)

            if q_syms:
                continue  # Q-absorbed, not expanded

            # Expand by source symbols via base_dfa arcs
            lm_logp = particle.lm_state.logp_next
            for x, next_state in base_dfa.arcs(particle.dfa_state):
                child_w = particle.log_weight + lm_logp[x]
                if child_w > -np.inf:
                    child_path = particle.source_path + (x,)
                    _advance_z_states(particle.source_path, x)
                    heapq.heappush(queue, Particle(
                        next_state,
                        particle.lm_state >> x,
                        child_w,
                        child_path,
                    ))

        # Budget exhausted — score remaining without expanding
        while queue:
            particle = heapq.heappop(queue)
            q_syms, r_syms = _score(particle)
            _carry(particle, q_syms, r_syms)

        scores[self.eos] = scores.get(self.eos, -np.inf)
        self._logp_next_cache = scores.normalize()
        self._carry_forward = carry_forward

    def __repr__(self):
        return f'PyniniTransducedState(target={self._target!r})'


class PyniniTransducedLM(LM):
    """Pynini-backed transduced language model.

    Uses pynini composition for DFA construction and universality-based
    Q/R classification, with the same particle-based beam search as
    TransducedLM.

    Args:
        inner_lm: LM with StateLM interface.
        fst: FST instance mapping source -> target.
        K: Carry-forward budget (max particles retained across steps).
        max_expansions: Best-first search budget per step.
        eos: Outer EOS token.
    """

    def __init__(self, inner_lm, fst, K, max_expansions=1000, eos='<EOS>'):
        self.inner_lm = inner_lm
        self.fst = fst
        self.K = K
        self.max_expansions = max_expansions
        self.eos = eos

        self._backend = PyniniPrecover(fst)
        self._pd = self._backend._pd

    def initial(self):
        """Return the initial PyniniTransducedState (empty target prefix)."""
        target = ()
        base_dfa = self._backend.build_dfa(target)
        inner_initial = self.inner_lm.initial()

        particles = [
            Particle(s, inner_initial, 0.0, ())
            for s in base_dfa.start
        ]

        return PyniniTransducedState(self, particles, target, 0.0)

    def __repr__(self):
        return f'PyniniTransducedLM(inner={self.inner_lm!r})'
