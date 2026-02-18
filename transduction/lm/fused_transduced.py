"""
Fused decomposition + LM search for transduced language models.

Given an inner LM P(source) and an FST (source -> target), computes
P(y | target_so_far) for each next target symbol y.  This is the same
computation as TransducedLM, but fused: instead of pre-computing the full
decomposition and then running LM-weighted search, we interleave both.
The powerset DFA is built lazily, so only states reachable through
high-probability source paths are ever materialized.

The algorithm is a best-first search over source-side paths.  Each
particle carries a DFA state, an inner LM state, and a cumulative log-weight.
At each step we pop the highest-weight item and ask: what role does this
DFA state play?

  QUOTIENT for y:   all continuations produce y -> score y with full weight
  REMAINDER for y:  source can stop (EOS) here and produce y -> score with weight + P(EOS)
  PREIMAGE:         source produced exactly the target at a final state -> score EOS
  none of the above: expand by advancing the inner LM one source symbol

After the search budget is exhausted, scores are normalized into a
conditional distribution over next target symbols.

Usage:
    from transduction import examples
    from transduction.lm.ngram import CharNgramLM
    from transduction.lm.fused_transduced import FusedTransducedLM

    inner = CharNgramLM.train("hello world", n=2)
    fst = examples.lowercase()
    tlm = FusedTransducedLM(inner, fst)

    state = tlm >> 'h'
    print(state.logp_next['e'])
"""

import heapq
import numpy as np
from collections import defaultdict

from transduction.lm.base import LM, LMState
from transduction.util import logsumexp, LogVector
from transduction.lm.transduced import Particle, _select_top_k
from transduction.rust_bridge import to_rust_fst


# ---------------------------------------------------------------------------
# Fused search
# ---------------------------------------------------------------------------

class _FusedSearch:
    """Best-first search that lazily builds the DFA and scores target symbols.

    Instantiated once per call to _compute_logp_next, then discarded.
    """

    def __init__(self, tlm, target, particles):
        self._max_steps = tlm.max_steps
        self._tlm = tlm
        self._inner_eos = tlm.inner_lm.eos

        # Create fresh Rust lazy DFA for this target prefix
        target_u32 = [tlm._sym_map[y] for y in target]
        tlm._rust_helper.new_step(target_u32)

        # Search accumulators
        self.scores = LogVector()                     # symbol -> log-weight
        self.eos_score = -np.inf
        self.carry_forward = defaultdict(list)        # symbol -> [Particle]

        # Resolve sentinel particles: dfa_state=None means the particle came
        # from a truncated Q/R state.  Replay source path through the new DFA
        # to find the correct state (same strategy as TransducedLM).
        resolved = []
        for item in particles:
            if item.dfa_state is None:
                s = tlm.run(item.source_path)
                assert s is not None
                resolved.append(Particle(s, item.lm_state, item.log_weight, item.source_path))
            else:
                resolved.append(item)
        particles = resolved

        self._queue = list(particles)
        heapq.heapify(self._queue)

    # --- Carry-forward ---

    def _add_carry(self, y, item):
        self.carry_forward[y].append(item)

    def _add_carry_sentinel(self, y, item):
        """Carry forward item with sentinel dfa_state=None for truncated states.

        The sentinel is resolved via source-path replay in the next step's
        _FusedSearch.__init__.
        """
        self._add_carry(y, Particle(None, item.lm_state, item.log_weight, item.source_path))

    # --- Scoring (no expansion) ---

    def _score_item(self, item):
        """Accumulate this item's contributions to scores.  Returns True if quotient."""
        result = self._tlm._rust_helper.classify(item.dfa_state)
        inv = self._tlm._inv_sym_map
        carry = self._add_carry if not result.has_truncated else self._add_carry_sentinel

        q_sym = inv[result.quotient_sym] if result.quotient_sym is not None else None
        r_syms = [inv[s] for s in result.remainder_syms]

        if q_sym is not None:
            self.scores.logaddexp(q_sym, item.log_weight)
            carry(q_sym, item)

        if result.is_preimage or r_syms:
            eos_lp = item.lm_state.logp_next[self._inner_eos]

            if result.is_preimage:
                self.eos_score = np.logaddexp(self.eos_score, item.log_weight + eos_lp)

            for y in r_syms:
                eos_w = item.log_weight + eos_lp
                self.scores.logaddexp(y, eos_w)
                carry(y, item)

        return q_sym is not None

    # --- Expansion ---

    def _expand(self, item):
        """Advance by each source symbol and push successors into the queue."""
        helper = self._tlm._rust_helper
        result = helper.classify(item.dfa_state)  # cached
        inv = self._tlm._inv_sym_map
        lm_logp_next = item.lm_state.logp_next

        trunc_resume_syms = set()

        for x_u32, dest_sid in helper.arcs(item.dfa_state):
            x = inv[x_u32]
            w = float(item.log_weight + lm_logp_next[x])
            if w > -np.inf:
                child = Particle(dest_sid, item.lm_state >> x, w,
                                 item.source_path + (x,))
                heapq.heappush(self._queue, child)

            # Truncation-boundary carry-forward: if the item is non-truncated
            # but its successor crosses into truncation, carry forward the item
            # at the truncation-boundary symbols (mirrors resume_frontiers).
            if not result.has_truncated:
                dest_result = helper.classify(dest_sid)
                if dest_result.has_truncated:
                    for y_u32 in dest_result.trunc_output_syms:
                        trunc_resume_syms.add(inv[y_u32])

        for y in trunc_resume_syms:
            self._add_carry(y, item)

    # --- Main loop ---

    def search(self):
        """Run the search.  Returns (scores, eos_score, carry_forward)."""
        steps = 0
        while self._queue and steps < self._max_steps:
            steps += 1
            item = heapq.heappop(self._queue)
            if not self._score_item(item):
                self._expand(item)

        # Budget exhausted — score remaining items without expanding
        while self._queue:
            self._score_item(heapq.heappop(self._queue))

        return self.scores, self.eos_score, self.carry_forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FusedTransducedState(LMState):
    """Immutable state for FusedTransducedLM.

    Unlike TransducedState, this does not carry a PeekabooState — it performs
    decomposition inline during the LM-weighted search.
    """

    def __init__(self, tlm, particles, target, logp, path=()):
        self.tlm = tlm
        self.eos = tlm.eos
        self._particles = particles
        self._target = target
        self.logp = logp
        self.path = path
        self._logp_next_cache = None
        self._carry_forward_cache = None

    def decode_dfa_state(self, state_id):
        """Decode a DFA state ID to NFA constituents.

        Delegates to FusedTransducedLM.decode_dfa_state with the current target.
        """
        return self.tlm.decode_dfa_state(state_id, self._target)

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

        new_particles = self._carry_forward_cache.get(y, [])
        new_particles = _select_top_k(new_particles, self.tlm.max_beam)

        return FusedTransducedState(
            self.tlm, new_particles,
            self._target + (y,),
            self.logp + self._logp_next_cache[y],
            path=self.path + (y,),
        )

    def _compute_logp_next(self):
        search = _FusedSearch(self.tlm, self._target, self._particles)
        scores, eos_score, carry_forward = search.search()

        scores[self.eos] = eos_score
        self._logp_next_cache = scores.normalize()
        self._carry_forward_cache = carry_forward

    def _repr_html_(self):
        from transduction.viz import render_particles_html, _format_nfa_set
        decode_fn = None
        if hasattr(self.tlm, 'decode_dfa_state'):
            decode_cache = {}
            target_tuple = tuple(self._target)
            def decode_fn(dfa_state):    # pylint: disable=E0102
                if dfa_state not in decode_cache:
                    try:
                        decoded = self.tlm.decode_dfa_state(dfa_state, target_tuple)
                        decode_cache[dfa_state] = _format_nfa_set(decoded)
                    except Exception:
                        decode_cache[dfa_state] = str(dfa_state)
                return decode_cache[dfa_state]
        return render_particles_html(
            'FusedTransducedState', self._particles,
            list(self._target), self.logp,
            decode_fn=decode_fn,
        )

    def __repr__(self):
        return f'FusedTransducedState(target={self._target!r})'


class FusedTransducedLM(LM):
    """Fused transduced language model.

    Combines decomposition and LM-weighted search into a single best-first
    pass.  The lazy DFA is constructed on demand per target step; only states
    reachable via high-probability source paths are expanded.
    """

    def __init__(self, inner_lm, fst, max_steps=1000, max_beam=100, eos='<EOS>'):
        import transduction_core

        self.inner_lm = inner_lm
        self.fst = fst
        self.max_steps = max_steps
        self.max_beam = max_beam
        self.eos = eos

        # Build Rust FST and helper
        rust_fst, sym_map, state_map = to_rust_fst(fst)
        self._rust_helper = transduction_core.RustLazyPeekabooDFA(rust_fst)
        self._sym_map = {k: v for k, v in sym_map.items()}
        self._inv_sym_map = {v: k for k, v in sym_map.items()}
        self._state_map = state_map

    def decode_dfa_state(self, state_id, target):
        """Decode a lazy-DFA state ID to NFA constituents.

        Returns frozenset of (fst_state, buffer_tuple, truncated) matching
        the Python PeekabooLookaheadNFA representation.
        """
        if not hasattr(self, '_inv_maps'):
            idx_to_sym_raw = self._rust_helper.idx_to_sym_map()
            inv_sym = self._inv_sym_map
            inv_state = {v: k for k, v in self._state_map.items()}
            self._inv_maps = (idx_to_sym_raw, inv_sym, inv_state)
        idx_to_sym_raw, inv_sym, inv_state = self._inv_maps

        NO_EXTRA = 0xFFFF
        raw = self._rust_helper.decode_state(state_id)

        result = set()
        for fst_state_u32, buf_len, extra_sym_idx, truncated in raw:
            py_fst_state = inv_state.get(fst_state_u32, fst_state_u32)
            if extra_sym_idx == NO_EXTRA:
                buf = target[:buf_len]
            else:
                sym_u32 = idx_to_sym_raw[extra_sym_idx]
                py_sym = inv_sym[sym_u32]
                buf = target[:buf_len - 1] + (py_sym,)
            result.add((py_fst_state, buf, truncated))

        return frozenset(result)

    def run(self, source_path):
        """Run a source path through the current-step DFA. Returns state or None."""
        path_u32 = [self._sym_map[x] for x in source_path]
        return self._rust_helper.run(path_u32)

    def initial(self):
        """Return the initial FusedTransducedState (empty target prefix)."""
        self._rust_helper.new_step([])
        start_ids = self._rust_helper.start_ids()

        inner_initial = self.inner_lm.initial()
        particles = [Particle(s, inner_initial, 0.0, ()) for s in start_ids]

        return FusedTransducedState(self, particles, (), 0.0)

    def __repr__(self):
        return f'FusedTransducedLM(inner={self.inner_lm!r}, fst={self.fst!r})'
