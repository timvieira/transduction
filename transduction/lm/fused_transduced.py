"""
Fused decomposition + LM search for transduced language models.

Given an inner LM P(source) and an FST (source -> target), computes
P(y | target_so_far) for each next target symbol y.  This is the same
computation as TransducedLM, but fused: instead of pre-computing the full
decomposition and then running LM-weighted search, we interleave both.
The powerset DFA is built lazily, so only states reachable through
high-probability source paths are ever materialized.

The algorithm is a best-first search over source-side paths.  Each beam
item carries a DFA state, an inner LM state, and a cumulative log-weight.
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

from transduction.lm.base import LM, LMState, LogpNext
from transduction.lm.transduced import logsumexp, BeamItem, _format_source_path
from transduction.rust_bridge import to_rust_fst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(scores, eos_scores, eos_token):
    """Turn per-symbol log-weight lists into a conditional distribution."""
    all_raw = [logsumexp(ws) for ws in scores.values()]

    eos_raw = logsumexp(eos_scores)
    if eos_raw > -np.inf:
        all_raw.append(eos_raw)

    Z = logsumexp(all_raw)

    out = {y: logsumexp(ws) - Z for y, ws in scores.items()}
    out[eos_token] = (eos_raw - Z) if Z > -np.inf else -np.inf
    return LogpNext(out)


# ---------------------------------------------------------------------------
# Fused search
# ---------------------------------------------------------------------------

class _FusedSearch:
    """Best-first search that lazily builds the DFA and scores target symbols.

    Instantiated once per call to _compute_logp_next, then discarded.
    """

    def __init__(self, tlm, target, beam):
        self._target = target
        self._max_steps = tlm.max_steps
        self._tlm = tlm

        # Create fresh Rust lazy DFA for this target prefix
        target_u32 = [tlm._sym_map[y] for y in target]
        tlm._rust_helper.new_step(target_u32)

        # Inverse symbol map for converting u32 back to Python symbols
        self._inv = tlm._inv_sym_map

        # Search accumulators
        self.scores = {}          # symbol -> [log-weights]
        self.eos_scores = []      # [log-weights]
        self.carry_forward = {}   # symbol -> [BeamItem]

        # Root-family tracking for carry-forward deduplication.
        # Same invariant as TransducedLM: within each root family, only the
        # shallowest carry-forward entry per target symbol is kept.  The
        # priority queue pops highest-weight items first, and within a root
        # family the shallowest item has the highest weight (monotone), so
        # "first one wins" is correct.
        self._root_of = {}        # id(item) -> root index
        self._carried = set()     # (root_id, y) pairs already added
        for i, item in enumerate(beam):
            self._root_of[id(item)] = i

        # Inner LM's EOS token
        self._inner_eos = (
            tlm.inner_lm.eos if hasattr(tlm.inner_lm, 'eos')
            else tlm.inner_lm.initial().eos
        )

        # Seed the priority queue
        self._queue = list(beam)
        heapq.heapify(self._queue)

    # --- Carry-forward dedup ---

    def _add_carry(self, y, item):
        rid = self._root_of[id(item)]
        if (rid, y) in self._carried:
            return
        self._carried.add((rid, y))
        self.carry_forward.setdefault(y, []).append(item)

    # --- Scoring (no expansion) ---

    def _score_item(self, item):
        """Accumulate this item's contributions to scores.  Returns True if quotient."""
        result = self._tlm._rust_helper.classify(item.dfa_state)
        inv = self._inv

        q_sym = inv[result.quotient_sym] if result.quotient_sym is not None else None
        r_syms = [inv[s] for s in result.remainder_syms]

        # Quotient: full weight, no LM eval needed
        is_quotient = q_sym is not None
        if is_quotient:
            self.scores.setdefault(q_sym, []).append(item.weight)
            self._add_carry(q_sym, item)

        # Preimage and remainder both need P_inner(EOS)
        if result.is_preimage or r_syms:
            eos_lp = item.lm_state.logp_next[self._inner_eos]

            if result.is_preimage:
                self.eos_scores.append(item.weight + eos_lp)

            for y in r_syms:
                self.scores.setdefault(y, []).append(item.weight + eos_lp)
                self._add_carry(y, item)

        return is_quotient

    # --- Expansion ---

    def _expand(self, item):
        """Advance by each source symbol and push successors into the queue."""
        result = self._tlm._rust_helper.classify(item.dfa_state)  # cached
        arcs = self._tlm._rust_helper.arcs(item.dfa_state)
        inv = self._inv
        lm_logp_next = item.lm_state.logp_next
        rid = self._root_of[id(item)]

        trunc_resume_syms = set()

        for x_u32, dest_sid in arcs:
            x = inv[x_u32]
            w = float(item.weight + lm_logp_next[x])
            if w == -np.inf:
                continue

            child = BeamItem(
                dfa_state=dest_sid,
                lm_state=item.lm_state >> x,
                weight=w,
            )
            self._root_of[id(child)] = rid
            heapq.heappush(self._queue, child)

            # If successor has truncated tuples, note which target symbols
            # sit at the truncation boundary so we can resume there next step.
            if result.has_truncated:
                dest_result = self._tlm._rust_helper.classify(dest_sid)
                if dest_result.has_truncated:
                    for y_u32 in dest_result.trunc_output_syms:
                        trunc_resume_syms.add(inv[y_u32])

        for y in trunc_resume_syms:
            self._add_carry(y, item)

    # --- Main loop ---

    def run(self):
        """Run the search.  Returns (scores, eos_scores, carry_forward)."""
        steps = 0
        while self._queue and steps < self._max_steps:
            steps += 1
            item = heapq.heappop(self._queue)
            is_quotient = self._score_item(item)
            if not is_quotient:
                self._expand(item)

        # Budget exhausted — score remaining items without expanding
        while self._queue:
            item = heapq.heappop(self._queue)
            self._score_item(item)

        return self.scores, self.eos_scores, self.carry_forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FusedTransducedState(LMState):
    """Immutable state for FusedTransducedLM.

    Unlike TransducedState, this does not carry a PeekabooState — it performs
    decomposition inline during the LM-weighted search.
    """

    def __init__(self, tlm, beam, target, logp, history=()):
        self.tlm = tlm
        self.eos = tlm.eos
        self._beam = beam
        self._target = target
        self.logp = logp
        self.history = history
        self._logp_next_cache = None
        self._carry_forward_cache = None

    def _ensure_computed(self):
        if self._logp_next_cache is None:
            self._compute_logp_next()

    @property
    def logp_next(self):
        self._ensure_computed()
        return self._logp_next_cache

    def __rshift__(self, y):
        if y == self.eos:
            raise ValueError(f"Out of vocabulary: {y!r}")
        self._ensure_computed()

        if y not in self._logp_next_cache:
            raise ValueError(f"Out of vocabulary: {y!r}")

        lp_y = self._logp_next_cache[y]
        new_beam = self._carry_forward_cache.get(y, [])

        if len(new_beam) > self.tlm.max_beam:
            new_beam = sorted(new_beam, key=lambda it: it.weight, reverse=True)
            new_beam = new_beam[:self.tlm.max_beam]

        return FusedTransducedState(
            self.tlm, new_beam,
            self._target + (y,),
            self.logp + lp_y,
            history=(self.history, y),
        )

    def _compute_logp_next(self):
        search = _FusedSearch(self.tlm, self._target, self._beam)
        scores, eos_scores, carry_forward = search.run()
        self._logp_next_cache = _normalize(scores, eos_scores, self.tlm.eos)
        self._carry_forward_cache = carry_forward

    def path(self):
        tokens = []
        h = self.history
        while h:
            h, token = h
            tokens.append(token)
        tokens.reverse()
        return tokens

    def _repr_html_(self):
        from transduction.lm.transduced import _render_beam_html, _format_nfa_set
        decode_fn = None
        if hasattr(self.tlm, 'decode_dfa_state'):
            decode_cache = {}
            target_tuple = tuple(self._target)
            def decode_fn(dfa_state):
                if dfa_state not in decode_cache:
                    try:
                        decoded = self.tlm.decode_dfa_state(dfa_state, target_tuple)
                        decode_cache[dfa_state] = _format_nfa_set(decoded)
                    except Exception:
                        decode_cache[dfa_state] = str(dfa_state)
                return decode_cache[dfa_state]
        return _render_beam_html(
            'FusedTransducedState', self._beam,
            list(self._target), self.logp,
            weight_attr='weight',
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

    def initial(self):
        """Return the initial FusedTransducedState (empty target prefix)."""
        self._rust_helper.new_step([])
        start_ids = self._rust_helper.start_ids()

        inner_initial = self.inner_lm.initial()
        beam = [
            BeamItem(
                dfa_state=s,
                lm_state=inner_initial,
                weight=0.0,
            )
            for s in start_ids
        ]

        return FusedTransducedState(self, beam, (), 0.0)

    def __repr__(self):
        return f'FusedTransducedLM(inner={self.inner_lm!r}, fst={self.fst!r})'
