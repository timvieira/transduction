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

    state = tlm.initial()
    state = state << 'h'
    print(state.logp_next['e'])
"""

import numpy as np
from dataclasses import dataclass

from arsenal.datastructures import LocatorMaxHeap

from transduction.lm.base import LMState, LogpNext
from transduction.lm.transduced import logsumexp, _to_key, BeamItem
from transduction.fst import EPSILON
from transduction.precover_nfa import PeekabooLookaheadNFA as PeekabooPrecover
from transduction.peekaboo_incremental import FstUniversality, TruncatedDFA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class DFAStateMeta:
    """Summary of what a powerset DFA state tells us about the target prefix.

    A DFA state is a frozenset of NFA tuples (fst_state, output_string, truncated).
    This dataclass pre-computes the four things the search needs to know.
    """
    relevant_symbols: set   # candidate next target symbols (at position len(target))
    final_symbols: set      # subset where the FST state is also final
    is_preimage: bool       # output == target at a final FST state (exact match)
    has_truncated: bool     # at least one tuple hit the lookahead bound


def _extract_meta(dfa_state, target, fst_is_final):
    """Build a DFAStateMeta from a powerset DFA state."""
    N = len(target)
    relevant = set()
    final = set()
    has_truncated = False
    is_preimage = False
    for fst_state, output, truncated in dfa_state:
        if len(output) == N and fst_is_final(fst_state):
            is_preimage = True
        if len(output) > N:
            sym = output[N]
            relevant.add(sym)
            if output.startswith(target) and fst_is_final(fst_state):
                final.add(sym)
        has_truncated = has_truncated or truncated
    return DFAStateMeta(relevant, final, is_preimage, has_truncated)


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
        fst = tlm.fst
        self._target = target
        self._max_steps = tlm.max_steps
        self._fst = fst

        # Lazy DFA for the current target prefix
        nfa = PeekabooPrecover(fst, target)
        self._raw_dfa = nfa.det()
        self._dfa = self._raw_dfa.cache()

        # Universality (precomputed per-FST, shared across steps)
        self._univ = tlm._univ
        self._all_input_universal = self._univ.all_input_universal
        self._source_alphabet = fst.A - {EPSILON}

        # Caches (populated lazily)
        self._meta_cache = {}
        self._classify_cache = {}
        self._univ_filters = {}

        # Search accumulators
        self.scores = {}          # symbol -> [log-weights]
        self.eos_scores = []      # [log-weights]
        self.carry_forward = {}   # symbol -> [BeamItem]

        # Inner LM's EOS token
        self._inner_eos = (
            tlm.inner_lm.eos if hasattr(tlm.inner_lm, 'eos')
            else tlm.inner_lm.initial().eos
        )

        # Seed the priority queue
        self._queue = LocatorMaxHeap()
        for item in beam:
            self._queue[item] = item.weight

    # --- Caching helpers ---

    def _get_meta(self, dfa_state):
        meta = self._meta_cache.get(dfa_state)
        if meta is None:
            meta = _extract_meta(dfa_state, self._target, self._fst.is_final)
            self._meta_cache[dfa_state] = meta
        return meta

    def _get_univ_filter(self, symbol):
        uf = self._univ_filters.get(symbol)
        if uf is None:
            trunc_dfa = TruncatedDFA(
                dfa=self._raw_dfa, fst=self._fst, target=self._target + symbol
            )
            uf = self._univ.make_filter(
                self._fst, self._target + symbol, trunc_dfa, self._source_alphabet
            )
            self._univ_filters[symbol] = uf
        return uf

    # --- Classification ---

    def _is_universal(self, symbol, dfa_state):
        """Is `symbol` a quotient at `dfa_state`?"""
        if self._all_input_universal:
            return symbol in self._get_meta(dfa_state).final_symbols
        return self._get_univ_filter(symbol).is_universal(dfa_state)

    def _classify(self, dfa_state):
        """Classify relevant symbols as quotient or remainder.

        Returns (quotient_syms, remainder_syms).

        Peekaboo shortcut: at most one symbol can be the quotient.  Once found,
        the expensive universality check is skipped for remaining symbols.
        """
        result = self._classify_cache.get(dfa_state)
        if result is not None:
            return result

        meta = self._get_meta(dfa_state)

        # Find the (at most one) quotient symbol.
        quotient = None
        for y in meta.relevant_symbols:
            if self._is_universal(y, dfa_state):
                quotient = y
                break

        quotient_syms = {quotient} if quotient is not None else set()
        remainder_syms = {
            y for y in meta.relevant_symbols
            if y in meta.final_symbols and y != quotient
        }

        result = (quotient_syms, remainder_syms)
        self._classify_cache[dfa_state] = result
        return result

    # --- Scoring (no expansion) ---

    def _score_item(self, item):
        """Accumulate this item's contributions to scores.  Returns True if quotient."""
        meta = self._get_meta(item.dfa_state)
        q_syms, r_syms = self._classify(item.dfa_state)

        # Quotient: full weight, no LM eval needed
        for y in q_syms:
            self.scores.setdefault(y, []).append(item.weight)
            self.carry_forward.setdefault(y, []).append(item)

        # Preimage and remainder both need P_inner(EOS)
        if meta.is_preimage or r_syms:
            eos_lp = item.lm_state.logp_next[self._inner_eos]

            if meta.is_preimage:
                self.eos_scores.append(item.weight + eos_lp)

            for y in r_syms:
                self.scores.setdefault(y, []).append(item.weight + eos_lp)
                self.carry_forward.setdefault(y, []).append(item)

        return bool(q_syms)

    # --- Expansion ---

    def _expand(self, item):
        """Advance by each source symbol and push successors into the queue."""
        meta = self._get_meta(item.dfa_state)
        lm_logp_next = item.lm_state.logp_next
        N = len(self._target)

        trunc_resume_syms = set()

        for x, next_dfa_state in self._dfa.arcs(item.dfa_state):
            w = float(item.weight + lm_logp_next[x])
            if w == -np.inf:
                continue

            self._queue[BeamItem(
                dfa_state=next_dfa_state,
                lm_state=item.lm_state << x,
                weight=w,
            )] = w

            # If successor has truncated tuples, note which target symbols
            # sit at the truncation boundary so we can resume there next step.
            if meta.has_truncated:
                next_meta = self._get_meta(next_dfa_state)
                if next_meta.has_truncated:
                    for _, output, truncated in next_dfa_state:
                        if truncated and len(output) > N:
                            trunc_resume_syms.add(output[N])

        for y in trunc_resume_syms:
            self.carry_forward.setdefault(y, []).append(item)

    # --- Main loop ---

    def run(self):
        """Run the search.  Returns (scores, eos_scores, carry_forward)."""
        steps = 0
        while self._queue and steps < self._max_steps:
            steps += 1
            item, _ = self._queue.pop()
            is_quotient = self._score_item(item)
            if not is_quotient:
                self._expand(item)

        # Budget exhausted — score remaining items without expanding
        while self._queue:
            item, _ = self._queue.pop()
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

    def __lshift__(self, y):
        if y == self.eos:
            raise ValueError(f"Out of vocabulary: {y!r}")
        self._ensure_computed()

        key = _to_key(y)
        if key is None or key not in self._logp_next_cache:
            raise ValueError(f"Out of vocabulary: {y!r}")

        lp_y = self._logp_next_cache[y]
        new_beam = self._carry_forward_cache.get(key, [])

        if len(new_beam) > self.tlm.max_beam:
            new_beam = sorted(new_beam, key=lambda it: it.weight, reverse=True)
            new_beam = new_beam[:self.tlm.max_beam]

        return FusedTransducedState(
            self.tlm, new_beam,
            self._target + key,
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

    def __repr__(self):
        return f'FusedTransducedState(target={self._target!r})'


class FusedTransducedLM:
    """Fused transduced language model.

    Combines decomposition and LM-weighted search into a single best-first
    pass.  The lazy DFA is constructed on demand per target step; only states
    reachable via high-probability source paths are expanded.
    """

    def __init__(self, inner_lm, fst, max_steps=1000, max_beam=100, eos='<EOS>'):
        self.inner_lm = inner_lm
        self.fst = fst
        self.max_steps = max_steps
        self.max_beam = max_beam
        self.eos = eos
        self._univ = FstUniversality(fst)

    def initial(self):
        """Return the initial FusedTransducedState (empty target prefix)."""
        nfa = PeekabooPrecover(self.fst, '')
        raw_dfa = nfa.det()

        start_states = list(raw_dfa.start())

        inner_initial = self.inner_lm.initial()
        beam = [
            BeamItem(
                dfa_state=s,
                lm_state=inner_initial,
                weight=0.0,
            )
            for s in start_states
        ]

        return FusedTransducedState(self, beam, '', 0.0)

    def __repr__(self):
        return f'FusedTransducedLM(inner={self.inner_lm!r}, fst={self.fst!r})'
