"""
Incremental transduced language model with particle-based approximate inference.

Given an inner LM P_inner(source) and an FST mapping source -> target,
TransducedLM estimates P(y | target_so_far) by maintaining a beam of K
particles (source-side hypotheses) that evolve as each target symbol is added.
Each particle tracks a DFA state (powerset of precover NFA states) and an
inner LM state; at quotient states, exact marginalization sums over infinitely
many source continuations, providing dramatic variance reduction.

Per-step expansion uses beam search through the decomposition's DFA with beam
width K.  Layers proceed by source-symbol depth: at each layer, all beam
particles are classified (Q/R/preimage), scored, and non-Q particles are
expanded.  Top K candidates form the next beam.  Beam search terminates when:
  - The beam is empty (all particles absorbed by Q/R), or
  - The monotone weight bound is satisfied: the best expandable particle has
    weight <= the K-th highest carry-forward weight.  Since particle weights
    only decrease with source-symbol depth, no descendant can displace the
    current top-K carry-forward set.

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

import heapq
import numpy as np

from transduction.lm.base import LM, LMState, LogpNext

# Default incremental decomposition (Rust backend).
from transduction.rust_bridge import RustPeekabooState as _DefaultDecompState
from transduction.rust_bridge import _RustPeekabooUniv as _DefaultUniv


def logsumexp(arr):
    arr = np.array(arr, dtype=np.float64)
    arr = arr[arr > -np.inf]
    if len(arr) == 0:
        return -np.inf
    vmax = arr.max()
    arr -= vmax
    np.exp(arr, out=arr)
    out = np.log(arr.sum())
    out += vmax
    return out


def _format_source_path(lm_state):
    """Format an LM state's source path for display."""
    path = lm_state.path()
    if not path:
        return '(empty)'
    try:
        return bytes(path).decode('utf-8', errors='replace')
    except TypeError:
        return ''.join(str(s) for s in path)


def _format_nfa_element(fst_state, buf, truncated):
    """Format a single NFA element (fst_state, buffer, truncated) compactly."""
    if not buf:
        buf_str = 'ε'
    elif all(isinstance(b, str) and len(b) == 1 for b in buf):
        s = ''.join(buf)
        buf_str = repr(s) if len(s) <= 6 else repr(s[:5]) + '…'
    else:
        items = [str(b) for b in buf[:4]]
        buf_str = '(' + ','.join(items) + ('…' if len(buf) > 4 else '') + ')'
    trunc = '†' if truncated else ''
    return f'({fst_state}, {buf_str}{trunc})'


def _format_nfa_set(decoded_set):
    """Format a decoded NFA state set for compact display."""
    elements = sorted(decoded_set, key=lambda x: (repr(x[0]), x[1]))
    parts = [_format_nfa_element(*e) for e in elements]
    MAX = 4
    if len(parts) <= MAX:
        return '{' + ', '.join(parts) + '}'
    return '{' + ', '.join(parts[:MAX]) + f', \u2026+{len(parts)-MAX}' + '}'


# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------

class Particle:
    """A weighted source-prefix hypothesis.

    Tracks a DFA state (powerset of precover NFA states) that captures the
    transducer's progress, the inner LM state, and a log importance weight.
    """
    __slots__ = ('dfa_state', 'lm_state', 'log_weight')

    def __init__(self, dfa_state, lm_state, log_weight):
        self.dfa_state = dfa_state
        self.lm_state = lm_state
        self.log_weight = log_weight

    def __lt__(self, other):
        # max-heap: higher weight = higher priority
        return self.log_weight > other.log_weight

    def __repr__(self):
        return f'Particle(w={self.log_weight:.3f})'


# Backward-compat alias for FusedTransducedLM and benchmarks.
class BeamItem:
    # NOTE: intentionally no __slots__ — autoreload + __slots__ causes
    # "descriptor doesn't apply" errors when cross-module reload.
    def __init__(self, dfa_state, lm_state, weight):
        self.dfa_state = dfa_state
        self.lm_state = lm_state
        self.weight = weight
    def __lt__(self, other):
        return self.weight > other.weight


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _select_top_k(particles, k):
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

class TransducedState(LMState):
    """Immutable state for the TransducedLM.

    Supports:
        state >> y         -> new state (advance by target symbol y)
        state.logp_next[y] -> log P(y | target_so_far)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token

    Each state holds K particles (source-prefix hypotheses). Computing
    logp_next runs beam search (width K) through the decomposition's DFA:
    at each layer, particles are classified, scored at Q/R/preimage states,
    and non-Q particles are expanded by source symbols. The carry-forward
    mechanism passes particles at Q/R/resume_frontier states to the next
    target step.
    """

    def __init__(self, tlm, peekaboo_state, particles, logp, history=()):
        self.tlm = tlm
        self.eos = tlm.eos
        self._peekaboo_state = peekaboo_state
        self._particles = particles
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
        if y == self.eos or y not in self.tlm.fst.B:
            raise ValueError(f"Out of vocabulary: {y!r}")
        self._ensure_computed()

        if y not in self._logp_next_cache:
            raise ValueError(f"Out of vocabulary: {y!r}")

        lp_y = self._logp_next_cache[y]

        # Advance peekaboo decomposition state
        new_peekaboo = self._peekaboo_state >> y

        # Get carry-forward particles for this symbol
        cf_particles = self._carry_forward_cache.get(y, [])

        K = self.tlm.K
        new_logp = self.logp + lp_y
        new_particles = _select_top_k(cf_particles, K)

        return TransducedState(
            self.tlm, new_peekaboo, new_particles,
            new_logp, history=(self.history, y),
        )

    def _compute_logp_next(self):
        """Beam search through the DFA to score next target symbols.

        Layer-by-layer beam search with beam width K:
        1. Classify all beam particles: Q particles score and exit (exact
           marginalization), R particles score (weight * P(EOS)), preimage
           particles score outer EOS.
        2. Carry-forward: particles at Q/R/resume_frontier states are saved.
           Root-family tracking ensures at most one entry per (root, symbol)
           pair — deeper descendants are redundant (DFA determinism).
        3. Expand non-Q particles by source symbols.
        4. Select top K candidates -> new beam.
        5. Terminate when beam is empty or monotone weight bound is met:
           best expandable weight <= K-th highest carry-forward weight.
           Since weights only decrease with depth, no descendant can
           displace the current top-K carry-forward set.
        6. Normalize scores -> conditional distribution.
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

        preimage_stops = self._peekaboo_state.preimage_stops

        # Inner LM EOS token
        EOS = (self.tlm.inner_lm.eos if hasattr(self.tlm.inner_lm, 'eos')
               else self.tlm.inner_lm.initial().eos)

        K = self.tlm.K

        # --- Accumulators ---
        scores = {}          # y -> [log-weights]
        eos_scores = []      # [log-weights] for outer EOS
        carry_forward = {}   # y -> [particle, ...]

        # Root-family tracking for carry-forward deduplication.
        #
        # Each initial particle (from self._particles) defines a "root
        # family".  Within a single BFS, prefix-dominated carry-forward
        # entries can only arise among descendants of the same root:
        # initial particles have non-prefix source paths (guaranteed by the
        # previous step's invariant), and the DFA is deterministic with a
        # single start state, so particles from different roots can never
        # reach the same DFA state.
        #
        # For each (root_id, y), only the shallowest descendant needs to be
        # carried forward — deeper ones will be reproduced when the shallow
        # particle is expanded in the next step's BFS.  Since BFS processes
        # layers shallowest-first, the first entry for a (root_id, y) pair
        # is always the shallowest, so "first one wins" is sufficient.
        root_of = {}               # id(particle) -> root index
        carried = set()            # (root_id, y) pairs already added

        for i, p in enumerate(self._particles):
            root_of[id(p)] = i

        def _add_carry(y, particle):
            rid = root_of[id(particle)]
            if (rid, y) in carried:
                return
            carried.add((rid, y))
            carry_forward.setdefault(y, []).append(particle)

        # Top-K carry-forward weight tracker (min-heap of size K).
        # cf_top_k[0] is the K-th highest carry-forward weight when full.
        cf_top_k = []

        def classify_and_score(particle):
            """Classify particle's DFA state, accumulate scores, save carry-forward.

            Returns True if the particle is at a quotient state (skip expansion).
            """
            d = particle.dfa_state
            w = particle.log_weight

            q_syms = q_of.get(d)
            r_syms = r_of.get(d)
            is_preimage = d in preimage_stops
            is_quotient = q_syms is not None

            # Only trigger an LM forward pass (logp_next[EOS]) when needed:
            # preimage scoring, or remainder for symbols not covered by quotient.
            needs_eos = is_preimage or (
                r_syms is not None
                and (not is_quotient or not r_syms.issubset(q_syms))
            )
            eos_lp = None
            if needs_eos:
                eos_lp = particle.lm_state.logp_next[EOS]

            # Preimage: source produced exactly the target prefix at a final state
            if is_preimage and eos_lp is not None and eos_lp > -np.inf:
                eos_scores.append(w + eos_lp)

            # Quotient: full weight contributes (exact marginalization over futures)
            if is_quotient:
                for y in q_syms:
                    scores.setdefault(y, []).append(w)

            # Remainder: weight * P(EOS), skipping symbols covered by quotient
            if r_syms is not None and eos_lp is not None and eos_lp > -np.inf:
                eos_w = w + eos_lp
                for y in r_syms:
                    if not is_quotient or y not in q_syms:
                        scores.setdefault(y, []).append(eos_w)

            # Carry-forward: save particle at resume_frontier / Q / R states.
            # _add_carry enforces the no-prefix-domination invariant.
            carry_syms = set()
            rf = resume_of.get(d)
            if rf is not None:
                carry_syms |= rf
            if q_syms is not None:
                carry_syms |= q_syms
            if r_syms is not None:
                carry_syms |= r_syms
            for y in carry_syms:
                _add_carry(y, particle)

            # Track carry-forward weights for early termination.
            if carry_syms:
                if len(cf_top_k) < K:
                    heapq.heappush(cf_top_k, w)
                elif w > cf_top_k[0]:
                    heapq.heapreplace(cf_top_k, w)

            return is_quotient

        # --- Beam search ---
        # Each iteration: classify candidates -> prune non-Q to K -> expand.
        # Initial "candidates" are the carry-forward particles from the parent.
        # Classification happens BEFORE pruning so that all Q/R candidates
        # are scored regardless of beam width.
        candidates = list(self._particles)

        for _ in range(self.tlm.max_layers):
            if not candidates:
                break

            # Classify all candidates: Q/R score and exit, non-Q continue.
            to_expand = []
            for particle in candidates:
                is_q = classify_and_score(particle)
                if not is_q:
                    to_expand.append(particle)

            if not to_expand:
                break

            # Early termination: monotone weight bound.
            # Particle weights only decrease with source-symbol depth, so if
            # the best non-Q candidate can't beat the K-th highest carry-
            # forward weight, no descendant can either.
            if len(cf_top_k) >= K:
                best_w = max(p.log_weight for p in to_expand)
                if best_w <= cf_top_k[0]:
                    break

            # Prune non-Q candidates to K (beam width).
            beam = _select_top_k(to_expand, K)

            # Expand beam particles by source symbols.
            candidates = []
            for particle in beam:
                rid = root_of[id(particle)]
                lm_logp = particle.lm_state.logp_next
                for x, next_dfa_state in dfa.arcs(particle.dfa_state):
                    child_w = particle.log_weight + lm_logp[x]
                    if child_w > -np.inf:
                        child = Particle(
                            next_dfa_state,
                            particle.lm_state >> x,
                            child_w,
                        )
                        root_of[id(child)] = rid
                        candidates.append(child)

        # --- Normalize ---
        raw_scores = {}
        for y, w_list in scores.items():
            raw_scores[y] = logsumexp(w_list)
        raw_scores[EOS] = logsumexp(eos_scores)

        Z_parts = list(raw_scores.values())
        Z = logsumexp(Z_parts)

        normalized = {}
        for y, raw in raw_scores.items():
            normalized[y] = raw - Z

        self._logp_next_cache = LogpNext(normalized)
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
        import html as _html
        from transduction.viz import format_table
        particles = self._particles
        target = self.path()
        target_str = ''.join(str(y) for y in target) if target else '(empty)'
        header = (f'<b>TransducedState</b> '
                  f'target={target_str!r}, K={len(particles)}, '
                  f'logp={self.logp:.4f}<br>')
        if not particles:
            return header
        # Classify DFA states from current decomposition
        decomp = self._peekaboo_state.decomp
        q_states = set()
        r_states = set()
        for d in decomp.values():
            q_states.update(d.quotient)
            r_states.update(d.remainder)
        log_weights = np.array([p.log_weight for p in particles])
        log_Z = logsumexp(log_weights)

        # Check if state decoding is available
        can_decode = hasattr(self._peekaboo_state, 'decode_dfa_state')

        # Cache decoded state names
        decode_cache = {}
        def decode_state(dfa_state):
            if not can_decode:
                return str(dfa_state)
            if dfa_state not in decode_cache:
                try:
                    decoded = self._peekaboo_state.decode_dfa_state(dfa_state)
                    decode_cache[dfa_state] = _format_nfa_set(decoded)
                except Exception:
                    decode_cache[dfa_state] = str(dfa_state)
            return decode_cache[dfa_state]

        groups = {}
        for p in particles:
            source = _format_source_path(p.lm_state)
            key = (source, p.dfa_state)
            groups.setdefault(key, []).append(p.log_weight)
        table_rows = []
        for (source, dfa_state), lws in groups.items():
            group_log_w = logsumexp(np.array(lws))
            posterior = np.exp(group_log_w - log_Z)
            is_q = dfa_state in q_states
            is_r = dfa_state in r_states
            role = ('Q+R' if is_q and is_r else
                    'Q' if is_q else
                    'R' if is_r else 'frontier')
            table_rows.append((source, dfa_state, role, len(lws),
                               group_log_w, posterior))
        table_rows.sort(key=lambda r: -r[5])
        rows = []
        for source, dfa_state, role, count, log_w, posterior in table_rows:
            rows.append([repr(source), decode_state(dfa_state), role,
                         str(count), f'{log_w:.2f}', f'{posterior:.4f}'])
        result = header + format_table(
            rows,
            headings=['Source prefix', 'DFA state', 'Role', 'Count',
                      'log w', 'p(x|y)'],
            column_styles={0: 'text-align:left'},
        )

        # Add collapsible Q/R FSA visualizations
        if can_decode and hasattr(self._peekaboo_state, 'build_qr_fsa'):
            particle_states = {p.dfa_state for p in particles}

            def sty_node(state):
                if state in particle_states:
                    return {'fillcolor': '#ADD8E6',
                            'style': 'filled,rounded'}
                return {}

            def fmt_node(state):
                return decode_state(state)

            # Only show Q/R for symbols relevant to current particles
            relevant_syms = set()
            for y, d in decomp.items():
                if d.quotient & particle_states or \
                   d.remainder & particle_states:
                    relevant_syms.add(y)

            MAX_QR = 10
            shown = 0
            for y in sorted(relevant_syms, key=repr):
                if shown >= MAX_QR:
                    rest = len(relevant_syms) - shown
                    result += (f'<details><summary>\u2026 and {rest} more '
                               f'Q/R sections</summary></details>')
                    break
                try:
                    q_fsa, r_fsa = self._peekaboo_state.build_qr_fsa(y)
                except Exception:
                    continue
                if not q_fsa.states and not r_fsa.states:
                    continue

                parts = []
                y_label = _html.escape(repr(y))
                if q_fsa.states:
                    try:
                        g = q_fsa.graphviz(fmt_node=fmt_node,
                                           sty_node=sty_node)
                        svg = g._repr_image_svg_xml()
                        parts.append(f'<b>Q({y_label})</b><br>{svg}')
                    except Exception:
                        pass
                if r_fsa.states:
                    try:
                        g = r_fsa.graphviz(fmt_node=fmt_node,
                                           sty_node=sty_node)
                        svg = g._repr_image_svg_xml()
                        parts.append(f'<b>R({y_label})</b><br>{svg}')
                    except Exception:
                        pass
                if parts:
                    content = '<br>'.join(parts)
                    result += (f'<details><summary>Q/R for y={y_label}'
                               f'</summary>{content}</details>')
                    shown += 1

        return result

    def __repr__(self):
        return (f'TransducedState(target={self._peekaboo_state.target!r},'
                f' K={len(self._particles)})')


# ---------------------------------------------------------------------------
# TransducedLM
# ---------------------------------------------------------------------------

class TransducedLM(LM):
    """Incremental transduced language model with particle-based inference.

    Computes the pushforward of an inner LM through an FST using a beam of
    K particles.  Each particle tracks a DFA state (powerset of precover NFA
    states) and an inner LM state.  At quotient states, exact marginalization
    sums over infinitely many source continuations.

    Per-step expansion uses beam search with beam width K through the
    decomposition's DFA.  Layers proceed by source-symbol depth: at each
    layer, all beam particles are classified (Q/R/preimage), scored, and
    non-Q particles are expanded.  Top K candidates form the next beam.
    Termination is by the monotone weight bound: since particle weights only
    decrease with depth, expansion stops when no beam particle can displace
    the current top-K carry-forward set.

    Args:
        inner_lm: LM with StateLM interface (has .initial(), state >> x,
            state.logp_next).
        fst: FST instance mapping source -> target.
        K: Beam width — number of particles maintained during both per-step
            expansion and carry-forward between target steps.  This is the
            primary knob controlling approximation quality.
        max_layers: Maximum beam search depth (source-symbol layers) per
            target step.  The primary termination criterion is the monotone
            weight bound (automatic); this is a safety valve for pathological
            FSTs where weight decay is very slow.
        eos: Outer EOS token.
        decomp_state_cls: Incremental decomposition class (default: PeekabooState).
        univ_cls: Universality precomputation class (default: FstUniversality).
        max_beam: Backward-compat alias for K.
        max_steps: Accepted for backward compat (unused).
        max_expansions: Accepted for backward compat (unused).
    """

    def __init__(self, inner_lm, fst, K=100, max_layers=100,
                 max_expansions=None, eos='<EOS>',
                 decomp_state_cls=None, univ_cls=None,
                 max_beam=None, max_steps=None):
        self.inner_lm = inner_lm
        self.fst = fst
        self.K = max_beam if max_beam is not None else K
        self.max_layers = max_layers
        # Backward-compat attribute (no longer controls beam search).
        self.max_expansions = max_steps or max_expansions or 1000
        self.eos = eos
        self._decomp_state_cls = decomp_state_cls or _DefaultDecompState
        self._univ_cls = univ_cls or _DefaultUniv
        self._univ = self._univ_cls(fst)

    def initial(self):
        """Return the initial TransducedState (empty target prefix)."""
        peekaboo = self._decomp_state_cls(
            self.fst, '', parent=None, univ=self._univ,
        )
        dfa = peekaboo.dfa
        start_states = list(dfa.start())
        inner_initial = self.inner_lm.initial()
        particles = [
            Particle(s, inner_initial, 0.0)
            for s in start_states
        ]
        return TransducedState(self, peekaboo, particles, 0.0)

    def __repr__(self):
        return f'TransducedLM(inner={self.inner_lm!r}, K={self.K})'
