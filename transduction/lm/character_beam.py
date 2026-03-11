"""Character-level beam search over tokenized language models (SPM property).

Exploits the strict-prefix-monotone (SPM) property of BPE tokenization:
once a character position is processed, the LM cannot regress.  This
enables efficient memoized character-by-character inference with K-beam
pruning over alternative tokenization hypotheses.

The algorithm maintains a beam of TrieState hypotheses — each tracking an
LM state (after completed tokens) and a position in a token-character trie.
At each character step:

  1. **Extend**: Hypotheses at end-of-token advance the LM (expensive).
  2. **Actions**: Each hypothesis yields next-character probabilities via the trie.
  3. **Prune**: Top-K hypotheses are kept; low-weight ones are dropped.
  4. **Advance**: The selected character narrows the beam.

Uses the library's ``LM``/``LMState`` interface for language model states
and ``LogDistr`` for probability distributions.

The trie mass update uses a vectorized log-space sparse matvec via
a precomputed CSR reachability matrix (leaf → all ancestors).

Usage::

    from transduction.lm.character_beam import CharacterBeam

    lm = ...  # any LM instance
    vocab = {token: byte_decomposition for ...}
    cb = CharacterBeam(lm, vocab, K=5)
    state = cb.initial()
    logp = (state >> ord('H') >> ord('e')).logp_next  # next-byte log-probs after "He"
"""

from __future__ import annotations

import torch
from functools import cached_property
from typing import Any

from transduction.util import logsumexp, LogDistr, LogVector
from transduction.lm.base import LM, LMState


# ---------------------------------------------------------------------------
# Token-character trie
# ---------------------------------------------------------------------------

class TokenCharacterTrie:
    """Trie mapping tokens to their byte decompositions.

    Precomputes a COO reachability index so that ``log_mass_sum`` can propagate
    token log-probabilities to all ancestor nodes in one ``np.logaddexp.at`` call.
    """

    def __init__(self, vocab: dict[Any, bytes], eos_token: Any) -> None:
        self.eos_token = eos_token

        # Exclude EOS from the trie — EOS mass is computed separately
        # at root hypotheses to avoid emitting spurious EOS bytes.
        self._tokens = [tok for tok in vocab if tok != eos_token]
        token_to_idx = {tok: i for i, tok in enumerate(self._tokens)}

        self.root = 0
        children: list[dict[Any, int]] = [{}]
        leaf2lm_token: dict[int, Any] = {}
        node2prefix: dict[int, bytes] = {0: b''}
        coo_nodes: list[int] = []
        coo_token_ids: list[int] = []

        for token in self._tokens:
            spelling = vocab[token]
            idx = token_to_idx[token]

            # Walk root → leaf, creating nodes and recording the path.
            curr = 0
            path = [0]
            for byte_val in spelling:
                if byte_val not in children[curr]:
                    nid = len(children)
                    children[curr][byte_val] = nid
                    children.append({})
                    node2prefix[nid] = node2prefix[curr] + bytes([byte_val])
                curr = children[curr][byte_val]
                path.append(curr)

            # End-of-token sentinel leaf.  If another token already created
            # a sentinel at this node (same spelling), reuse it and append.
            if None in children[curr]:
                sentinel = children[curr][None]
                leaf2lm_token[sentinel].append(token)
            else:
                sentinel = len(children)
                children[curr][None] = sentinel
                children.append({})
                node2prefix[sentinel] = node2prefix[curr]
                leaf2lm_token[sentinel] = [token]

            # COO: every node on the path (incl. sentinel) maps to this token.
            for node in path:
                coo_nodes.append(node)
                coo_token_ids.append(idx)
            coo_nodes.append(sentinel)
            coo_token_ids.append(idx)

        self.children = children
        self.leaf2lm_token = leaf2lm_token
        self.node2prefix = node2prefix

        # Pre-compute integer token ID tensor for fast indexing
        # when logp_next is a TokenLogProbs (backed by a tensor).
        try:
            self._token_ids = torch.tensor(self._tokens, dtype=torch.long)
        except (TypeError, ValueError):
            self._token_ids = None

        # Cache for index tensors keyed by id(key_to_idx) from tensor-backed LogDistr.
        self._key_idx_cache: dict[int, torch.Tensor] = {}

        # Sparse reachability matrix for fast log_mass_sum via matvec.
        # CSR float32 for optimized CPU sparse BLAS (MKL/OpenBLAS); ~15x faster
        # than COO float64.  Precision loss is <1e-6, negligible for beam search.
        n_nodes = len(children)
        n_tokens = len(self._tokens)
        indices = torch.tensor([coo_nodes, coo_token_ids], dtype=torch.long)
        values = torch.ones(len(coo_nodes), dtype=torch.float32)
        self._reach = torch.sparse_coo_tensor(
            indices, values, (n_nodes, n_tokens),
        ).coalesce().to_sparse_csr()

        # Pre-compute root child indices for vectorized _logp_next.
        # _root_byte_actions: list of (byte_val, child_node) for root, sorted.
        # _root_child_nodes: tensor of child node indices, shape (num_root_children,).
        root_actions = [(a, c) for a, c in children[self.root].items() if a is not None]
        root_actions.sort(key=lambda x: x[0])
        self._root_bytes = [a for a, _ in root_actions]
        self._root_child_nodes = torch.tensor([c for _, c in root_actions], dtype=torch.long)

    def _logp_row(self, logp_next: LogDistr) -> torch.Tensor:
        """Extract a (n_tokens,) logp vector from logp_next, using fastest available path."""
        p = getattr(logp_next, '_p', None)
        if p is not None:
            # Path 1: tensor-backed LogDistr with _key_to_idx (e.g., CharNgramLM)
            key_to_idx = getattr(logp_next, '_key_to_idx', None)
            if key_to_idx is not None:
                cache_key = id(key_to_idx)
                indices = self._key_idx_cache.get(cache_key)
                if indices is None:
                    try:
                        indices = torch.tensor(
                            [key_to_idx[s] for s in self._tokens],
                            dtype=torch.long)
                    except KeyError:
                        indices = False  # type: ignore[assignment]
                    self._key_idx_cache[cache_key] = indices
                if indices is not False:
                    return p[indices].double()
            elif self._token_ids is not None:
                # TokenLogProbs — _p is indexed by raw integer token IDs
                return p[self._token_ids].double()
        return torch.tensor([logp_next[tok] for tok in self._tokens],
                            dtype=torch.float64)

    def log_mass_sum(self, logp_next: LogDistr) -> torch.Tensor:
        """Bottom-up log-probability mass at each trie node via sparse matvec."""
        logp = self._logp_row(logp_next).float()
        logp_max = logp.max()
        prob = torch.exp(logp - logp_max)
        mass = torch.mv(self._reach, prob)
        log_mass = torch.log(mass) + logp_max
        return log_mass

    def log_mass_sum_from_logits(self, logp_batch: torch.Tensor) -> list[torch.Tensor]:
        """Compute trie mass vectors from (N, vocab_size) log-prob tensor.

        Fuses token selection + sparse matmul. No LogDistr construction.
        """
        logp = logp_batch[:, self._token_ids].float()   # (N, n_tokens)
        logp_max = logp.max(dim=1, keepdim=True).values
        prob = torch.exp(logp - logp_max)
        mass = torch.sparse.mm(self._reach, prob.T)     # (n_nodes, N)
        log_mass = torch.log(mass) + logp_max.T          # (n_nodes, N)
        return [log_mass[:, i] for i in range(logp_batch.shape[0])]

    def log_mass_sum_batch(self, logp_nexts: list[LogDistr]) -> list:
        """Batched log_mass_sum: single sparse matmul for multiple LM states."""
        n = len(logp_nexts)
        if n == 0:
            return []
        if n == 1:
            return [self.log_mass_sum(logp_nexts[0])]

        # Deduplicate by identity
        unique: dict[int, tuple[int, torch.Tensor]] = {}
        idx_map: list[int] = []
        for logp_next in logp_nexts:
            key = id(logp_next)
            if key not in unique:
                unique[key] = (len(unique), self._logp_row(logp_next))
            idx_map.append(unique[key][0])

        unique_rows = [row for _, row in sorted(unique.values())]
        logp_mat = torch.stack(unique_rows).float()

        logp_max = logp_mat.max(dim=1, keepdim=True).values
        prob_mat = torch.exp(logp_mat - logp_max)
        mass_mat = torch.sparse.mm(self._reach, prob_mat.T)
        log_mass_mat = torch.log(mass_mat) + logp_max.T

        return [log_mass_mat[:, idx_map[i]] for i in range(n)]


# ---------------------------------------------------------------------------
# Trie state (single partial tokenization hypothesis)
# ---------------------------------------------------------------------------

class TrieState:
    """A single hypothesis: an LM state + position within the token trie.

    ``log_mass[node]`` holds log P(subtree) at each trie node for the current
    LM state, enabling O(1) conditional probability lookups per character.
    """

    def __init__(self, lm_state: LMState, trie: TokenCharacterTrie,
                 node: int, log_mass: torch.Tensor, weight: float) -> None:
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.log_mass = log_mass
        self.weight = weight

    def __repr__(self) -> str:
        return f'TrieState({self.weight:.2f}, {self.partial!r})'

    @property
    def actions(self) -> dict[Any, int]:
        return self.trie.children[self.node]

    def __rshift__(self, a: Any) -> TrieState | None:
        """Advance by action ``a``: a byte value (int) or None (end-of-token)."""
        next_node = self.trie.children[self.node].get(a)
        if next_node is None:
            return None
        new_weight = float(self.weight + self.log_mass[next_node] - self.log_mass[self.node])
        if a is None:
            # End-of-token: complete current token, advance LM state.
            tokens = self.trie.leaf2lm_token[next_node]
            if len(tokens) == 1:
                lm_token = tokens[0]
            else:
                # Multiple tokens share this spelling — pick the most probable
                # and adjust weight proportionally.  The full fan-out is handled
                # by _prepare_eot; this path is for __rshift__ which returns a
                # single TrieState.
                logp = self.lm_state.logp_next
                best_idx = max(range(len(tokens)), key=lambda i: logp[tokens[i]])
                lm_token = tokens[best_idx]
                new_weight += logp[lm_token] - logsumexp([logp[t] for t in tokens])
            next_lm = self.lm_state >> lm_token
            next_log_mass = self.trie.log_mass_sum(next_lm.logp_next)
            return TrieState(next_lm, self.trie, self.trie.root,
                             next_log_mass, new_weight)
        # Intra-token: move within trie, LM state unchanged.
        return TrieState(self.lm_state, self.trie, next_node,
                         self.log_mass, new_weight)

    def _prepare_eot(self) -> list[tuple[LMState, float]]:
        """Prepare EOT transitions: create child LM states without triggering forward passes.

        Returns list of (child_lm_state, new_weight), one per token at the leaf.
        Empty list if no EOT action.

        When multiple tokens share a trie leaf (duplicate spellings), the
        sentinel mass is the sum of their individual probabilities.  We split
        ``w_total`` proportionally by each token's inner-LM probability so
        that the total mass is conserved.
        """
        next_node = self.trie.children[self.node].get(None)
        if next_node is None:
            return []
        w_total = float(self.weight + self.log_mass[next_node] - self.log_mass[self.node])
        tokens = self.trie.leaf2lm_token[next_node]
        if len(tokens) == 1:
            child_lm = self.lm_state >> tokens[0]
            return [(child_lm, w_total)]
        # Multiple tokens share this spelling — split proportionally
        logp = self.lm_state.logp_next
        token_logps = [logp[tok] for tok in tokens]
        total_lp = logsumexp(token_logps)
        results = []
        for tok, tok_lp in zip(tokens, token_logps):
            child_lm = self.lm_state >> tok
            results.append((child_lm, w_total + tok_lp - total_lp))
        return results

    def _complete_eot(self, child_lm: LMState, weight: float) -> TrieState:
        """Complete EOT transition using a (possibly prefetched) child LM state.

        Calls log_mass_sum(child_lm.logp_next) which consumes the now-cached logp_next.
        """
        next_log_mass = self.trie.log_mass_sum(child_lm.logp_next)
        return TrieState(child_lm, self.trie, self.trie.root,
                         next_log_mass, weight)

    @classmethod
    def initial(cls, lm: LM, trie: TokenCharacterTrie) -> TrieState:
        lm_state = lm.initial()
        return cls(lm_state, trie, trie.root,
                   trie.log_mass_sum(lm_state.logp_next), 0.0)

    @cached_property
    def logp_next(self) -> LogDistr:
        logZ = float(self.log_mass[self.node])
        return LogDistr({a: float(self.log_mass[i]) - logZ
                         for a, i in self.actions.items()})

    def has_EOT(self) -> bool:
        return None in self.actions

    @property
    def partial(self) -> bytes:
        """Byte prefix of the partially matched current token."""
        return self.trie.node2prefix[self.node]

    def __lt__(self, other: TrieState) -> bool:
        return self.weight < other.weight


# ---------------------------------------------------------------------------
# Bundle (internal: collection of hypotheses at one character position)
# ---------------------------------------------------------------------------

class _Bundle:
    """A set of TrieState hypotheses at the same character position."""

    def __init__(self, alg: CharacterBeam, states: list[TrieState]) -> None:
        self.alg = alg
        self.states = states

    def __iter__(self):
        return iter(self.states)

    def __len__(self) -> int:
        return len(self.states)

    @classmethod
    def initial(cls, alg: CharacterBeam) -> _Bundle:
        return cls(alg, [alg._trie_init])

    def __rshift__(self, a: int) -> _Bundle:
        """Advance all hypotheses by byte ``a``."""
        return _Bundle(self.alg, [
            s for hyp in self.states if a in hyp.actions
            for s in [hyp >> a] if s is not None
        ])

    @cached_property
    def _logp_next(self) -> LogDistr:
        trie = self.alg.trie
        eos_token = trie.eos_token
        root = trie.root

        # Separate root hypotheses (vectorizable) from non-root (Python loop).
        root_hyps: list = []
        nonroot_hyps: list = []
        for hyp in self.extend:
            if hyp.node == root:
                root_hyps.append(hyp)
            else:
                nonroot_hyps.append(hyp)

        scores: LogVector = LogVector()

        # Vectorized path for root hypotheses (dominate after extend).
        if root_hyps:
            root_child_nodes = trie._root_child_nodes
            root_bytes = trie._root_bytes
            n = len(root_hyps)

            # Stack log_mass tensors and weights.
            log_masses = torch.stack([h.log_mass for h in root_hyps])  # (n, num_nodes)
            weights = torch.tensor([h.weight for h in root_hyps], dtype=torch.float64)

            # log_mass at root for each hyp.
            root_masses = log_masses[:, root]  # (n,)

            # log_mass at each root child for each hyp.
            child_masses = log_masses[:, root_child_nodes]  # (n, num_children)

            # weighted[i, j] = weight[i] + child_mass[i, j] - root_mass[i]
            weighted = weights.unsqueeze(1) + child_masses - root_masses.unsqueeze(1)

            # logsumexp across hypotheses for each byte.
            byte_scores = torch.logsumexp(weighted, dim=0)  # (num_children,)

            for j, a in enumerate(root_bytes):
                scores.logaddexp(a, float(byte_scores[j]))

            # EOS for root hypotheses.
            eos_lps = torch.tensor([h.lm_state.logp_next[eos_token] for h in root_hyps],
                                   dtype=torch.float64)
            eos_weighted = weights + eos_lps
            mask = eos_lps > float('-inf')
            if mask.any():
                scores.logaddexp(None, float(torch.logsumexp(eos_weighted[mask], dim=0)))

        # Scalar path for non-root hypotheses.
        for hyp in nonroot_hyps:
            node_mass = float(hyp.log_mass[hyp.node])
            for a, child_node in hyp.actions.items():
                if a is not None:
                    scores.logaddexp(a, hyp.weight + float(hyp.log_mass[child_node]) - node_mass)

        return scores.normalize()

    @cached_property
    def extend(self) -> _Bundle:
        """Extend hypotheses at end-of-token by advancing the LM.

        Uses a 3-phase prepare/prefetch/complete pattern to enable batched
        forward passes when the LM supports prefetch (e.g. HuggingFaceLM).
        """
        batch = [
            hyp for hyp in self
            if hyp.has_EOT() and (
                self.alg.extend_threshold is None
                or torch.exp(torch.tensor(hyp.weight + hyp.logp_next[None]
                          - self.weight)).item() >= self.alg.extend_threshold
            )
        ]

        # Limit extensions to top-N by weight
        if self.alg.max_extend is not None and len(batch) > self.alg.max_extend:
            batch = sorted(batch, key=lambda h: -h.weight)[:self.alg.max_extend]

        # Phase 1: Prepare — create child LM states without forward passes.
        prepared: list[tuple[TrieState, LMState, float]] = []
        child_states: list[LMState] = []
        for hyp in batch:
            for child_lm, weight in hyp._prepare_eot():
                prepared.append((hyp, child_lm, weight))
                child_states.append(child_lm)

        # Phase 2: Prefetch — batch forward passes for all children.
        if child_states:
            self.alg.lm.prefetch(child_states)

        # Phase 3: Complete — batch log_mass_sum across all children.
        if prepared:
            trie = prepared[0][0].trie
            batch_masses = trie.log_mass_sum_batch(
                [child_lm.logp_next for _, child_lm, _ in prepared])
            extended = [
                TrieState(child_lm, trie, trie.root, batch_masses[i], weight)
                for i, (hyp, child_lm, weight) in enumerate(prepared)
            ]
        else:
            extended = []

        return _Bundle(self.alg, self.states + extended)

    @cached_property
    def weight(self) -> float:
        return logsumexp([h.weight for h in self])

    def prune(self) -> _Bundle:
        """Apply beam-width pruning."""
        S = sorted([h for h in self if h.weight > float('-inf')],
                   key=lambda h: -h.weight)

        if self.alg.relative_score_threshold is not None:
            from math import exp
            S = [h for h in S
                 if exp(S[0].weight - h.weight)
                    <= self.alg.relative_score_threshold
                 or (self.alg.eot_immunity and not h.has_EOT())]

        if not self.alg.eot_immunity:
            S = S[:self.alg.K]
        elif self.alg.K is not None:
            # Hypotheses without EOT are immune from pruning.
            kept = []
            eot_count = 0
            for h in S:
                if h.has_EOT():
                    eot_count += 1
                kept.append(h)
                if eot_count >= self.alg.K:
                    break
            S = kept

        return _Bundle(self.alg, S)


# ---------------------------------------------------------------------------
# CharacterBeamState (public LM state)
# ---------------------------------------------------------------------------

class CharacterBeamState(LMState):
    """LM state for CharacterBeam: wraps a ``_Bundle`` of trie hypotheses."""

    eos = None

    def __init__(self, cb: CharacterBeam, context: bytes, logprefix: float,
                 candidates: _Bundle) -> None:
        self.cb = cb
        self.context = context
        self.logprefix = logprefix
        self._candidates = candidates

    def __repr__(self) -> str:
        n = len(self._candidates.states)
        ctx = self.context[-40:] if len(self.context) > 40 else self.context
        return (f'CharacterBeamState(context={ctx!r}, logprefix={self.logprefix:.2f}, '
                f'beam={n})')

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        from math import exp
        states = self._candidates.states
        n = len(states)
        ctx = repr(self.context)
        if len(ctx) > 80:
            ctx = '...' + ctx[-77:]

        rows = ['<tr><th>#</th><th>weight</th><th>rel. prob</th>'
                '<th>partial</th><th>trie node</th><th>has EOT</th></tr>']
        top_w = states[0].weight if states else 0.0
        for i, h in enumerate(sorted(states, key=lambda h: -h.weight)):
            rel = exp(h.weight - top_w) if h.weight > float('-inf') else 0.0
            partial = repr(h.partial) if h.partial else ''
            eot = 'yes' if h.has_EOT() else ''
            rows.append(
                f'<tr><td>{i}</td><td>{h.weight:.3f}</td>'
                f'<td>{rel:.4f}</td>'
                f'<td><code>{partial!r}</code></td>'
                f'<td>{h.node}</td><td>{eot}</td></tr>')

        return (
            f'<div style="font-family:monospace; font-size:13px">'
            f'<b>CharacterBeamState</b> &mdash; '
            f'context=<code>{ctx!r}</code>, '
            f'logprefix={self.logprefix:.2f}, '
            f'K={self.cb.K}, beam={n}<br>'
            f'<table style="border-collapse:collapse; margin-top:4px">'
            + '\n'.join(rows) +
            '</table></div>'
        )

    @property
    def context_key(self):
        return self.context

    @cached_property
    def logp_next(self) -> LogDistr:
        return self._candidates._logp_next

    def __rshift__(self, byte_val: int) -> CharacterBeamState:
        logp_delta = self.logp_next[byte_val]
        if self.cb.verbosity > 0:
            print(self.context)
        pre = self._candidates
        extended = pre.extend
        pruned = extended.prune()
        advanced = pruned >> byte_val
        next_state = CharacterBeamState(
            self.cb, self.context + bytes([byte_val]),
            self.logprefix + logp_delta, advanced,
        )
        next_state._step_info = {
            'pre_extend': len(pre.states),
            'post_extend': len(extended.states),
            'post_prune': len(pruned.states),
            'post_advance': len(advanced.states),
        }
        self.cb.trace.append(next_state)
        return next_state


# ---------------------------------------------------------------------------
# CharacterBeam (main entry point)
# ---------------------------------------------------------------------------

class CharacterBeam(LM):
    """Character-level beam search over a tokenized LM.

    Args:
        lm: Language model (``LM`` instance).
        vocab: Mapping from LM tokens to their byte decompositions.
        K: Beam width (max hypotheses to keep after pruning).
        eos_token: Which token in ``vocab`` represents end-of-sequence.
            Defaults to ``lm.eos`` if not provided.
        relative_score_threshold: Optional ratio threshold for pruning.
        eot_immunity: If True, hypotheses without EOT are immune from pruning.
        extend_threshold: Minimum probability threshold for extending.
        max_extend: Maximum number of EOT hypotheses to extend per step.
            When set, only the top ``max_extend`` hypotheses (by weight) are
            extended; the rest remain at their current trie position.
    """

    eos = None

    def __init__(self, lm: LM, vocab: dict[Any, bytes], K: int,
                 eos_token: Any | None = None,
                 relative_score_threshold: float | None = None,
                 eot_immunity: bool = False,
                 extend_threshold: float | None = None,
                 max_extend: int | None = None,
                 verbosity: int = 0) -> None:
        self.lm = lm
        self.V: set[int] = {x for y in vocab.values() for x in y}

        self.trie = TokenCharacterTrie(
            vocab=vocab,
            eos_token=eos_token if eos_token is not None else lm.eos,
        )

        self._trie_init = TrieState.initial(self.lm, self.trie)

        self.K = K
        self.relative_score_threshold = relative_score_threshold
        self.extend_threshold = extend_threshold
        self.max_extend = max_extend
        self.eot_immunity = eot_immunity
        self.verbosity = verbosity
        self.trace: list[CharacterBeamState] = []

    def initial(self) -> CharacterBeamState:
        self.trace = []
        state = CharacterBeamState(self, b'', 0.0, _Bundle.initial(self))
        self.trace.append(state)
        return state

    def visualize(self) -> None:
        """Display interactive beam search visualization of the current trace."""
        from transduction.lm.beam_viz import visualize_beam_trace
        visualize_beam_trace(self.trace)
