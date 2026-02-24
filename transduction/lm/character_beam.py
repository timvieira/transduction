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

The trie mass update uses a vectorized log-space scatter via ``np.logaddexp.at``
over a precomputed COO reachability index (leaf → all ancestors).

Usage::

    from transduction.lm.character_beam import CharacterBeam

    lm = ...  # any LM instance
    vocab = {token: byte_decomposition for ...}
    cb = CharacterBeam(lm, vocab, K=5)
    state = cb.initial()
    logp = (state >> ord('H') >> ord('e')).logp_next  # next-byte log-probs after "He"
"""

from __future__ import annotations

import numpy as np
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

    def __init__(self, vocab: dict[Any, bytes], eos_token: Any,
                 eos_bytes: bytes) -> None:
        self._tokens = list(vocab.keys())
        token_to_idx = {tok: i for i, tok in enumerate(self._tokens)}

        self.root = 0
        children: list[dict[Any, int]] = [{}]
        leaf2lm_token: dict[int, Any] = {}
        node2prefix: dict[int, bytes] = {0: b''}
        coo_nodes: list[int] = []
        coo_token_ids: list[int] = []

        for token, spelling in vocab.items():
            trie_spelling = eos_bytes if token == eos_token else spelling
            idx = token_to_idx[token]

            # Walk root → leaf, creating nodes and recording the path.
            curr = 0
            path = [0]
            for byte_val in trie_spelling:
                if byte_val not in children[curr]:
                    nid = len(children)
                    children[curr][byte_val] = nid
                    children.append({})
                    node2prefix[nid] = node2prefix[curr] + bytes([byte_val])
                curr = children[curr][byte_val]
                path.append(curr)

            # End-of-token sentinel leaf.
            sentinel = len(children)
            children[curr][None] = sentinel
            children.append({})
            node2prefix[sentinel] = node2prefix[curr]
            leaf2lm_token[sentinel] = token

            # COO: every node on the path (incl. sentinel) maps to this token.
            for node in path:
                coo_nodes.append(node)
                coo_token_ids.append(idx)
            coo_nodes.append(sentinel)
            coo_token_ids.append(idx)

        self.children = children
        self.leaf2lm_token = leaf2lm_token
        self.node2prefix = node2prefix
        self._coo_nodes = np.array(coo_nodes, dtype=np.intp)
        self._coo_token_ids = np.array(coo_token_ids, dtype=np.intp)

    def log_mass_sum(self, logp_next: LogDistr) -> np.ndarray:
        """Bottom-up log-probability mass at each trie node via logaddexp scatter."""
        logp = np.array([logp_next[tok] for tok in self._tokens])
        log_mass = np.full(len(self.children), -np.inf, dtype=np.float64)
        np.logaddexp.at(log_mass, self._coo_nodes, logp[self._coo_token_ids])
        return log_mass


# ---------------------------------------------------------------------------
# Trie state (single partial tokenization hypothesis)
# ---------------------------------------------------------------------------

class TrieState:
    """A single hypothesis: an LM state + position within the token trie.

    ``log_mass[node]`` holds log P(subtree) at each trie node for the current
    LM state, enabling O(1) conditional probability lookups per character.
    """

    def __init__(self, lm_state: LMState, trie: TokenCharacterTrie,
                 node: int, log_mass: np.ndarray, weight: float) -> None:
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
        new_weight = self.weight + self.log_mass[next_node] - self.log_mass[self.node]
        if a is None:
            # End-of-token: complete current token, advance LM state.
            lm_token = self.trie.leaf2lm_token[next_node]
            if lm_token == self.lm_state.eos:
                return None
            next_lm = self.lm_state >> lm_token
            next_log_mass = self.trie.log_mass_sum(next_lm.logp_next)
            return TrieState(next_lm, self.trie, self.trie.root,
                             next_log_mass, new_weight)
        # Intra-token: move within trie, LM state unchanged.
        return TrieState(self.lm_state, self.trie, next_node,
                         self.log_mass, new_weight)

    @classmethod
    def initial(cls, lm: LM, trie: TokenCharacterTrie) -> TrieState:
        lm_state = lm.initial()
        return cls(lm_state, trie, trie.root,
                   trie.log_mass_sum(lm_state.logp_next), 0.0)

    @cached_property
    def logp_next(self) -> LogDistr:
        logZ = self.log_mass[self.node]
        return LogDistr({a: float(self.log_mass[i] - logZ)
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
        scores: LogVector = LogVector()
        for hyp in self.extend:
            node_mass = hyp.log_mass[hyp.node]
            for a, child_node in hyp.actions.items():
                if a is not None:
                    scores.logaddexp(a, hyp.weight + hyp.log_mass[child_node] - node_mass)
        return scores.normalize()

    @cached_property
    def extend(self) -> _Bundle:
        """Extend hypotheses at end-of-token by advancing the LM."""
        batch = [
            hyp for hyp in self
            if hyp.has_EOT() and (
                self.alg.extend_threshold is None
                or np.exp(hyp.weight + hyp.logp_next[None]
                          - self.weight) >= self.alg.extend_threshold
            )
        ]
        extended = [s for s in (h >> None for h in batch) if s is not None]
        return _Bundle(self.alg, self.states + extended)

    @cached_property
    def weight(self) -> float:
        return logsumexp([h.weight for h in self])

    def prune(self) -> _Bundle:
        """Apply beam-width pruning."""
        S = sorted([h for h in self if h.weight > -np.inf],
                   key=lambda h: -h.weight)

        if self.alg.relative_score_threshold is not None:
            S = [h for h in S
                 if np.exp(S[0].weight - h.weight)
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

    def __init__(self, cb: CharacterBeam, context: bytes, logp: float,
                 candidates: _Bundle) -> None:
        self.cb = cb
        self.context = context
        self.logp = logp
        self._candidates = candidates

    @cached_property
    def logp_next(self) -> LogDistr:
        return self._candidates._logp_next

    def __rshift__(self, byte_val: int) -> CharacterBeamState:
        logp_delta = self.logp_next[byte_val]
        if self.cb.verbosity > 0:
            print(self.context)
        next_candidates = self._candidates.extend.prune() >> byte_val
        return CharacterBeamState(
            self.cb, self.context + bytes([byte_val]),
            self.logp + logp_delta, next_candidates,
        )


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
        eos: End-of-string marker (bytes).  The EOS token is replaced
            with this sequence in the trie so that EOS has a distinct
            character representation.
        relative_score_threshold: Optional ratio threshold for pruning.
        eot_immunity: If True, hypotheses without EOT are immune from pruning.
        extend_threshold: Minimum probability threshold for extending.
    """

    eos = None

    def __init__(self, lm: LM, vocab: dict[Any, bytes], K: int,
                 eos_token: Any | None = None,
                 eos: bytes = '▪'.encode(),
                 relative_score_threshold: float | None = None,
                 eot_immunity: bool = False,
                 extend_threshold: float | None = None,
                 verbosity: int = 0) -> None:
        self.lm = lm
        self._eos_bytes = eos
        self.V: set[int] = {x for y in vocab.values() for x in y}

        self.trie = TokenCharacterTrie(
            vocab=vocab,
            eos_token=eos_token if eos_token is not None else lm.eos,
            eos_bytes=self._eos_bytes,
        )

        self._trie_init = TrieState.initial(self.lm, self.trie)

        self.K = K
        self.relative_score_threshold = relative_score_threshold
        self.extend_threshold = extend_threshold
        self.eot_immunity = eot_immunity
        self.verbosity = verbosity

    def initial(self) -> CharacterBeamState:
        return CharacterBeamState(self, b'', 0.0, _Bundle.initial(self))
