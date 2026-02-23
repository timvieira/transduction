"""Character-level beam search over tokenized language models (SPM property).

Exploits the strict-prefix-monotone (SPM) property of BPE tokenization:
once a character position is processed, the LM cannot regress.  This
enables efficient memoized character-by-character inference with K-beam
pruning over alternative tokenization hypotheses.

The algorithm maintains a beam of TrieState bundles — each tracking an LM
state (after completed tokens) and a position in a token-character trie.
At each character step:

  1. **Extend**: Bundles at end-of-token advance the LM (expensive).
  2. **Actions**: Each bundle yields next-character probabilities via the trie.
  3. **Prune**: Top-K bundles are kept; low-weight bundles are dropped.
  4. **Advance**: The selected character narrows the beam.

Ported from ``tokenization/character_beam_trie.py`` and
``tokenization/trie.py``.  Uses the library's ``LM``/``LMState`` interface
for language model states and ``LogDistr`` for probability distributions.

The trie mass update uses a vectorized scatter-add via ``np.add.at`` over
a precomputed COO reachability index (leaf → all ancestors).  This avoids
the numba JIT dependency while staying O(total_path_lengths) — the same
asymptotic cost as the original numba loop.

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
from collections import defaultdict
from functools import cached_property
from typing import Any

from transduction.util import logsumexp, LogDistr
from transduction.lm.base import LM, LMState


# ---------------------------------------------------------------------------
# Token-character trie
# ---------------------------------------------------------------------------

class TokenCharacterTrie:
    """Trie mapping tokens to their byte decompositions.

    Precomputes a COO reachability index so that ``mass_sum`` can propagate
    token probabilities to all ancestor nodes in one ``np.add.at`` call.
    """

    def __init__(self, vocab: dict[Any, bytes], eos_token: Any,
                 new_eos: bytes) -> None:
        self._tokens = list(vocab.keys())
        self._token_to_idx: dict[Any, int] = {
            tok: i for i, tok in enumerate(self._tokens)
        }

        self.root = 0
        children: list[dict[Any, int]] = [{}]
        leaf2lm_token: dict[int, Any] = {}
        node2prefix: dict[int, bytes] = {0: b''}
        coo_nodes: list[int] = []
        coo_token_ids: list[int] = []

        for token, word in vocab.items():
            trie_word = new_eos if token == eos_token else word
            idx = self._token_to_idx[token]

            # Walk root → leaf, creating nodes and recording the path.
            curr = 0
            path = [0]
            for letter in trie_word:
                if letter not in children[curr]:
                    nid = len(children)
                    children[curr][letter] = nid
                    children.append({})
                    node2prefix[nid] = node2prefix[curr] + bytes([letter])
                curr = children[curr][letter]
                path.append(curr)

            # End-of-token sentinel leaf.
            last = len(children)
            children[curr][None] = last
            children.append({})
            node2prefix[last] = node2prefix[curr]
            leaf2lm_token[last] = token

            # COO: every node on the path (incl. sentinel) maps to this token.
            for node in path:
                coo_nodes.append(node)
                coo_token_ids.append(idx)
            coo_nodes.append(last)
            coo_token_ids.append(idx)

        self.children = children
        self.leaf2lm_token = leaf2lm_token
        self.node2prefix = node2prefix
        self._coo_nodes = np.array(coo_nodes, dtype=np.intp)
        self._coo_token_ids = np.array(coo_token_ids, dtype=np.intp)

    def mass_sum(self, logp_next: LogDistr) -> np.ndarray:
        """Bottom-up probability mass at each trie node via scatter-add."""
        p = np.exp([logp_next[tok] for tok in self._tokens])
        mass = np.zeros(len(self.children), dtype=np.float64)
        np.add.at(mass, self._coo_nodes, p[self._coo_token_ids])
        return mass



# ---------------------------------------------------------------------------
# Trie state (single partial tokenization hypothesis)
# ---------------------------------------------------------------------------

class TrieState:
    """A single partial-tokenization hypothesis within a beam.

    Tracks the LM state (after completed tokens), the current trie node
    (position within the current partial token), and a log-mass vector over
    trie nodes for fast probability lookups.
    """

    def __init__(self, lm_state: LMState, trie: TokenCharacterTrie,
                 node: int, mass: np.ndarray, weight: float) -> None:
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.mass = mass          # log-space mass vector over trie nodes
        self.weight = weight      # cumulative log-weight of this path

    def __repr__(self) -> str:
        return f'TrieState({self.weight:.2f}, {self.partial!r})'

    def __lshift__(self, a: Any) -> TrieState | None:
        """Advance by action ``a``: a byte value (int) or None (end-of-token)."""
        next_node = self.trie.children[self.node].get(a)
        if next_node is None:
            return None
        if a is None:
            # End-of-token: complete current token, advance LM state.
            lm_token = self.trie.leaf2lm_token[next_node]
            if lm_token == self.lm_state.eos:
                # LM EOS token completed; cannot advance past EOS — drop path.
                return None
            next_lm_state = self.lm_state >> lm_token
            next_mass = np.log(self.trie.mass_sum(next_lm_state.logp_next))
            return TrieState(
                lm_state=next_lm_state,
                trie=self.trie,
                mass=next_mass,
                node=self.trie.root,
                weight=self.weight + self.mass[next_node] - self.mass[self.node],
            )
        else:
            # Intra-token: move within trie, LM state unchanged.
            return TrieState(
                lm_state=self.lm_state,
                trie=self.trie,
                mass=self.mass,
                node=next_node,
                weight=self.weight + self.mass[next_node] - self.mass[self.node],
            )

    @classmethod
    def initial(cls, lm: LM, trie: TokenCharacterTrie) -> TrieState:
        lm_state = lm.initial()
        return cls(
            trie=trie,
            node=trie.root,
            lm_state=lm_state,
            mass=np.log(trie.mass_sum(lm_state.logp_next)),
            weight=0.0,
        )

    def actions(self) -> dict[Any, int]:
        return self.trie.children[self.node]

    @cached_property
    def logp_next(self) -> LogDistr:
        logZ = self.mass[self.node]
        return LogDistr({a: float(self.mass[i] - logZ)
                         for a, i in self.actions().items()})

    def has_EOT(self) -> bool:
        return None in self.actions()

    @property
    def partial(self):
        """Return the string prefix of the partially matched current token."""
        return self.trie.node2prefix[self.node]

    def __lt__(self, other: TrieState) -> bool:
        return self.weight < other.weight


# ---------------------------------------------------------------------------
# Bundle (internal beam state: collection of hypotheses at one position)
# ---------------------------------------------------------------------------

class _Bundle:
    """A bundle of TrieState hypotheses at the same character position."""

    def __init__(self, alg: CharacterBeam,
                 states: list[TrieState]) -> None:
        self.alg = alg
        self.states = states

    def __iter__(self):
        return iter(self.states)

    @classmethod
    def initial(cls, alg: CharacterBeam) -> _Bundle:
        return cls(alg, [alg._trie_init])

    def __len__(self) -> int:
        return len(self.states)

    def __lshift__(self, a: int) -> _Bundle:
        """Advance all bundles by character ``a`` (a byte value)."""
        return _Bundle(
            self.alg,
            [s for s in (bundle << a for bundle in self.extend
                         if a in bundle.actions()) if s is not None],
        )

    @cached_property
    def _logp_next(self) -> LogDistr:
        A = self.actions()
        Z = logsumexp([bs.weight for _, bs in A.items()])
        return LogDistr({k: float(bs.weight - Z) for k, bs in A.items()})

    @cached_property
    def extend(self) -> _Bundle:
        """Extend bundles at end-of-token by advancing the LM.

        Only extends bundles whose end-of-token probability exceeds the
        ``extend_threshold`` (if set).  This is the expensive step: each
        extension requires an LM forward pass.
        """
        batch = []
        for bundle in self:
            if bundle.has_EOT():
                if (
                    self.alg.extend_threshold is None
                    or np.exp(bundle.weight + bundle.logp_next[None]
                              - self.weight) >= self.alg.extend_threshold
                ):
                    batch.append(bundle)
        extended = [s for s in (b << None for b in batch) if s is not None]
        return _Bundle(self.alg, self.states + extended)

    def actions(self) -> dict[Any, _Bundle]:
        """Group next states by their character action."""
        A: dict[Any, list[TrieState]] = defaultdict(list)
        for bundle in self.extend:
            for a in bundle.logp_next:
                if a is not None:
                    next_state = bundle << a
                    if next_state is not None:
                        A[a].append(next_state)
        return {a: _Bundle(self.alg, next_states)
                for a, next_states in A.items()}

    @cached_property
    def weight(self) -> float:
        return logsumexp([b.weight for b in self])

    def prune(self) -> _Bundle:
        """Apply beam-width pruning."""
        S = sorted([x for x in self if x.weight > -np.inf],
                   key=lambda b: -b.weight)

        if self.alg.relative_score_threshold is not None:
            S = [
                bundle for bundle in S
                if np.exp(S[0].weight - bundle.weight) <= self.alg.relative_score_threshold
                or (self.alg.eot_immunity and not bundle.has_EOT())
            ]

        if not self.alg.eot_immunity:
            S = S[:self.alg.K]
        elif self.alg.K is not None:
            # Bundles without EOT are immune from pruning to avoid dead ends.
            tmp = []
            count = 0
            for bundle in S:
                if bundle.has_EOT():
                    count += 1
                tmp.append(bundle)
                if count >= self.alg.K:
                    break
            S = tmp

        return _Bundle(self.alg, S)


# ---------------------------------------------------------------------------
# CharacterBeamState (public LM state)
# ---------------------------------------------------------------------------

class CharacterBeamState(LMState):
    """Public LM state for CharacterBeam.

    Wraps an internal ``_Bundle`` of trie-state hypotheses and exposes
    the standard ``logp_next`` / ``>>`` interface.
    """

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
        next_candidates = self._candidates.prune() << byte_val
        return CharacterBeamState(
            self.cb, self.context + bytes([byte_val]),
            self.logp + logp_delta, next_candidates,
        )


# ---------------------------------------------------------------------------
# Character beam (main algorithm)
# ---------------------------------------------------------------------------

class CharacterBeam(LM):
    """Character-level beam search over a tokenized LM.

    Exploits the strict-prefix-monotone (SPM) property: character
    positions are processed left-to-right without backtracking, enabling
    efficient memoization and beam pruning.

    Args:
        lm: Language model (``LM`` instance).
        vocab: Mapping from LM tokens to their byte decompositions.
        K: Beam width (max bundles to keep after pruning).
        eos_token: Which token in ``vocab`` represents end-of-sequence.
            Defaults to ``lm.eos`` if not provided.
        eos: End-of-string marker (bytes).  The EOS token is replaced
            with this sequence in the trie so that EOS has a distinct
            character representation.
        relative_score_threshold: Optional ratio threshold for pruning.
        eot_immunity: If True, bundles without EOT are immune from pruning.
        extend_threshold: Minimum probability threshold for extending bundles.
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
            new_eos=self._eos_bytes,
        )

        self._trie_init = TrieState.initial(self.lm, self.trie)

        self.K = K
        self.relative_score_threshold = relative_score_threshold
        self.extend_threshold = extend_threshold
        self.eot_immunity = eot_immunity
        self.verbosity = verbosity

    def initial(self) -> CharacterBeamState:
        """Return the initial CharacterBeamState (empty context)."""
        return CharacterBeamState(self, b'', 0.0, _Bundle.initial(self))
