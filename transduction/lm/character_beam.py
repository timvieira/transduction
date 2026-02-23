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
``tokenization/trie.py``.  Uses ``transduction.lm.statelm`` for LM states
and vocabulary decoding.

The trie mass update uses a vectorized scatter-add via ``np.add.at`` over
a precomputed COO reachability index (leaf → all ancestors).  This avoids
the numba JIT dependency while staying O(total_path_lengths) — the same
asymptotic cost as the original numba loop.

Usage::

    from transduction.lm.statelm import load_model_by_name
    from transduction.lm.character_beam import CharacterBeam

    llm = load_model_by_name('gpt2')
    cb = CharacterBeam(llm, K=5)
    logp = cb.logp_next(b'Hello')  # log-prob distribution over next bytes
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from functools import cached_property
from typing import Any

from transduction.util import logsumexp, LogDistr
from transduction.lm.statelm import TokenizedLLM, StateLM, LazyProb, flatten


# ---------------------------------------------------------------------------
# Token-character trie
# ---------------------------------------------------------------------------

class TokenCharacterTrie:
    """Trie mapping BPE tokens to their character (byte) decompositions.

    Each token is a path through the trie; internal nodes correspond to
    partial tokens, leaves to complete tokens.  The ``mass_sum`` method
    propagates token probabilities bottom-up so that each node's mass
    equals the total probability of all tokens in its subtree.

    The bottom-up propagation uses a precomputed COO reachability index:
    for every (token, ancestor_node) pair we store the token-id and node
    index.  A single ``np.add.at`` scatter-add then computes all node
    masses in one vectorized pass — no numba, no scipy, no torch.
    """

    def __init__(self, words: list[bytes], encode: dict[bytes, int],
                 old_eos: bytes, new_eos: bytes) -> None:
        use_bytes = isinstance(words[0], bytes)
        if use_bytes:
            if not isinstance(old_eos, bytes):
                old_eos = old_eos.encode('utf-8')
            if not isinstance(new_eos, bytes):
                new_eos = new_eos.encode('utf-8')

        self.old_eos = old_eos
        self.old_eos_id = encode[old_eos]
        self.new_eos = new_eos

        root = 0
        children: list[dict[Any, int]] = [{}]
        word2leaf: dict[Any, int] = {}
        leaf2lm_token: dict[int, Any] = {}   # leaf -> original (un-renamed) token
        token_id_to_leaf: list[tuple[int, int]] = []

        for word in words:
            original_word = word
            if word == self.old_eos:
                word = self.new_eos

            curr = root
            for letter in word:
                if letter not in children[curr]:
                    children[curr][letter] = len(children)
                    children.append({})
                curr = children[curr][letter]

            children[curr][None] = last = len(children)
            children.append({})
            word2leaf[word] = last
            leaf2lm_token[last] = original_word
            token_id_to_leaf.append((encode[original_word], last))

        self.root = root
        self.children = children
        self.word2leaf = word2leaf
        self.leaf2word = dict(zip(word2leaf.values(), word2leaf.keys()))
        self.leaf2lm_token = leaf2lm_token
        self.token_id_to_leaf = token_id_to_leaf   # list; converted to np in _rename

        # Renumber nodes to contiguous topological order (memory locality)
        ordering_map: dict[int, int] = {}
        for i, x in enumerate(self._order_full(self.root)):
            ordering_map[x] = i
        self._rename(lambda x: ordering_map[x])

        # Build node -> prefix mapping on the renamed trie
        node2prefix: dict[int, Any] = {self.root: b'' if use_bytes else ''}
        for x in reversed(range(len(self.children))):
            for letter, y in self.children[x].items():
                if isinstance(letter, int):
                    letter = bytes([letter])
                if letter is None:
                    node2prefix[y] = node2prefix[x]
                else:
                    node2prefix[y] = node2prefix[x] + letter
        self.node2prefix = node2prefix

        # Build COO reachability index for vectorized mass_sum.
        # For each (token, ancestor_node) pair we record the token-id and
        # node index.  Walking from each leaf to root gives O(total_path_len)
        # entries — about vocab_size * avg_token_length.
        parent: dict[int, int] = {}
        for node in range(len(self.children)):
            for child in self.children[node].values():
                parent[child] = node

        coo_nodes: list[int] = []
        coo_token_ids: list[int] = []
        for k in range(self.token_id_to_leaf.shape[0]):
            token_id = int(self.token_id_to_leaf[k, 0])
            cur = int(self.token_id_to_leaf[k, 1])
            while True:
                coo_nodes.append(cur)
                coo_token_ids.append(token_id)
                if cur not in parent:
                    break
                cur = parent[cur]

        self._coo_nodes = np.array(coo_nodes, dtype=np.intp)
        self._coo_token_ids = np.array(coo_token_ids, dtype=np.intp)

    def _rename(self, f):
        N = len(self.children)
        new_children: list[dict[Any, int]] = [{} for _ in range(N)]
        for x in range(N):
            for letter, y in self.children[x].items():
                new_children[f(x)][letter] = f(y)

        self.root = f(self.root)
        self.children = new_children
        self.word2leaf = {w: f(x) for w, x in self.word2leaf.items()}
        self.leaf2word = dict(zip(self.word2leaf.values(), self.word2leaf.keys()))
        self.leaf2lm_token = {f(k): v for k, v in self.leaf2lm_token.items()}

        self.token_id_to_leaf = np.array(
            [(i, f(x)) for i, x in self.token_id_to_leaf], dtype=np.int32
        )

    def mass_sum(self, p_llm: LazyProb) -> np.ndarray:
        """Compute bottom-up probability mass at each trie node.

        Args:
            p_llm: Probability distribution over tokens (**not** log-probs).
                   Typically ``lm_state.logp_next.apply(np.exp)``.

        Returns:
            Array where ``mass[node]`` = sum of token probs in subtree.
        """
        mass = np.zeros(len(self.children), dtype=np.float64)
        np.add.at(mass, self._coo_nodes, p_llm._p[self._coo_token_ids])
        return mass

    def _order_full(self, node: int):
        """Topological ordering of ALL nodes beneath ``node``."""
        for a in self.children[node]:
            yield from self._order_full(self.children[node][a])
        yield node


# ---------------------------------------------------------------------------
# Trie state (single partial tokenization hypothesis)
# ---------------------------------------------------------------------------

class TrieState:
    """A single partial-tokenization hypothesis within a beam.

    Tracks the LM state (after completed tokens), the current trie node
    (position within the current partial token), and a log-mass vector over
    trie nodes for fast probability lookups.
    """

    def __init__(self, lm_state: StateLM, trie: TokenCharacterTrie,
                 node: int, mass: np.ndarray, weight: float,
                 parent: tuple[TrieState | None, Any]) -> None:
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.mass = mass          # log-space mass vector over trie nodes
        self.weight = weight      # cumulative log-weight of this path
        self.parent = parent

    @property
    def key(self):
        return self.lm_state.context

    def advance(self, actions):
        s = self
        for a in actions:
            s <<= a
            if s is None:
                return s
        return s

    def __repr__(self) -> str:
        return f'TrieState({self.weight:.2f}, {flatten(self.key)} -> {self.partial!r})'

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
            next_mass = np.log(self.trie.mass_sum(
                next_lm_state.logp_next.apply(np.exp)
            ))
            return TrieState(
                lm_state=next_lm_state,
                trie=self.trie,
                mass=next_mass,
                node=self.trie.root,
                parent=(self, a),
                weight=self.weight + self.mass[next_node] - self.mass[self.node],
            )
        else:
            # Intra-token: move within trie, LM state unchanged.
            return TrieState(
                lm_state=self.lm_state,
                trie=self.trie,
                mass=self.mass,
                node=next_node,
                parent=(self, a),
                weight=self.weight + self.mass[next_node] - self.mass[self.node],
            )

    @classmethod
    def initial(cls, lm: TokenizedLLM, trie: TokenCharacterTrie) -> TrieState:
        lm_state = lm.initial()
        return cls(
            trie=trie,
            node=trie.root,
            lm_state=lm_state,
            mass=np.log(trie.mass_sum(lm_state.logp_next.apply(np.exp))),
            weight=0.0,
            parent=(None, None),
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
# Beam state (bundle of hypotheses at one character position)
# ---------------------------------------------------------------------------

class CharacterBeamState:
    """A bundle of TrieState hypotheses at the same character position."""

    def __init__(self, alg: CharacterBeam,
                 states: list[TrieState]) -> None:
        self.alg = alg
        self.states = states

    def __iter__(self):
        return iter(self.states)

    @classmethod
    def initial(cls, alg: CharacterBeam) -> CharacterBeamState:
        return cls(alg, [alg.trie_init])

    def __len__(self) -> int:
        return len(self.states)

    def __lshift__(self, a: int) -> CharacterBeamState:
        """Advance all bundles by character ``a`` (a byte value)."""
        return CharacterBeamState(
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
    def extend(self) -> CharacterBeamState:
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
        return CharacterBeamState(
            self.alg,
            self.states + self._extend_batch(batch),
        )

    def actions(self) -> dict[Any, CharacterBeamState]:
        """Group next states by their character action."""
        A: dict[Any, list[TrieState]] = defaultdict(list)
        for bundle in self.extend:
            for a in bundle.logp_next:
                if a is not None:
                    next_state = bundle << a
                    if next_state is not None:
                        A[a].append(next_state)
        return {a: CharacterBeamState(self.alg, next_states)
                for a, next_states in A.items()}

    @cached_property
    def weight(self) -> float:
        return logsumexp([b.weight for b in self])

    def _extend_batch(self, batch: list[TrieState]) -> list[TrieState]:
        # TODO: batch LM state computation for efficiency
        return [s for s in (bundle << None for bundle in batch)
                if s is not None]

    def prune(self) -> CharacterBeamState:
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

        return CharacterBeamState(self.alg, S)


# ---------------------------------------------------------------------------
# Character beam (main algorithm)
# ---------------------------------------------------------------------------

class CharacterBeam:
    """Character-level beam search over a tokenized LM.

    Exploits the strict-prefix-monotone (SPM) property: character
    positions are processed left-to-right without backtracking, enabling
    efficient memoization and beam pruning.

    Args:
        llm: Tokenized language model (``TokenizedLLM`` instance).
        K: Beam width (max bundles to keep after pruning).
        eos: End-of-string marker (bytes).  The LM's EOS token is replaced
            with this sequence in the trie so that EOS has a distinct
            character representation.
        relative_score_threshold: Optional ratio threshold for pruning.
        eot_immunity: If True, bundles without EOT are immune from pruning.
        extend_threshold: Minimum probability threshold for extending bundles.
    """

    def __init__(self, llm: TokenizedLLM, K: int,
                 eos: bytes = '▪'.encode(),
                 relative_score_threshold: float | None = None,
                 eot_immunity: bool = False,
                 extend_threshold: float | None = None,
                 verbosity: int = 0) -> None:
        self.llm = llm
        self.eos = eos
        self.V: set[int] = {x for y in self.llm.V if y is not None for x in y}

        self.trie = TokenCharacterTrie(
            words=[w for w in llm._decode if w is not None],
            encode=llm._encode,
            old_eos=llm.eos,
            new_eos=self.eos,
        )

        self.trie_init = TrieState.initial(self.llm, self.trie)

        self.K = K
        self.relative_score_threshold = relative_score_threshold
        self.extend_threshold = extend_threshold
        self.eot_immunity = eot_immunity
        self.verbosity = verbosity

        self._beam_cache: dict[bytes, CharacterBeamState] = {}
        self._candidate_cache: dict[bytes, CharacterBeamState] = {}

    def logprefix(self, context: bytes) -> float:
        """Log-probability of the byte-string prefix ``context``."""
        return logsumexp([bundle.weight
                          for bundle in self.candidates(context)])

    def logp_next_seq(self, context: bytes, extension: bytes) -> float:
        """Log-probability of ``extension`` given ``context``."""
        return self.logprefix(context + extension) - self.logprefix(context)

    def beam(self, context: bytes) -> CharacterBeamState:
        """Return pruned candidates for ``context`` (cached)."""
        y = self._beam_cache.get(context)
        if y is not None:
            return y
        y = self._beam(context)
        self._beam_cache[context] = y
        return y

    def _beam(self, context: bytes) -> CharacterBeamState:
        if self.verbosity > 0:
            print(context)
        return self.candidates(context).prune()

    def candidates(self, context: bytes) -> CharacterBeamState:
        """Return unpruned candidates for ``context`` (cached)."""
        y = self._candidate_cache.get(context)
        if y is not None:
            return y
        y = self._candidates(context)
        self._candidate_cache[context] = y
        return y

    def _candidates(self, context: bytes) -> CharacterBeamState:
        if len(context) == 0:
            return CharacterBeamState.initial(self)
        else:
            return self.beam(context[:-1]) << context[-1]

    def logp_next(self, context: bytes) -> LogDistr:
        """Log-probability distribution over next bytes given ``context``."""
        return self.candidates(context)._logp_next

    def greedy(self, prompt: bytes, steps: int) -> bytes:
        """Greedy byte-by-byte generation starting from ``prompt``."""
        context = prompt
        for _ in range(steps):
            p = self.logp_next(context)
            if len(p) == 0:
                break
            x = p.argmax()
            context += bytes([x])
        return context

    def clear_cache(self) -> None:
        """Clear all memoization caches."""
        self._beam_cache.clear()
        self._candidate_cache.clear()
