"""Test CharacterBeam vs FusedTransducedLM on subsampled BPE FST.

Both algorithms compute the same quantity: P(next_byte | bytes_so_far),
marginalizing over all BPE tokenizations.  CharacterBeam exploits the
strict-prefix-monotone (SPM) property for efficiency; FusedTransducedLM
is a general-purpose fused decomposition + LM search.

We verify they agree by wrapping a CharNgramLM (token-level) in a mock
TokenizedLLM so CharacterBeam can consume it, then comparing their
logp_next distributions at each byte position.
"""

import numpy as np
import pytest

from transduction.fst import FST
from transduction.fsa import EPSILON
from transduction.lm.ngram import CharNgramLM, CharNgramState
from transduction.lm.fused_transduced import FusedTransducedLM
from transduction.lm.character_beam import CharacterBeam
from transduction.lm.statelm import HfTokenizerVocab, LazyProb, flatten
from transduction.util import LogDistr


# ---------------------------------------------------------------------------
# Mock classes: wrap CharNgramLM in the TokenizedLLM/StateLM interface
# that CharacterBeam expects.
# ---------------------------------------------------------------------------

class MockStateLM:
    """Wraps a CharNgramState to present the StateLM interface."""

    def __init__(self, llm: 'MockTokenizedLLM', inner: CharNgramState,
                 logp: float, context: tuple) -> None:
        self._llm = llm
        self._inner = inner
        self.logp = logp
        self.context = context
        self.eos = llm.eos

    def __rshift__(self, token_bytes: bytes) -> 'MockStateLM':
        token_id = self._llm._encode[token_bytes]
        next_inner = self._inner >> token_id
        return MockStateLM(
            self._llm, next_inner,
            logp=self.logp + self._inner.logp_next[token_id],
            context=(self.context, token_bytes),
        )

    @property
    def logp_next(self) -> LazyProb:
        """Build a LazyProb indexed by token_id from the inner CharNgramState."""
        vocab_size = len(self._llm._decode)
        p = np.full(vocab_size, -np.inf)
        for token_id, logp_val in self._inner.logp_next.items():
            if isinstance(token_id, int):
                p[token_id] = logp_val
        return LazyProb(p, self._llm._encode, self._llm._decode)


class MockTokenizedLLM:
    """Wraps CharNgramLM + vocab tables to present the TokenizedLLM interface."""

    def __init__(self, decode: list, encode: dict, eos: bytes,
                 inner_lm: CharNgramLM) -> None:
        self._decode = decode
        self._encode = encode
        self.eos = eos
        self.V: set = {x for x in decode if x is not None}
        self._inner_lm = inner_lm

    def initial(self) -> MockStateLM:
        return MockStateLM(self, self._inner_lm.initial(), logp=0.0, context=())


# ---------------------------------------------------------------------------
# BPE FST builder (same as bench_vectorization.py)
# ---------------------------------------------------------------------------

def subsampled_bpe_fst(decode, token_ids, drop=frozenset()):
    m = FST()
    m.add_start(())
    for i in token_ids:
        x = decode[i]
        if x in drop:
            continue
        bx = tuple(x)
        for j in range(len(bx)):
            m.add_arc(bx[:j], EPSILON, bx[j], bx[:j+1])
        m.add_arc(bx, i, EPSILON, ())
    m.add_stop(())
    return m


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def setup():
    """Build the subsampled BPE FST, CharNgramLM, and both algorithms."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False,
                                              local_files_only=True)
    vocab = HfTokenizerVocab(tokenizer)

    drop = {x.encode() for x in tokenizer.all_special_tokens}

    # Training data
    train_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A stitch in time saves nine.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Actions speak louder than words.",
    ] * 3
    train_ids = [tokenizer.encode(s) for s in train_sentences]
    train_used = sorted(set(tid for seq in train_ids for tid in seq))

    # Subsample ~300 tokens (keep test fast)
    all_token_ids = sorted(
        i for i in range(len(vocab.decode)) if vocab.decode[i] not in drop
    )
    used = sorted(set(all_token_ids[:300]) | set(train_used))

    # Build FST
    fst = subsampled_bpe_fst(vocab.decode, used, drop)

    # Source alphabet for CharNgramLM = token IDs used by the FST
    source_alpha = fst.A - {EPSILON}
    inner_lm = CharNgramLM.train(train_ids, n=3, alpha=0.5,
                                 alphabet=source_alpha)

    # EOS bytes
    eos_bytes = tokenizer.eos_token.encode()

    # Build restricted decode/encode tables: only the used tokens.
    # Tokens not in `used` get None in _decode, so the trie excludes them.
    used_set_local = set(used)
    restricted_decode = [
        vocab.decode[i] if i in used_set_local else None
        for i in range(len(vocab.decode))
    ]
    restricted_encode = {
        vocab.decode[i]: i for i in used if vocab.decode[i] is not None
    }
    # EOS must be in the vocab for CharacterBeam
    eos_id = vocab.encode[eos_bytes]
    if eos_id not in used_set_local:
        restricted_decode[eos_id] = eos_bytes
        restricted_encode[eos_bytes] = eos_id

    # CharacterBeam
    mock_llm = MockTokenizedLLM(restricted_decode, restricted_encode,
                                eos_bytes, inner_lm)
    cb = CharacterBeam(mock_llm, K=100)

    # FusedTransducedLM
    fused = FusedTransducedLM(inner_lm, fst, max_steps=500, max_beam=50,
                              helper='rust')

    return cb, fused, fst, train_ids, set(used)


def test_cb_vs_fused(setup):
    """CharacterBeam and FusedTransducedLM agree on P(next_byte | context)."""
    cb, fused, fst, train_ids, used_set = setup

    # Find a byte context that is transducible under the subsampled FST
    target_bytes = None
    for seq in train_ids:
        if all(tid in used_set for tid in seq):
            try:
                target_bytes = list(fst.transduce(seq))
            except ValueError:
                continue
            if len(target_bytes) >= 6:
                target_bytes = target_bytes[:6]
                break
    assert target_bytes is not None, "Could not find transducible sequence"

    context = bytes(target_bytes)

    fused_state = fused.initial()
    max_diffs = []

    for i in range(len(context)):
        byte_prefix = context[:i]

        # CharacterBeam logp_next
        cb_logp = cb.logp_next(byte_prefix)

        # FusedTransducedLM logp_next
        fused_logp = fused_state.logp_next

        # Compare on common byte keys (int 0-255)
        common = set(cb_logp.keys()) & set(fused_logp.keys())
        # Exclude EOS-like keys
        common = {k for k in common if isinstance(k, int)}

        assert len(common) > 0, f"No common keys at position {i}"

        diffs = []
        for k in common:
            cb_val = float(cb_logp[k])
            fused_val = float(fused_logp[k])
            if cb_val > -10 or fused_val > -10:  # skip negligible entries
                diffs.append(abs(cb_val - fused_val))

        if diffs:
            max_diff = max(diffs)
            max_diffs.append(max_diff)
            print(f"  pos {i} byte={context[i]!r}: "
                  f"{len(common)} common keys, max|diff|={max_diff:.4f}")

        # Advance fused state
        y = context[i]
        if y in fused_logp:
            fused_state = fused_state >> y

    # With large beams, both should be close
    assert max_diffs, "No comparisons were made"
    worst = max(max_diffs)
    print(f"\n  Worst max|diff| across all positions: {worst:.4f}")
    assert worst < 0.15, f"Distributions differ too much: max|diff| = {worst:.4f}"


def test_cb_self_consistency(setup):
    """CharacterBeam logp_next is consistent with logprefix ratios."""
    cb, _, _, _, _ = setup

    context = b"The"
    logp = cb.logp_next(context)
    Z = cb.logprefix(context)

    # Check top-5 keys: logp_next[k] == logprefix(context + k) - Z
    for k in list(logp.top(5)):
        if k == cb.eos:
            continue
        have = logp[k]
        want = cb.logprefix(context + bytes([k])) - Z
        assert abs(have - want) < 1e-3, f"key={k!r}: {have:.6f} vs {want:.6f}"
