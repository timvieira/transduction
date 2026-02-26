"""Test CharacterBeam vs FusedTransducedLM and ReferenceTransducedLM.

Both algorithms compute the same quantity: P(next_byte | bytes_so_far),
marginalizing over all BPE tokenizations.  CharacterBeam exploits the
strict-prefix-monotone (SPM) property for efficiency; FusedTransducedLM
is a general-purpose fused decomposition + LM search; ReferenceTransducedLM
enumerates Q/R languages exactly (ground truth for finite-relation FSTs).

We verify they agree by passing a CharNgramLM (token-level) and a vocab
dict directly to CharacterBeam, then comparing their logp_next
distributions at each byte position.
"""

import numpy as np
import pytest

from transduction.fst import FST
from transduction.fsa import EPSILON
from transduction.lm.ngram import CharNgramLM
from transduction.lm.fused_transduced import FusedTransducedLM
from transduction.lm.reference_transduced import ReferenceTransducedLM
from transduction.lm.character_beam import CharacterBeam
from transduction.lm.huggingface_lm import HfTokenizerVocab
from transduction.util import LogDistr


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
    hf_vocab = HfTokenizerVocab(tokenizer)

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
        i for i in range(len(hf_vocab.decode)) if hf_vocab.decode[i] not in drop
    )
    used = sorted(set(all_token_ids[:300]) | set(train_used))

    # Build FST
    fst = subsampled_bpe_fst(hf_vocab.decode, used, drop)

    # Source alphabet for CharNgramLM = token IDs used by the FST
    source_alpha = fst.A - {EPSILON}
    inner_lm = CharNgramLM.train(train_ids, n=3, alpha=0.5,
                                 alphabet=source_alpha)

    # EOS handling
    eos_bytes = tokenizer.eos_token.encode()
    eos_id = hf_vocab.encode[eos_bytes]

    # Build vocab: token_id -> bytes (only tokens in `used`)
    cb_vocab: dict = {}
    for tid in used:
        word = hf_vocab.decode[tid]
        if word is not None:
            cb_vocab[tid] = word
    # EOS must be in the vocab for CharacterBeam
    if eos_id not in cb_vocab:
        cb_vocab[eos_id] = eos_bytes

    # CharacterBeam
    cb = CharacterBeam(inner_lm, cb_vocab, K=100, eos_token=eos_id)

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
                target_bytes = list(next(fst.transduce(seq)))
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
        cb_logp = cb(byte_prefix).logp_next

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
    state = cb(context)
    logp = state.logp_next
    Z = state.logp

    # Check top-5 keys: logp_next[k] == logp(context + k) - logp(context)
    for k in list(logp.top(5)):
        if k == cb._eos_bytes:
            continue
        have = logp[k]
        want = (state >> k).logp - Z
        assert abs(have - want) < 1e-3, f"key={k!r}: {have:.6f} vs {want:.6f}"


# ---------------------------------------------------------------------------
# CharacterBeam vs ReferenceTransducedLM (exact ground truth)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def small_setup():
    """Build a small BPE FST (~20 tokens) for exact reference comparison."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False,
                                              local_files_only=True)
    hf_vocab = HfTokenizerVocab(tokenizer)

    drop = {x.encode() for x in tokenizer.all_special_tokens}

    # Training data (small)
    train_sentences = [
        "The cat sat on the mat.",
        "The dog ran fast.",
    ] * 3
    train_ids = [tokenizer.encode(s) for s in train_sentences]
    train_used = sorted(set(tid for seq in train_ids for tid in seq))

    # Keep only the tokens that appear in training data (~20 tokens)
    used = sorted(
        tid for tid in train_used
        if hf_vocab.decode[tid] not in drop
    )

    # Build FST
    fst = subsampled_bpe_fst(hf_vocab.decode, used, drop)

    # Inner LM
    source_alpha = fst.A - {EPSILON}
    inner_lm = CharNgramLM.train(train_ids, n=2, alpha=0.5,
                                 alphabet=source_alpha)

    # EOS
    eos_bytes = tokenizer.eos_token.encode()
    eos_id = hf_vocab.encode[eos_bytes]

    # Vocab for CharacterBeam
    cb_vocab: dict = {}
    for tid in used:
        word = hf_vocab.decode[tid]
        if word is not None:
            cb_vocab[tid] = word
    if eos_id not in cb_vocab:
        cb_vocab[eos_id] = eos_bytes

    # CharacterBeam with large K (effectively exact for small vocab)
    cb = CharacterBeam(inner_lm, cb_vocab, K=500, eos_token=eos_id)

    # ReferenceTransducedLM (exact)
    ref = ReferenceTransducedLM(inner_lm, fst)

    return cb, ref, fst, train_ids, set(used)


def _renormalize_bytes(logp):
    """Extract byte keys and renormalize to a conditional distribution."""
    byte_keys = [k for k in logp.keys() if isinstance(k, int) and logp[k] > -50]
    if not byte_keys:
        return {}
    vals = np.array([logp[k] for k in byte_keys])
    Z = np.logaddexp.reduce(vals)
    return {k: logp[k] - Z for k in byte_keys}


def test_cb_vs_reference(small_setup):
    """CharacterBeam matches ReferenceTransducedLM (exact) on small BPE FST.

    ReferenceTransducedLM puts EOS mass under a separate '<EOS>' key, while
    CharacterBeam encodes EOS as a byte sequence.  We compare the conditional
    distribution over bytes (renormalized to exclude EOS) so the comparison
    is apples-to-apples.
    """
    cb, ref, fst, train_ids, used_set = small_setup

    # Find a transducible byte context
    target_bytes = None
    for seq in train_ids:
        if all(tid in used_set for tid in seq):
            try:
                target_bytes = list(next(fst.transduce(seq)))
            except ValueError:
                continue
            if len(target_bytes) >= 6:
                target_bytes = target_bytes[:6]
                break
    assert target_bytes is not None, "Could not find transducible sequence"

    context = bytes(target_bytes)

    ref_state = ref.initial()
    max_diffs = []

    for i in range(len(context)):
        byte_prefix = context[:i]

        # CharacterBeam logp_next (renormalized over bytes)
        cb_norm = _renormalize_bytes(cb(byte_prefix).logp_next)

        # ReferenceTransducedLM logp_next (renormalized over bytes)
        ref_norm = _renormalize_bytes(ref_state.logp_next)

        # Compare on common byte keys
        common = set(cb_norm.keys()) & set(ref_norm.keys())
        assert len(common) > 0, f"No common keys at position {i}"

        diffs = []
        for k in common:
            diffs.append(abs(cb_norm[k] - ref_norm[k]))

        if diffs:
            max_diff = max(diffs)
            max_diffs.append(max_diff)
            print(f"  pos {i} byte={context[i]!r}: "
                  f"{len(common)} common keys, max|diff|={max_diff:.6f}")

        # Advance reference state
        y = context[i]
        ref_logp = ref_state.logp_next
        if y in ref_logp:
            ref_state = ref_state >> y

    assert max_diffs, "No comparisons were made"
    worst = max(max_diffs)
    print(f"\n  Worst max|diff| across all positions: {worst:.6f}")
    # With large K and small vocab, CharacterBeam should be near-exact
    assert worst < 0.01, f"Distributions differ too much: max|diff| = {worst:.6f}"
