import pytest
import numpy as np
from collections import defaultdict

genparse = pytest.importorskip('genparse')
from genparse import EarleyLM, EOS

from transduction import examples
from transduction.lm.base import LMState
from transduction.enumeration import (
    prioritized_enumeration,
    importance_sampling,
    crude_importance_sampling,
)
from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp


GRAMMAR = """
.5: S -> a S
.4: S -> b S
.1: S -> c
"""


class StatefulLM(LMState):
    """Wraps a stateless EarleyLM into the stateful interface expected by
    the enumeration classes (`.eos`, `.logp_next`, `>> token`)."""

    def __init__(self, lm, eos, context=()):
        self._lm = lm
        self._eos = eos
        self._context = context

    @property
    def eos(self):
        return self._eos

    @property
    def logp_next(self):
        p = self._lm.p_next(self._context)
        return defaultdict(lambda: float('-inf'), {
            k: np.log(v) if v > 0 else float('-inf')
            for k, v in p.items()
        })

    def __rshift__(self, token):
        return StatefulLM(self._lm, self._eos, self._context + (token,))

    def __repr__(self):
        return f'StatefulLM({list(self._context)})'


@pytest.fixture
def lm():
    return StatefulLM(EarleyLM.from_string(GRAMMAR), eos=EOS)


@pytest.fixture
def fst():
    return examples.replace([('a', 'a'), ('b', 'b'), ('c', 'c')])


# --- prioritized_enumeration tests ---

class TestPrioritizedEnumeration:

    def test_abc(self, lm, fst):
        pe = prioritized_enumeration(lm, fst, 'abc', max_steps=50)
        assert len(pe.quotient_terms) == 1
        weight = pe.quotient_terms[0].weight
        expected = np.log(0.5 * 0.4 * 0.1)
        assert weight == pytest.approx(expected, abs=1e-6)

    def test_aa(self, lm, fst):
        pe = prioritized_enumeration(lm, fst, 'aa', max_steps=100)
        assert len(pe.quotient_terms) == 1
        weight = pe.quotient_terms[0].weight
        expected = np.log(0.5 * 0.5)
        assert weight == pytest.approx(expected, abs=1e-6)

    def test_empty_target(self, lm, fst):
        pe = prioritized_enumeration(lm, fst, '', max_steps=50)
        # empty target: the quotient is immediately reachable
        assert len(pe.quotient_terms) >= 1

    def test_with_nonrecursive_decompose(self, lm, fst):
        pe = prioritized_enumeration(lm, fst, 'abc', max_steps=50,
                                     decompose=NonrecursiveDFADecomp)
        assert len(pe.quotient_terms) == 1
        weight = pe.quotient_terms[0].weight
        expected = np.log(0.5 * 0.4 * 0.1)
        assert weight == pytest.approx(expected, abs=1e-6)


# --- importance_sampling tests ---

class TestImportanceSampling:

    def test_abc(self, lm, fst):
        sampler = importance_sampling(lm, fst, 'abc')
        item = sampler.sample()
        assert item is not None
        expected = np.log(0.5 * 0.4 * 0.1)
        assert item.weight == pytest.approx(expected, abs=1e-6)

    def test_aa(self, lm, fst):
        sampler = importance_sampling(lm, fst, 'aa')
        item = sampler.sample()
        assert item is not None
        expected = np.log(0.5 * 0.5)
        assert item.weight == pytest.approx(expected, abs=1e-6)

    def test_with_nonrecursive_decompose(self, lm, fst):
        sampler = importance_sampling(lm, fst, 'abc',
                                      decompose=NonrecursiveDFADecomp)
        item = sampler.sample()
        assert item is not None
        expected = np.log(0.5 * 0.4 * 0.1)
        assert item.weight == pytest.approx(expected, abs=1e-6)


# --- crude_importance_sampling tests ---

class TestCrudeImportanceSampling:

    def test_abc_returns_item(self, lm, fst):
        sampler = crude_importance_sampling(lm, fst, 'abc')
        item = sampler.sample(max_length=100)
        assert item is not None
        assert np.isfinite(item.weight)

    def test_aa_returns_item(self, lm, fst):
        sampler = crude_importance_sampling(lm, fst, 'aa')
        item = sampler.sample(max_length=100)
        assert item is not None
        assert np.isfinite(item.weight)


# ---------------------------------------------------------------------------
# BPE-scale tests with a real LM (GPT-2) and subsampled vocabulary
# ---------------------------------------------------------------------------

try:
    from transduction.lm import StateLM
    from transformers import AutoTokenizer
    HAS_LLM_DEPS = True
except ImportError:
    HAS_LLM_DEPS = False

pytestmark_bpe = pytest.mark.skipif(not HAS_LLM_DEPS, reason="requires transformers + genlm + tokenization")


def _subsampled_bpe_fst(decode, token_ids, special=frozenset()):
    """Build a BPE WFST from a subset of token IDs.

    Uses the bytes representation of each token as the FST input label
    (matching what StateLM expects for ``<<`` and ``logp_next``).
    """
    from transduction import FST, EPSILON
    m = FST()
    m.add_start(())
    for i in token_ids:
        x = decode[i]
        if x in special:
            continue
        bx = tuple(x)
        for j in range(len(bx)):
            m.add_arc(bx[:j], EPSILON, bytes([bx[j]]), bx[:j + 1])
        m.add_arc(bx, x, EPSILON, ())   # bytes token as input label
    m.add_stop(())
    return m


@pytest.fixture(scope='session')
def gpt2_lm():
    if not HAS_LLM_DEPS:
        pytest.skip("requires transformers + genlm + tokenization")
    return StateLM.initial('gpt2')


@pytest.fixture(scope='session')
def gpt2_vocab(gpt2_lm):
    if not HAS_LLM_DEPS:
        pytest.skip("requires transformers + genlm + tokenization")
    decode = gpt2_lm.lm._decode
    special = {gpt2_lm.lm.tokenizer.eos_token.encode()}
    return decode, special


@pytest.fixture(scope='session')
def small_bpe_fst(gpt2_vocab):
    """BPE FST subsampled to ~500 tokens (all single-byte + target-relevant + random)."""
    import random
    rng = random.Random(42)
    decode, special = gpt2_vocab

    token_ids = set()
    for i, x in enumerate(decode):
        if x in special:
            continue
        if len(x) == 1:                             # all single-byte tokens
            token_ids.add(i)
        if b'the' in x or b'th' in x or b'he' in x:  # target-relevant
            token_ids.add(i)

    # pad to ~350 with random multi-byte tokens
    remaining = [i for i, x in enumerate(decode) if x not in special and i not in token_ids]
    rng.shuffle(remaining)
    token_ids.update(remaining[:max(0, 350 - len(token_ids))])

    return _subsampled_bpe_fst(decode, token_ids, special).renumber()


@pytestmark_bpe
class TestBPEScale:

    @pytest.mark.timeout(60)
    def test_decomposition(self, small_bpe_fst):
        """NonrecursiveDFADecomp completes on a subsampled BPE FST."""
        target = (b't', b'h', b'e')
        decomp = NonrecursiveDFADecomp(small_bpe_fst, target)
        q = decomp.quotient.trim()
        r = decomp.remainder.trim()
        # quotient should be non-trivial (there exist tokenizations of "the")
        assert q.states
        assert not r.states
        assert q.start

    @pytest.mark.timeout(60)
    def test_prioritized_enumeration(self, gpt2_lm, small_bpe_fst):
        """prioritized_enumeration with GPT-2 on a subsampled BPE FST."""
        target = (b't', b'h', b'e')
        pe = prioritized_enumeration(
            gpt2_lm, small_bpe_fst, target, max_steps=5,
            decompose=NonrecursiveDFADecomp,
        )
        assert len(pe.quotient_terms) >= 1
        for item in pe.quotient_terms:
            assert np.isfinite(item.weight)

    @pytest.mark.timeout(60)
    def test_importance_sampling(self, gpt2_lm, small_bpe_fst):
        """importance_sampling with GPT-2 on a subsampled BPE FST."""
        target = (b't', b'h', b'e')
        sampler = importance_sampling(
            gpt2_lm, small_bpe_fst, target,
            decompose=NonrecursiveDFADecomp,
        )
        item = sampler.sample(max_length=50)
        assert item is not None
        assert np.isfinite(item.weight)
