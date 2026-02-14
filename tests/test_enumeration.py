import pytest
import numpy as np
from collections import defaultdict

from transduction import examples
from transduction.lm.base import LMState
from transduction.enumeration import (
    prioritized_enumeration,
    importance_sampling,
    crude_importance_sampling,
)
from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp


EOS = '<EOS>'


class SimpleGrammarLM(LMState):
    """LM for the grammar: S -> aS (.5) | bS (.4) | c (.1).

    Single-state weighted regular language — no parser needed.
    Before 'c': P(a)=0.5, P(b)=0.4, P(c)=0.1.
    After 'c': P(EOS)=1.0.
    """

    def __init__(self, finished=False):
        self._finished = finished

    @property
    def eos(self):
        return EOS

    @property
    def logp_next(self):
        if self._finished:
            return defaultdict(lambda: float('-inf'), {EOS: 0.0})
        return defaultdict(lambda: float('-inf'), {
            'a': np.log(0.5),
            'b': np.log(0.4),
            'c': np.log(0.1),
        })

    def __rshift__(self, token):
        if token == 'c':
            return SimpleGrammarLM(finished=True)
        return SimpleGrammarLM(finished=False)

    def __repr__(self):
        return f'SimpleGrammarLM(finished={self._finished})'


class PalindromeLM(LMState):
    """LM for the grammar: S -> aSa (.5) | bSb (.4) | c (.1).

    Generates palindromes over {a,b} with center 'c'.
    Before 'c': P(a)=0.5, P(b)=0.4, P(c)=0.1 (building left half, push).
    After 'c': deterministically mirror the left half (pop), then EOS.
    """

    def __init__(self, stack=()):
        self._stack = stack   # None means finished (empty stack after mirror)

    @property
    def eos(self):
        return EOS

    @property
    def logp_next(self):
        if self._stack is None:
            return defaultdict(lambda: float('-inf'), {EOS: 0.0})
        if self._stack and self._stack[-1] == 'c':
            # In mirror phase: must output top of stack (symbol before 'c')
            mirror_stack = self._stack[:-1]  # drop the 'c' sentinel
            if not mirror_stack:
                return defaultdict(lambda: float('-inf'), {EOS: 0.0})
            return defaultdict(lambda: float('-inf'), {mirror_stack[-1]: 0.0})
        # Still in left half: push phase
        return defaultdict(lambda: float('-inf'), {
            'a': np.log(0.5),
            'b': np.log(0.4),
            'c': np.log(0.1),
        })

    def __rshift__(self, token):
        if self._stack is None:
            raise ValueError("LM is finished")
        if token == 'c' and (not self._stack or self._stack[-1] != 'c'):
            # Transition to mirror phase: push 'c' as sentinel
            return PalindromeLM(self._stack + ('c',))
        if self._stack and self._stack[-1] == 'c':
            # Mirror phase: pop
            mirror_stack = self._stack[:-1]
            assert mirror_stack and mirror_stack[-1] == token
            remaining = mirror_stack[:-1]
            if not remaining:
                return PalindromeLM(None)  # finished
            return PalindromeLM(remaining + ('c',))
        # Push phase
        return PalindromeLM(self._stack + (token,))

    def __repr__(self):
        return f'PalindromeLM(stack={self._stack})'


@pytest.fixture(params=['simple', 'palindrome'])
def lm(request):
    if request.param == 'simple':
        return SimpleGrammarLM()
    return PalindromeLM()


@pytest.fixture
def fst():
    return examples.replace([('a', 'a'), ('b', 'b'), ('c', 'c')])


# Both LMs share the same left-half distribution P(a)=0.5, P(b)=0.4, P(c)=0.1,
# so any prefix before the first 'c' has the same weight under both.
# Targets:
#   'a'    — single-symbol prefix, weight log(0.5), quotient under both
#   'ab'   — two-symbol prefix, weight log(0.5*0.4), quotient under both
#   'abc'  — simple: remainder (complete string), palindrome: quotient (needs mirror 'a')
#   'aca'  — simple: quotient (prefix of 'aca...'), palindrome: remainder (complete palindrome)
#   'c'    — remainder under both (single-symbol complete string), weight log(0.1)
#   ''     — empty prefix, quotient under both


# --- prioritized_enumeration tests ---

class TestPrioritizedEnumeration:

    @pytest.mark.parametrize('target, expected_logp', [
        ('a', np.log(0.5)),
        ('ab', np.log(0.5 * 0.4)),
        ('c', np.log(0.1)),
    ])
    def test_exact_weight(self, lm, fst, target, expected_logp):
        pe = prioritized_enumeration(lm, fst, target, max_steps=100)
        assert len(pe.quotient_terms) + len(pe.remainder_terms) >= 1
        # The top term should match the expected weight
        all_terms = pe.quotient_terms + pe.remainder_terms
        best = max(all_terms, key=lambda t: t.weight)
        assert best.weight == pytest.approx(expected_logp, abs=1e-6)

    @pytest.mark.parametrize('target', ['', 'a', 'ab', 'abc', 'aca', 'c'])
    def test_completes(self, lm, fst, target):
        """Enumeration completes without error on various targets."""
        pe = prioritized_enumeration(lm, fst, target, max_steps=100)

    def test_with_nonrecursive_decompose(self, lm, fst):
        pe = prioritized_enumeration(lm, fst, 'ab', max_steps=50,
                                     decompose=NonrecursiveDFADecomp)
        assert len(pe.quotient_terms) >= 1
        assert pe.quotient_terms[0].weight == pytest.approx(np.log(0.5 * 0.4), abs=1e-6)


# --- importance_sampling tests ---

class TestImportanceSampling:

    @pytest.mark.parametrize('target, expected_logp', [
        ('a', np.log(0.5)),
        ('ab', np.log(0.5 * 0.4)),
        ('c', np.log(0.1)),
    ])
    def test_exact_weight(self, lm, fst, target, expected_logp):
        sampler = importance_sampling(lm, fst, target)
        item = sampler.sample()
        assert item is not None
        assert item.weight == pytest.approx(expected_logp, abs=1e-6)

    @pytest.mark.parametrize('target', ['', 'a', 'ab', 'abc', 'aca', 'c'])
    def test_completes(self, lm, fst, target):
        """Sampling completes without error on various targets."""
        sampler = importance_sampling(lm, fst, target)
        sampler.sample()

    def test_with_nonrecursive_decompose(self, lm, fst):
        sampler = importance_sampling(lm, fst, 'ab',
                                      decompose=NonrecursiveDFADecomp)
        item = sampler.sample()
        assert item is not None
        assert item.weight == pytest.approx(np.log(0.5 * 0.4), abs=1e-6)


# --- crude_importance_sampling tests ---

class TestCrudeImportanceSampling:

    @pytest.mark.parametrize('target', ['', 'a', 'ab', 'abc', 'aca', 'c'])
    def test_completes(self, lm, fst, target):
        """Sampling completes without error on various targets."""
        sampler = crude_importance_sampling(lm, fst, target)
        sampler.sample(max_length=100)


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
