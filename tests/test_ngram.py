"""Tests for ByteNgramLM and CharNgramLM, extracted from ngram_demo.ipynb."""

import pytest
import numpy as np

from transduction.fst import FST
from transduction.lm.ngram import ByteNgramLM, CharNgramLM


CORPUS = b"""
the cat sat on the mat. the dog sat on the log.
the cat chased the dog. the dog chased the cat.
a bird flew over the lazy dog. the quick brown fox jumped.
the cat is on the mat. the dog is on the log.
""" * 10


@pytest.fixture
def lm():
    return ByteNgramLM.train(CORPUS, n=4, alpha=0.01)


class TestByteNgramTraining:

    def test_train_produces_contexts(self, lm):
        assert len(lm._tables) == 201

    def test_repr(self, lm):
        assert repr(lm) == 'ByteNgramLM(n=4, contexts=201)'


class TestByteNgramInterface:

    def test_advance_state(self, lm):
        state = lm.initial()
        for ch in b'the ':
            state = state >> bytes([ch])
        assert state._context == (ord('h'), ord('e'), ord(' '))
        assert state.logp == pytest.approx(-2.718, abs=0.01)

    def test_call_shorthand(self, lm):
        """lm(prompt) advances through each byte."""
        state = lm.initial()
        for ch in b'the ':
            state = state >> bytes([ch])
        state2 = lm([bytes([ch]) for ch in b'the '])
        assert state.logp == pytest.approx(state2.logp, abs=1e-10)

    def test_top_predictions_after_the_space(self, lm):
        state = lm.initial()
        for ch in b'the ':
            state = state >> bytes([ch])
        top = state.logp_next.materialize(top=5)
        top_bytes = list(top.keys())
        # After "the ", the top predictions from the corpus are c, d, l, m, q
        assert top_bytes == [b'c', b'd', b'l', b'm', b'q']

    def test_logp_next_sums_to_one(self, lm):
        state = lm.initial()
        for ch in b'the ':
            state = state >> bytes([ch])
        total = sum(np.exp(lp) for lp in state.logp_next.materialize().values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_path_recovery(self, lm):
        state = lm.initial()
        for ch in b'hello':
            state = state >> bytes([ch])
        assert state.path_bytes() == b'hello'


class TestByteNgramGreedyDecode:

    def test_greedy_the_space(self, lm):
        state = lm.initial()
        for ch in b'the ':
            state = state >> bytes([ch])
        decoded = b''.join(state.greedy_decode())
        # Greedy from "the " loops on "cat on the ..."
        assert decoded.startswith(b'cat')
        assert b'the' in decoded

    def test_greedy_a_space(self, lm):
        decoded = b''.join(lm([bytes([ch]) for ch in b'a ']).greedy_decode())
        assert decoded.startswith(b'bird')

    def test_greedy_the_d(self, lm):
        decoded = b''.join(lm([bytes([ch]) for ch in b'the d']).greedy_decode())
        assert decoded.startswith(b'og')


class TestByteNgramScoring:

    def test_in_distribution_vs_nonsense(self, lm):
        """In-distribution text should score much better than nonsense."""
        def score(text):
            state = lm.initial()
            for ch in text:
                state = state >> bytes([ch])
            return state.logp

        lp_good = score(b'the cat sat on the mat.')
        lp_bad = score(b'xyzzy plugh grault.')
        assert lp_good > lp_bad

    def test_similar_sentences_have_similar_scores(self, lm):
        def score(text):
            state = lm.initial()
            for ch in text:
                state = state >> bytes([ch])
            return state.logp

        lp1 = score(b'the cat sat on the mat.')
        lp2 = score(b'the dog sat on the log.')
        assert abs(lp1 - lp2) < 2.0  # similar sentences, similar scores

    def test_perplexity_in_distribution(self, lm):
        text = b'the cat sat on the mat.'
        state = lm.initial()
        for ch in text:
            state = state >> bytes([ch])
        ppl = np.exp(-state.logp / len(text))
        assert ppl < 10  # low perplexity for in-distribution text

    def test_perplexity_nonsense(self, lm):
        text = b'xyzzy plugh grault.'
        state = lm.initial()
        for ch in text:
            state = state >> bytes([ch])
        ppl = np.exp(-state.logp / len(text))
        assert ppl > 100  # high perplexity for nonsense


class TestCharNgramLM:

    def test_train_and_predict(self):
        lm = CharNgramLM.train("abcabc", n=2)
        state = lm.initial()
        state = state >> 'a'
        # After 'a', 'b' should be the most likely next character
        assert state.logp_next['b'] > state.logp_next['c']

    def test_logp_next_sums_to_one(self):
        lm = CharNgramLM.train("abcabc", n=2)
        state = lm.initial()
        total = sum(np.exp(lp) for lp in state.logp_next.materialize().values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_path_recovery(self):
        lm = CharNgramLM.train("abcabc", n=2)
        state = lm.initial()
        state = state >> 'a'
        state = state >> 'b'
        assert state.path() == ['a', 'b']


# ---------------------------------------------------------------------------
# WikiText + byte-level lowercase FST + prioritized enumeration
# (from ngram_demo.ipynb cells 14-18)
# ---------------------------------------------------------------------------

try:
    from transduction.applications.wikitext import load_wikitext
    from transduction.rust_bridge import RustDecomp
    from transduction.enumeration import prioritized_enumeration
    HAS_WIKITEXT_DEPS = True
except ImportError:
    HAS_WIKITEXT_DEPS = False


def byte_lowercase_fst():
    """Byte-level lowercase FST: maps uppercase ASCII to lowercase,
    passes lowercase and space through unchanged.

    Uses bytes labels (e.g., b't') to match ByteNgramLM's vocabulary.
    """
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    for i in range(256):
        c = chr(i) if i < 128 else None
        if c and c.isalpha():
            lo = ord(c.lower())
            fst.add_arc(0, bytes([i]), bytes([lo]), 0)
        elif c == ' ':
            fst.add_arc(0, bytes([i]), bytes([i]), 0)
    return fst


@pytest.fixture(scope='module')
def wikitext_lm():
    if not HAS_WIKITEXT_DEPS:
        pytest.skip("requires datasets, transduction_core")
    chunks = []
    for item in load_wikitext('train'):
        text = item['text'].strip()
        if text:
            chunks.append(text.encode('utf-8'))
        if sum(len(c) for c in chunks) > 1_000_000:
            break
    train_data = b'\n'.join(chunks)
    return ByteNgramLM.train(train_data, n=5, alpha=0.001)


@pytest.mark.skipif(not HAS_WIKITEXT_DEPS, reason="requires datasets, transduction_core")
class TestWikitextEnumeration:

    def test_train_wikitext_lm(self, wikitext_lm):
        """Training on ~1MB of WikiText produces a large model."""
        assert len(wikitext_lm._tables) > 10000

    def test_lowercase_fst_shape(self):
        fst = byte_lowercase_fst()
        assert len(fst.states) == 1
        assert len(fst.A) == 53   # 26 upper + 26 lower + space
        assert len(fst.B) == 27   # 26 lower + space

    def test_decomposition(self, wikitext_lm):
        """RustDecomp on 'in january' through lowercase FST."""
        fst = byte_lowercase_fst()
        target = tuple(bytes([b]) for b in b'in january')

        result = RustDecomp(fst, target)
        Q, R = result.quotient, result.remainder
        assert len(Q.stop) >= 1
        assert len(R.stop) == 0

    @pytest.mark.timeout(60)
    def test_prioritized_enumeration(self, wikitext_lm):
        """Prioritized enumeration through byte lowercase FST with WikiText LM.

        Target 'in january' appears frequently in WikiText as 'in January'.
        The n-gram LM should rank 'in January' as the most likely source.
        """
        fst = byte_lowercase_fst()
        target = tuple(bytes([b]) for b in b'in january')

        pe = prioritized_enumeration(
            wikitext_lm.initial(), fst, target,
            max_steps=100_000, decompose=RustDecomp,
        )
        all_terms = pe.quotient_terms + pe.remainder_terms
        assert len(all_terms) > 0
        best = max(all_terms, key=lambda x: x.weight)
        assert best.source.path_bytes() == b'in January'

    def test_greedy_decode_wikitext_lm(self, wikitext_lm):
        """Greedy decode from common prompts produces plausible continuations."""
        for prompt in [b'The ', b'In 200', b'He was ']:
            state = wikitext_lm.initial()
            for ch in prompt:
                state = state >> bytes([ch])
            decoded = b''.join(state.greedy_decode())
            assert len(decoded) > 0
