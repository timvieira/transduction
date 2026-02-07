"""Tests for the minimal PTB benchmark scripts.

Verifies that compute_full_distribution (both LM and no-LM variants) return
valid, normalized distributions on small FSTs.
"""

import numpy as np
import pytest
from collections import OrderedDict
from scipy.special import logsumexp

from transduction import FST, EPSILON
from transduction.benchmarking.lm.cached_byte_lm import CachedByteLM
from transduction.benchmarking.config.constants import NEG_INF
from transduction.rust_bridge import RustPeekaboo
from transduction.benchmarking.run_ptb_benchmark import (
    compute_full_distribution as compute_full_lm,
)
from transduction.benchmarking.run_ptb_benchmark_nolm import (
    compute_full_distribution as compute_full_nolm,
)
from tests.test_cached_byte_lm import MockBeamForCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_logp():
    rng = np.random.RandomState(42)
    raw = rng.dirichlet(np.ones(257))
    return np.log(raw).astype(np.float64)


@pytest.fixture
def mock_lm(base_logp):
    root = MockBeamForCache(base_logp)
    lm = CachedByteLM.__new__(CachedByteLM)
    lm.llm = None
    lm.root_beam = root
    lm._cache_size = 1000
    lm._beams = OrderedDict()
    lm._beams[()] = root
    lm._logp_cache = OrderedDict()
    lm.verbose = False
    lm.beam_hits = 0
    lm.beam_misses = 0
    return lm


@pytest.fixture
def byte_fst():
    """Simple FST on byte-valued symbols (matching PTB convention)."""
    fst = FST()
    fst.add_start(0)
    fst.add_arc(0, "65", "65", 0)  # A -> A
    fst.add_arc(0, "66", "66", 0)  # B -> B
    fst.add_arc(0, "65", "66", 1)  # A -> B (alternative)
    fst.add_arc(1, "66", "65", 0)  # B -> A
    fst.add_stop(0)
    return fst


@pytest.fixture
def peekaboo(byte_fst):
    return RustPeekaboo(byte_fst)


# ---------------------------------------------------------------------------
# No-LM tests
# ---------------------------------------------------------------------------


class TestNoLM:

    def test_returns_distribution(self, peekaboo, byte_fst):
        dist = compute_full_nolm(
            peekaboo, byte_fst, target=(), max_depth=10, max_paths=50
        )
        assert isinstance(dist, dict)
        assert len(dist) > 0

    def test_normalized(self, peekaboo, byte_fst):
        dist = compute_full_nolm(
            peekaboo, byte_fst, target=(), max_depth=10, max_paths=50
        )
        vals = np.array(list(dist.values()))
        total = logsumexp(vals)
        assert abs(total) < 1e-6, f"distribution not normalized: logsumexp = {total}"

    def test_values_are_log_probs(self, peekaboo, byte_fst):
        dist = compute_full_nolm(
            peekaboo, byte_fst, target=(), max_depth=10, max_paths=50
        )
        for sym, logp in dist.items():
            assert logp <= 0.0 + 1e-9, f"logp({sym}) = {logp} > 0"

    def test_after_one_step(self, peekaboo, byte_fst):
        """Distribution after consuming one symbol should still be valid."""
        dist = compute_full_nolm(
            peekaboo, byte_fst, target=("65",), max_depth=10, max_paths=50
        )
        assert isinstance(dist, dict)
        if dist:
            vals = np.array(list(dist.values()))
            total = logsumexp(vals)
            assert abs(total) < 1e-6


# ---------------------------------------------------------------------------
# LM tests
# ---------------------------------------------------------------------------


class TestWithLM:

    @pytest.mark.asyncio
    async def test_returns_distribution(self, peekaboo, byte_fst, mock_lm):
        dist = await compute_full_lm(
            peekaboo, byte_fst, target=(), lm=mock_lm, max_depth=10, max_paths=50
        )
        assert isinstance(dist, dict)
        assert len(dist) > 0

    @pytest.mark.asyncio
    async def test_normalized(self, peekaboo, byte_fst, mock_lm):
        dist = await compute_full_lm(
            peekaboo, byte_fst, target=(), lm=mock_lm, max_depth=10, max_paths=50
        )
        vals = np.array(list(dist.values()))
        total = logsumexp(vals)
        assert abs(total) < 1e-6, f"distribution not normalized: logsumexp = {total}"

    @pytest.mark.asyncio
    async def test_values_are_log_probs(self, peekaboo, byte_fst, mock_lm):
        dist = await compute_full_lm(
            peekaboo, byte_fst, target=(), lm=mock_lm, max_depth=10, max_paths=50
        )
        for sym, logp in dist.items():
            assert logp <= 0.0 + 1e-9, f"logp({sym}) = {logp} > 0"

    @pytest.mark.asyncio
    async def test_after_one_step(self, peekaboo, byte_fst, mock_lm):
        dist = await compute_full_lm(
            peekaboo, byte_fst, target=("65",), lm=mock_lm, max_depth=10, max_paths=50
        )
        assert isinstance(dist, dict)
        if dist:
            vals = np.array(list(dist.values()))
            total = logsumexp(vals)
            assert abs(total) < 1e-6

    @pytest.mark.asyncio
    async def test_differs_from_nolm(self, peekaboo, byte_fst, mock_lm):
        """LM-scored distribution should differ from uniform-weight distribution."""
        dist_lm = await compute_full_lm(
            peekaboo, byte_fst, target=(), lm=mock_lm, max_depth=10, max_paths=50
        )
        dist_nolm = compute_full_nolm(
            peekaboo, byte_fst, target=(), max_depth=10, max_paths=50
        )
        # Both should have entries, but at least some values should differ
        common = set(dist_lm) & set(dist_nolm)
        assert len(common) > 0
        diffs = [abs(dist_lm[s] - dist_nolm[s]) for s in common]
        assert max(diffs) > 1e-6, "LM and no-LM distributions are identical"
