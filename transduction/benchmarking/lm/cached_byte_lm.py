"""Cached byte-level language model using genlm-bytes with LRU beam cache."""

import numpy as np
from collections import OrderedDict
from typing import List, Tuple

from transduction.benchmarking.config.constants import NEG_INF, EOS_IDX
from transduction.benchmarking.config.pruning import PruningConfig, CONFIGS


class CachedByteLM:
    """Cached byte-level LM using genlm-bytes with LRU beam cache.

    Wraps genlm's ByteBeamState, caching intermediate beams so shared
    prefixes across BFS paths aren't recomputed.
    """

    @classmethod
    async def create(
        cls,
        model_name: str = "gpt2",
        cfg: PruningConfig = None,
        verbose: bool = False,
        # Legacy kwargs for backward compatibility
        K: int = None,
        prune_threshold: float = None,
        cache_size: int = None,
    ):
        """Create a CachedByteLM with the given model and config.

        Parameters
        ----------
        model_name : str
            HuggingFace model name (default: "gpt2").
        cfg : PruningConfig
            Configuration for beam pruning. If None, uses balanced preset.
        verbose : bool
            Print warnings on genlm errors.
        """
        if cfg is None:
            cfg = CONFIGS["balanced"]

        # Support legacy API
        beam_k = K if K is not None else cfg.beam_k
        beam_prune = (
            prune_threshold if prune_threshold is not None else cfg.beam_prune_threshold
        )
        beam_cache = cache_size if cache_size is not None else cfg.beam_cache_size

        from genlm.backend import load_model_by_name
        from genlm.bytes import ByteBeamState, BeamParams

        llm = load_model_by_name(model_name, backend="hf")
        eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
        root_beam = await ByteBeamState.initial(
            llm,
            BeamParams(K=beam_k, eos_tokens=[eos_token], prune_threshold=beam_prune),
        )
        return cls(llm, root_beam, beam_cache, verbose)

    def __init__(self, llm, root_beam, cache_size: int = 10000, verbose: bool = False):
        self.llm = llm
        self.root_beam = root_beam
        self._cache_size = cache_size
        self._beams = OrderedDict()
        self._beams[()] = root_beam
        self._logp_cache = OrderedDict()
        self.verbose = verbose
        self.beam_hits = 0
        self.beam_misses = 0

    def _cache_put(self, cache, key, value):
        """Insert into OrderedDict cache with LRU eviction."""
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._cache_size:
            cache.popitem(last=False)

    async def _beam_for(self, ctx: Tuple[int, ...]):
        """Recursively build (and cache) a beam for byte context."""
        if ctx in self._beams:
            self.beam_hits += 1
            self._beams.move_to_end(ctx)
            return self._beams[ctx]
        self.beam_misses += 1
        if ctx == ():
            raise RuntimeError("root beam should already be cached")
        parent_beam = await self._beam_for(ctx[:-1])
        beam = await (parent_beam.prune() << int(ctx[-1]))
        self._cache_put(self._beams, ctx, beam)
        return beam

    async def logp_next_for(self, ctx: Tuple[int, ...]) -> np.ndarray:
        """Get logp_next array [257] for byte context tuple."""
        if ctx in self._logp_cache:
            self._logp_cache.move_to_end(ctx)
            return self._logp_cache[ctx]

        from genlm.bytes.trie import EOS as GENLM_EOS

        try:
            beam = await self._beam_for(ctx)
            lbp = await beam.logp_next()
            ps = getattr(lbp, "ps", None)
            if ps is not None:
                arr = np.asarray(ps, dtype=np.float64)
                if arr.shape[0] < 257:
                    padded = np.full(257, NEG_INF, dtype=np.float64)
                    padded[: arr.shape[0]] = arr
                    arr = padded
            else:
                arr = np.full(257, NEG_INF, dtype=np.float64)
                Q = lbp.materialize()
                for k, v in Q.items():
                    if k == GENLM_EOS:
                        arr[EOS_IDX] = v
                    elif isinstance(k, int) and 0 <= k <= 255:
                        arr[k] = v
        except (AssertionError, ValueError) as e:
            if self.verbose:
                print(f"WARNING: genlm error: {e}")
            arr = np.full(257, NEG_INF, dtype=np.float64)

        self._cache_put(self._logp_cache, ctx, arr)
        return arr

    async def score_path(self, path_bytes: List[int]) -> float:
        """Score a byte path autoregressively. Returns logp."""
        logp = 0.0
        ctx = ()
        for b in path_bytes:
            arr = await self.logp_next_for(ctx)
            logp += arr[b]
            ctx = ctx + (b,)
        return logp

    async def score_path_with_eos(self, path_bytes: List[int]) -> float:
        """Score a byte path + EOS autoregressively. Returns logp."""
        logp = await self.score_path(path_bytes)
        eos_arr = await self.logp_next_for(tuple(path_bytes))
        return logp + eos_arr[EOS_IDX]

    async def cleanup(self):
        await self.root_beam.cleanup()
