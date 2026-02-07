"""Autoregressive scoring: decompose, enumerate paths, and score with LM."""

import numpy as np
from scipy.special import logsumexp
import time
import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from transduction.benchmarking.config.constants import NEG_INF, EOS_IDX
from transduction.benchmarking.config.pruning import PruningConfig, CONFIGS
from transduction.benchmarking.decomp.path_enumeration import enumerate_fsa_paths_bfs

# line_profiler support: no-op decorator when not profiling
try:
    profile  # noqa: F821 - injected by kernprof
except NameError:

    def profile(func):
        return func


@dataclass
class LogpNextResult:
    """Result of logp_next computation at one step."""

    target: Tuple[str, ...]  # Current output prefix
    logp_target: float  # logp(target) under transduced LM
    distribution: Dict[str, float]  # {symbol: logp(symbol | target)}
    eos_logp: float  # logp(EOS | target)
    decomp_time_ms: float  # Time for decomposition
    n_q_paths: int  # Number of quotient paths enumerated
    n_r_paths: int  # Number of remainder paths enumerated


@dataclass
class DecompResult:
    """Result of decomposing for a single target."""

    target: Tuple[str, ...]
    q_paths: List[List[int]]  # Quotient paths (bytes)
    r_paths: List[List[int]]  # Remainder paths (bytes)


@profile
def decompose_single(
    decomposer, target: Tuple[str, ...], cfg: PruningConfig, max_depth: int = None
) -> DecompResult:
    """Decompose FST for a single target and enumerate paths (no LM scoring yet)."""
    Q, R = decomposer.decompose(target)

    uniform = np.full(257, 0.0)

    q_paths = [
        path
        for path, _, _ in enumerate_fsa_paths_bfs(Q, uniform, cfg, max_depth=max_depth)
    ]
    r_paths = [
        path
        for path, _, _ in enumerate_fsa_paths_bfs(R, uniform, cfg, max_depth=max_depth)
    ]

    return DecompResult(target=target, q_paths=q_paths, r_paths=r_paths)


@profile
def decompose_batch(
    decomposer,
    targets: List[Tuple[str, ...]],
    cfg: PruningConfig,
    max_depth: int = None,
) -> Dict[Tuple[str, ...], DecompResult]:
    """Decompose FST for multiple targets in batch.

    This separates decomposition from LM scoring, allowing:
    1. All decompositions to happen first (could be parallelized)
    2. Path scoring to be optimized by prefix ordering
    """
    results = {}
    for target in targets:
        results[target] = decompose_single(decomposer, target, cfg, max_depth)
    return results


@profile
async def score_paths_optimized(
    paths_by_key: Dict[str, List[List[int]]],
    lm,
    add_eos: Dict[str, bool],
) -> Dict[str, float]:
    """Score all paths with LM, optimizing for cache hits.

    Sorts paths by prefix to maximize beam cache reuse, then scores
    and aggregates back by key.

    Parameters
    ----------
    paths_by_key : dict
        {key: [path1, path2, ...]} where each path is a list of byte values
    lm : CachedByteLM
        The autoregressive LM
    add_eos : dict
        {key: True/False} whether to add EOS to path scores

    Returns
    -------
    dict
        {key: logsumexp(path_scores)} aggregated log probabilities
    """
    # Flatten all paths with their keys
    all_paths = []
    for key, paths in paths_by_key.items():
        for path in paths:
            all_paths.append((key, tuple(path), add_eos.get(key, False)))

    if not all_paths:
        return {}

    # Sort by path prefix to maximize cache hits
    # Paths are sorted lexicographically so shared prefixes are adjacent
    all_paths.sort(key=lambda x: x[1])

    # Score paths in optimized order
    scores_by_key = {}
    for key, path_tuple, needs_eos in all_paths:
        path = list(path_tuple)
        if needs_eos:
            logp = await lm.score_path_with_eos(path)
        else:
            logp = await lm.score_path(path)

        if logp > NEG_INF:
            if key not in scores_by_key:
                scores_by_key[key] = []
            scores_by_key[key].append(logp)

    # Aggregate by logsumexp
    return {
        key: logsumexp(logps) if logps else NEG_INF
        for key, logps in scores_by_key.items()
    }


async def _score_decomp_result(
    decomp: DecompResult,
    lm,
) -> Tuple[float, float, float]:
    """Score a DecompResult with the autoregressive LM.

    Returns (total_logp, q_logp, r_logp).
    """
    paths_by_key = {}
    add_eos = {}
    if decomp.q_paths:
        paths_by_key["q"] = decomp.q_paths
        add_eos["q"] = False
    if decomp.r_paths:
        paths_by_key["r"] = decomp.r_paths
        add_eos["r"] = True

    if not paths_by_key:
        return NEG_INF, NEG_INF, NEG_INF

    scored = await score_paths_optimized(paths_by_key, lm, add_eos)
    q_logp = scored.get("q", NEG_INF)
    r_logp = scored.get("r", NEG_INF)
    total = logsumexp([q_logp, r_logp])
    return total, q_logp, r_logp


async def compute_logp_cached(
    fst,
    target: Tuple[str, ...],
    lm,
    cfg: PruningConfig = None,
    max_depth: int = None,
    max_paths: int = None,
    decomposer=None,
) -> Tuple[float, int, int]:
    """Compute logp(target) with autoregressive LM scoring.

    Convenience wrapper around decompose_single + score_paths_optimized.
    """
    from transduction.benchmarking.decomp.decomposer import CachedDecomposer

    if cfg is None:
        cfg = CONFIGS["balanced"]
    if decomposer is None:
        decomposer = CachedDecomposer(fst)

    decomp = decompose_single(decomposer, target, cfg, max_depth)

    paths_by_key = {}
    add_eos = {}

    if decomp.q_paths:
        paths_by_key["q"] = decomp.q_paths
        add_eos["q"] = False
    if decomp.r_paths:
        paths_by_key["r"] = decomp.r_paths
        add_eos["r"] = True

    scored = await score_paths_optimized(paths_by_key, lm, add_eos)

    q_logp = scored.get("q", NEG_INF)
    r_logp = scored.get("r", NEG_INF)

    return logsumexp([q_logp, r_logp]), len(decomp.q_paths), len(decomp.r_paths)


@profile
async def compute_logp_next_batched(
    decomposer,
    target: Tuple[str, ...],
    lm,
    cfg: PruningConfig = None,
    max_depth: int = None,
    candidate_symbols: Optional[List[str]] = None,
    peekaboo=None,
) -> LogpNextResult:
    """Compute logp(z | target) using peekaboo decomposition + early stopping.

    Algorithm:
    1. Decompose and score target (single rust_decompose call)
    2. Use peekaboo (incremental or one-shot) for per-symbol decomposition
    3. Select candidates, ordered by LM probability (most likely first)
    4. For each candidate: lazily get Q/R from peekaboo, enumerate paths, score
    5. Stop early once cumulative probability mass exceeds threshold
    """
    from transduction.benchmarking.decomp.decomposer import IncrementalPeekaboo
    from transduction.benchmarking.fsts.ptb_pynini import SEP

    if cfg is None:
        cfg = CONFIGS["balanced"]

    t0 = time.perf_counter()

    # Decompose and score target
    target_decomp = decompose_single(decomposer, target, cfg, max_depth)
    logp_target, target_q_logp, target_r_logp = await _score_decomp_result(
        target_decomp, lm
    )

    n_q = len(target_decomp.q_paths)
    n_r = len(target_decomp.r_paths)

    if logp_target < NEG_INF / 2:
        decomp_time = (time.perf_counter() - t0) * 1000
        return LogpNextResult(
            target=target,
            logp_target=logp_target,
            distribution={},
            eos_logp=NEG_INF,
            decomp_time_ms=decomp_time,
            n_q_paths=n_q,
            n_r_paths=n_r,
        )

    # Per-symbol decomposition
    peek = peekaboo if peekaboo is not None else decomposer.peekaboo(target)

    # Select and order candidates
    if candidate_symbols is None:
        all_symbols = decomposer.target_alphabet

        if cfg.top_k_bytes > 0:
            lm_probs = await lm.logp_next_for(())
            top_indices = np.argsort(lm_probs[:256])[-cfg.top_k_bytes :]
            candidate_symbols = [str(b) for b in top_indices if str(b) in all_symbols]
            if SEP in all_symbols:
                candidate_symbols.append(SEP)
        else:
            candidate_symbols = sorted(all_symbols)

    # Order by initial LM probability (most likely first for early stopping)
    lm_probs = await lm.logp_next_for(())

    def _sym_logp(z):
        try:
            v = int(z)
            return float(lm_probs[v]) if v < 256 else NEG_INF
        except (ValueError, IndexError):
            return NEG_INF

    ordered = sorted(candidate_symbols, key=lambda z: -_sym_logp(z))
    ordered_set = set(ordered)

    # Score candidates on-demand with early stopping ===

    if cfg.use_unified and isinstance(peek, IncrementalPeekaboo):
        from transduction.benchmarking.beam_search import (
            build_forward_graph,
            unified_beam_search,
        )

        graph = build_forward_graph(peek._state, peek._merged_incoming)
        symbol_joint_logps = await unified_beam_search(
            graph,
            lm,
            beam_width=cfg.lm_beam_width,
            max_depth=max_depth if max_depth is not None else cfg.max_depth,
            logp_floor=cfg.lm_logp_floor,
        )
        scores = {}
        for z, joint_logp in symbol_joint_logps.items():
            if z in ordered_set and joint_logp > NEG_INF:
                scores[z] = joint_logp - logp_target

    elif cfg.use_lm_guided:
        from transduction.benchmarking.beam_search import (
            lm_guided_beam_search,
            allocate_symbol_budgets,
        )

        budgets = allocate_symbol_budgets(
            ordered,
            lm_probs,
            total_budget=cfg.max_paths * len(ordered),
            min_per_sym=5,
            max_per_sym=cfg.max_paths,
        )
        scores = {}
        cumulative_mass = 0.0
        for z in ordered:
            qr = peek.get(z)
            if qr is None:
                continue
            Q_z, R_z = qr

            sym_max_paths = budgets.get(z, cfg.max_paths)
            q_logp = await lm_guided_beam_search(
                Q_z,
                lm,
                is_remainder=False,
                beam_width=cfg.lm_beam_width,
                max_depth=max_depth if max_depth is not None else cfg.max_depth,
                max_paths=sym_max_paths,
                logp_floor=cfg.lm_logp_floor,
            )
            r_logp = await lm_guided_beam_search(
                R_z,
                lm,
                is_remainder=True,
                beam_width=cfg.lm_beam_width,
                max_depth=max_depth if max_depth is not None else cfg.max_depth,
                max_paths=sym_max_paths,
                logp_floor=cfg.lm_logp_floor,
            )

            ext_logp = logsumexp([q_logp, r_logp])
            if ext_logp > NEG_INF:
                cond_logp = ext_logp - logp_target
                scores[z] = cond_logp
                cumulative_mass += np.exp(cond_logp)

            if (
                cfg.early_stop_mass is not None
                and cumulative_mass >= cfg.early_stop_mass
            ):
                break

    # --- Original BFS + separate LM scoring path ---
    else:
        uniform = np.full(257, 0.0)
        scores = {}
        cumulative_mass = 0.0
        n_evaluated = 0

        for z in ordered:
            qr = peek.get(z)
            if qr is None:
                continue
            Q_z, R_z = qr

            q_paths = [
                path
                for path, _, _ in enumerate_fsa_paths_bfs(
                    Q_z, uniform, cfg, max_depth=max_depth
                )
            ]
            r_paths = [
                path
                for path, _, _ in enumerate_fsa_paths_bfs(
                    R_z, uniform, cfg, max_depth=max_depth
                )
            ]

            if not q_paths and not r_paths:
                continue

            ext_decomp = DecompResult(
                target=target + (z,), q_paths=q_paths, r_paths=r_paths
            )
            ext_logp, _, _ = await _score_decomp_result(ext_decomp, lm)
            n_evaluated += 1

            if ext_logp > NEG_INF:
                cond_logp = ext_logp - logp_target
                scores[z] = cond_logp
                cumulative_mass += np.exp(cond_logp)

            if (
                cfg.early_stop_mass is not None
                and cumulative_mass >= cfg.early_stop_mass
            ):
                break

    # === STEP 5: EOS + normalize ===
    eos_cond = (
        target_r_logp - logp_target
        if target_r_logp > NEG_INF and logp_target > NEG_INF
        else NEG_INF
    )

    all_scores = dict(scores)
    if eos_cond > NEG_INF:
        all_scores["EOS"] = eos_cond

    if all_scores:
        vals = np.array(list(all_scores.values()))
        logZ = logsumexp(vals)
        normalized = {k: v - logZ for k, v in all_scores.items()}
    else:
        normalized = {}

    decomp_time = (time.perf_counter() - t0) * 1000

    return LogpNextResult(
        target=target,
        logp_target=logp_target,
        distribution=normalized,
        eos_logp=eos_cond,
        decomp_time_ms=decomp_time,
        n_q_paths=n_q,
        n_r_paths=n_r,
    )


@profile
async def run_benchmark(
    decomposer,
    output: Tuple[str, ...],
    lm,
    cfg: PruningConfig = None,
    max_steps: int = 10,
    full: bool = False,
) -> List[LogpNextResult]:
    """Run benchmark with autoregressive LM scoring via CachedByteLM.

    Parameters
    ----------
    decomposer : CachedDecomposer
        Cached decomposer (holds pre-built Rust FST).
    output : tuple of str
        Full output sequence (PTB-normalized).
    lm : CachedByteLM
        The autoregressive byte-level LM.
    cfg : PruningConfig
        Pruning configuration.
    max_steps : int
        Number of output positions to evaluate.
    full : bool
        If True, evaluate ALL output symbols (ignores top_k_bytes).

    Returns
    -------
    list of LogpNextResult
        One result per step with distribution over next symbols.
    """
    if cfg is None:
        cfg = CONFIGS["balanced"]

    results = []
    n_steps = min(max_steps, len(output))
    candidate_symbols = sorted(decomposer.target_alphabet) if full else None

    # Use Rust peekaboo per step (IncrementalPeekaboo requires single-char
    # symbols for PeekabooState's string concatenation; PTB has multi-char
    # byte symbols like '65').
    for step in tqdm.tqdm(range(n_steps), desc="Computing logp_next"):
        target = output[:step]
        result = await compute_logp_next_batched(
            decomposer,
            target,
            lm,
            cfg,
            candidate_symbols=candidate_symbols,
        )
        results.append(result)
        n_syms = len(result.distribution)
        tqdm.tqdm.write(
            f"  step {step}: {result.decomp_time_ms:.0f}ms, "
            f"{n_syms} symbols, logp={result.logp_target:.2f}"
        )

    return results
