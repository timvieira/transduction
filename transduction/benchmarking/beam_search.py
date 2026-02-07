"""
Currently very slow.

Beam search approximations for fast next-symbol distributions.

Provides LM-guided beam search (per-symbol and unified) as faster alternatives
to the exhaustive BFS + separate LM scoring pipeline in lm/scoring.

Components:
1. lm_guided_beam_search: Per-symbol best-first search guided by LM logprobs
2. ForwardGraph + unified_beam_search: One search over shared peekaboo DFA
3. allocate_symbol_budgets: Adaptive per-symbol path budget
4. JSD evaluation framework: Measures approximation accuracy
"""

import heapq
import numpy as np
from scipy.special import logsumexp
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import time
import json

from transduction.fsa import EPSILON
from transduction.benchmarking.config.constants import NEG_INF, EOS_IDX

# =============================================================================
# Component 1: LM-Guided Per-Symbol Beam Search
# =============================================================================


async def lm_guided_beam_search(
    fsa,
    lm,
    is_remainder: bool,
    beam_width: int = 32,
    max_depth: int = 30,
    max_paths: int = 100,
    logp_floor: float = -40.0,
) -> float:
    """Best-first search over FSA guided by LM log-probabilities.

    Instead of BFS with uniform weights (then scoring separately), this
    uses the LM to guide exploration, finding high-probability paths first.

    Parameters
    ----------
    fsa : FSA
        The quotient or remainder FSA to search.
    lm : CachedByteLM
        Autoregressive byte-level LM with .logp_next_for(ctx) method.
    is_remainder : bool
        If True, add EOS logp to accepting path scores.
    beam_width : int
        Max entries to keep at each expansion step.
    max_depth : int
        Maximum path depth in bytes.
    max_paths : int
        Stop after finding this many accepting paths.
    logp_floor : float
        Prune entries with logp below this absolute floor.

    Returns
    -------
    float
        logsumexp of found path scores (NEG_INF if no paths found).
    """
    if not fsa.states or not fsa.start:
        return NEG_INF

    # Max-heap using negative logp (heapq is a min-heap)
    # Entries: (-logp, counter, state, ctx_tuple)
    counter = 0
    heap = []
    for s in fsa.start:
        heapq.heappush(heap, (0.0, counter, s, ()))  # -logp = -0.0 = 0.0
        counter += 1

    path_scores = []

    while heap and len(path_scores) < max_paths:
        neg_logp, _, state, ctx = heapq.heappop(heap)
        logp = -neg_logp

        if logp < logp_floor:
            continue

        # Check if accepting state
        if state in fsa.stop:
            if is_remainder:
                eos_arr = await lm.logp_next_for(ctx)
                path_scores.append(logp + eos_arr[EOS_IDX])
            else:
                path_scores.append(logp)
            # Don't expand from stop states in quotient FSAs (Q stops are leaves).
            # For remainder FSAs, stop states may also have outgoing arcs,
            # but we still don't expand -- the path is complete at the stop.
            continue

        if len(ctx) >= max_depth:
            continue

        # Get LM distribution for current context
        lm_arr = await lm.logp_next_for(ctx)

        # Expand arcs
        candidates = []
        for x, next_state in fsa.arcs(state):
            if x == EPSILON:
                # Epsilon transitions don't consume a byte
                candidates.append((-logp, counter, next_state, ctx))
                counter += 1
            else:
                try:
                    byte_val = int(x)
                except ValueError:
                    continue
                if byte_val >= len(lm_arr):
                    continue
                next_logp = logp + lm_arr[byte_val]
                if next_logp < logp_floor:
                    continue
                candidates.append((-next_logp, counter, next_state, ctx + (byte_val,)))
                counter += 1

        # Push all candidates; beam pruning happens at pop time via heap ordering
        for entry in candidates:
            heapq.heappush(heap, entry)

        # Beam pruning: if heap is too large, keep only the best entries
        if len(heap) > beam_width * 3:
            heap = heapq.nsmallest(beam_width, heap)
            heapq.heapify(heap)

    if not path_scores:
        return NEG_INF
    return logsumexp(path_scores)


# =============================================================================
# Component 2: Unified Beam Search on Shared Forward Graph
# =============================================================================


@dataclass
class ForwardGraph:
    """Forward-arc representation of the shared peekaboo DFA.

    Built by inverting the merged reverse-arc (incoming) graph from
    IncrementalPeekaboo. The same DFA is shared across all symbols --
    only the stop states differ.
    """

    forward: Dict  # pred -> [(label, succ)]
    start_states: Set
    q_stops: Dict[str, Set]  # symbol -> Q stop states
    r_stops: Dict[str, Set]  # symbol -> R stop states


def build_forward_graph(peek_state, merged_incoming) -> ForwardGraph:
    """Build a ForwardGraph from IncrementalPeekaboo state.

    Parameters
    ----------
    peek_state : PeekabooState
        The current peekaboo state (has .decomp and .dfa).
    merged_incoming : dict
        The merged reverse-arc graph: state -> {(label, pred), ...}

    Returns
    -------
    ForwardGraph
    """
    # Invert incoming: state -> {(label, pred)} => pred -> [(label, state)]
    forward = defaultdict(list)
    for state, arcs in merged_incoming.items():
        for label, pred in arcs:
            forward[pred].append((label, state))

    start_states = set(peek_state.dfa.start())

    # Extract per-symbol Q/R stops from decomp
    q_stops = {}
    r_stops = {}
    for y, d in peek_state.decomp.items():
        q_stops[y] = set(d.quotient)
        r_stops[y] = set(d.remainder)

    return ForwardGraph(
        forward=dict(forward),
        start_states=start_states,
        q_stops=q_stops,
        r_stops=r_stops,
    )


async def unified_beam_search(
    graph: ForwardGraph,
    lm,
    beam_width: int = 64,
    max_depth: int = 30,
    logp_floor: float = -40.0,
) -> Dict[str, float]:
    """Run one beam search over the shared peekaboo DFA, collecting
    per-symbol results when Q/R stops are hit.

    Parameters
    ----------
    graph : ForwardGraph
        The shared forward graph.
    lm : CachedByteLM
        Autoregressive byte-level LM.
    beam_width : int
        Max beam entries per depth level.
    max_depth : int
        Maximum search depth.
    logp_floor : float
        Prune entries with logp below this.

    Returns
    -------
    Dict[str, float]
        {symbol: logp(paths)} for each reachable symbol. These are
        unnormalized joint logps (not conditional).
    """
    # Precompute which states are Q stops for any symbol, to avoid expansion
    all_q_stops = set()
    for states in graph.q_stops.values():
        all_q_stops |= states

    # Collect per-symbol log-probability contributions
    symbol_logps = defaultdict(list)  # symbol -> [logp, ...]

    # Beam: list of (logp, state, ctx_tuple)
    beam = [(0.0, s, ()) for s in graph.start_states]

    for depth in range(max_depth + 1):
        if not beam:
            break

        # Record Q/R contributions from current beam
        for logp, state, ctx in beam:
            for y, stops in graph.q_stops.items():
                if state in stops:
                    symbol_logps[y].append(logp)
            for y, stops in graph.r_stops.items():
                if state in stops:
                    # R stops need EOS logp
                    eos_arr = await lm.logp_next_for(ctx)
                    symbol_logps[y].append(logp + eos_arr[EOS_IDX])

        if depth >= max_depth:
            break

        # Expand beam (Q stops are leaves -- don't expand them)
        candidates = []
        for logp, state, ctx in beam:
            if state in all_q_stops:
                continue  # Q stops have no outgoing arcs by construction

            forward_arcs = graph.forward.get(state, [])
            if not forward_arcs:
                continue

            lm_arr = await lm.logp_next_for(ctx)
            for label, succ in forward_arcs:
                if label == EPSILON:
                    candidates.append((logp, succ, ctx))
                else:
                    try:
                        byte_val = int(label)
                    except ValueError:
                        continue
                    if byte_val >= len(lm_arr):
                        continue
                    next_logp = logp + lm_arr[byte_val]
                    if next_logp < logp_floor:
                        continue
                    candidates.append((next_logp, succ, ctx + (byte_val,)))

        # Beam pruning: keep top-K by logp
        if len(candidates) > beam_width:
            candidates.sort(key=lambda x: -x[0])
            candidates = candidates[:beam_width]

        beam = candidates

    # Aggregate per symbol
    result = {}
    for y, logps in symbol_logps.items():
        if logps:
            result[y] = logsumexp(logps)
    return result


# =============================================================================
# Component 3: Adaptive Symbol Budget
# =============================================================================


def allocate_symbol_budgets(
    symbols: List[str],
    lm_logps: np.ndarray,
    total_budget: int = 2000,
    min_per_sym: int = 5,
    max_per_sym: int = 200,
) -> Dict[str, int]:
    """Allocate per-symbol path budgets proportional to LM probability.

    High-probability symbols get more paths; low-probability ones get fewer.

    Parameters
    ----------
    symbols : list of str
        Candidate symbols (e.g., byte strings like '65').
    lm_logps : np.ndarray
        Log-probability array from the LM (256 or 257 entries).
    total_budget : int
        Total path budget across all symbols.
    min_per_sym : int
        Minimum paths per symbol.
    max_per_sym : int
        Maximum paths per symbol.

    Returns
    -------
    Dict[str, int]
        {symbol: max_paths} allocations.
    """
    if not symbols:
        return {}

    # Get LM logps for each symbol
    sym_logps = []
    for z in symbols:
        try:
            v = int(z)
            sym_logps.append(float(lm_logps[v]) if v < len(lm_logps) else NEG_INF)
        except (ValueError, IndexError):
            sym_logps.append(NEG_INF)

    sym_logps = np.array(sym_logps)

    # Convert to probabilities for proportional allocation
    max_logp = sym_logps.max()
    if max_logp <= NEG_INF / 2:
        # All symbols equally unlikely
        per_sym = max(min_per_sym, total_budget // len(symbols))
        return {z: min(per_sym, max_per_sym) for z in symbols}

    # Softmax to get proportions
    shifted = sym_logps - max_logp
    probs = np.exp(shifted)
    probs /= probs.sum()

    budgets = {}
    for i, z in enumerate(symbols):
        raw = int(probs[i] * total_budget)
        budgets[z] = max(min_per_sym, min(raw, max_per_sym))

    return budgets


# =============================================================================
# Component 4: JSD Evaluation Framework
# =============================================================================


@dataclass
class EvalResult:
    """Result of evaluating beam search accuracy at one step."""

    step: int
    beam_width: int
    jsd: float  # Jensen-Shannon divergence (nats)
    kl_exact_approx: float  # KL(exact || approx)
    top1_match: bool  # Does top-1 symbol match?
    top5_overlap: int  # How many of exact top-5 are in approx top-5?
    mass_captured: float  # Total probability mass of approx distribution
    exact_time_ms: float
    approx_time_ms: float
    n_symbols_exact: int
    n_symbols_approx: int


def compute_jsd(
    p_logps: Dict[str, float],
    q_logps: Dict[str, float],
) -> float:
    """Compute Jensen-Shannon divergence JSD(P || Q) in nats.

    JSD(P || Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q).

    Both inputs are log-probability dicts (need not be normalized; this
    function normalizes them).

    Returns
    -------
    float
        JSD in nats (>= 0). Returns 0.0 if both distributions are empty.
    """
    if not p_logps and not q_logps:
        return 0.0

    # Union of all symbols
    all_syms = set(p_logps.keys()) | set(q_logps.keys())
    if not all_syms:
        return 0.0

    syms = sorted(all_syms)

    # Build raw log-probability vectors
    p_raw = np.array([p_logps.get(s, NEG_INF) for s in syms])
    q_raw = np.array([q_logps.get(s, NEG_INF) for s in syms])

    # Normalize
    p_logZ = logsumexp(p_raw)
    q_logZ = logsumexp(q_raw)

    if p_logZ <= NEG_INF / 2 and q_logZ <= NEG_INF / 2:
        return 0.0

    p_log = p_raw - p_logZ
    q_log = q_raw - q_logZ

    # M = 0.5 * (P + Q) in log space
    m_log = np.logaddexp(p_log + np.log(0.5), q_log + np.log(0.5))

    # KL(P || M) = sum_i P_i * (log P_i - log M_i)
    p_probs = np.exp(p_log)
    q_probs = np.exp(q_log)

    kl_pm = 0.0
    kl_qm = 0.0
    for i in range(len(syms)):
        if p_probs[i] > 0:
            kl_pm += p_probs[i] * (p_log[i] - m_log[i])
        if q_probs[i] > 0:
            kl_qm += q_probs[i] * (q_log[i] - m_log[i])

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return max(0.0, jsd)  # Clamp numerical noise


def compute_kl(
    p_logps: Dict[str, float],
    q_logps: Dict[str, float],
) -> float:
    """Compute KL(P || Q) in nats. Returns inf if Q has zero mass where P doesn't."""
    if not p_logps:
        return 0.0

    all_syms = set(p_logps.keys()) | set(q_logps.keys())
    syms = sorted(all_syms)

    p_raw = np.array([p_logps.get(s, NEG_INF) for s in syms])
    q_raw = np.array([q_logps.get(s, NEG_INF) for s in syms])

    p_logZ = logsumexp(p_raw)
    q_logZ = logsumexp(q_raw)

    if p_logZ <= NEG_INF / 2:
        return 0.0

    p_log = p_raw - p_logZ
    q_log = q_raw - q_logZ
    p_probs = np.exp(p_log)

    kl = 0.0
    for i in range(len(syms)):
        if p_probs[i] > 1e-30:
            if q_log[i] <= NEG_INF / 2:
                return float("inf")
            kl += p_probs[i] * (p_log[i] - q_log[i])

    return max(0.0, kl)


async def run_jsd_evaluation(
    compute_exact_fn,
    compute_approx_fn,
    output: Tuple[str, ...],
    beam_widths: List[int] = None,
    max_steps: int = 10,
) -> Dict[int, List[EvalResult]]:
    """Run JSD evaluation comparing exact and approximate distributions.

    Parameters
    ----------
    compute_exact_fn : async callable(target) -> Dict[str, float]
        Function that returns exact logp distribution for a target.
    compute_approx_fn : async callable(target, beam_width) -> Dict[str, float]
        Function that returns approximate logp distribution.
    output : tuple of str
        Output sequence to iterate over.
    beam_widths : list of int
        Beam widths to evaluate.
    max_steps : int
        Number of output positions to evaluate.

    Returns
    -------
    Dict[int, List[EvalResult]]
        {beam_width: [EvalResult per step]}
    """
    if beam_widths is None:
        beam_widths = [8, 16, 32, 64, 128, 256]

    n_steps = min(max_steps, len(output))
    results = {bw: [] for bw in beam_widths}

    for step in range(n_steps):
        target = output[:step]

        # Compute exact distribution
        t0 = time.perf_counter()
        exact_dist = await compute_exact_fn(target)
        exact_time = (time.perf_counter() - t0) * 1000

        # For each beam width, compute approximate
        for bw in beam_widths:
            t0 = time.perf_counter()
            approx_dist = await compute_approx_fn(target, bw)
            approx_time = (time.perf_counter() - t0) * 1000

            # Compute metrics
            jsd = compute_jsd(exact_dist, approx_dist)
            kl = compute_kl(exact_dist, approx_dist)

            # Top-K analysis
            exact_sorted = sorted(exact_dist.items(), key=lambda x: -x[1])
            approx_sorted = sorted(approx_dist.items(), key=lambda x: -x[1])

            exact_top1 = exact_sorted[0][0] if exact_sorted else None
            approx_top1 = approx_sorted[0][0] if approx_sorted else None
            top1_match = exact_top1 == approx_top1

            exact_top5 = {s for s, _ in exact_sorted[:5]}
            approx_top5 = {s for s, _ in approx_sorted[:5]}
            top5_overlap = len(exact_top5 & approx_top5)

            # Mass captured
            if approx_dist:
                approx_vals = np.array(list(approx_dist.values()))
                mass = float(np.exp(logsumexp(approx_vals)))
            else:
                mass = 0.0

            results[bw].append(
                EvalResult(
                    step=step,
                    beam_width=bw,
                    jsd=jsd,
                    kl_exact_approx=kl,
                    top1_match=top1_match,
                    top5_overlap=top5_overlap,
                    mass_captured=mass,
                    exact_time_ms=exact_time,
                    approx_time_ms=approx_time,
                    n_symbols_exact=len(exact_dist),
                    n_symbols_approx=len(approx_dist),
                )
            )

    return results


def generate_jsd_report(
    eval_results: Dict[int, List[EvalResult]],
    output_path: str = "reports/jsd_beam_search_evaluation.json",
):
    """Generate a JSON report from JSD evaluation results.

    Parameters
    ----------
    eval_results : Dict[int, List[EvalResult]]
        {beam_width: [EvalResult per step]}
    output_path : str
        Where to write the report.
    """
    report = {"beam_widths": {}}

    for bw, results in eval_results.items():
        if not results:
            continue
        jsds = [r.jsd for r in results]
        kls = [r.kl_exact_approx for r in results if r.kl_exact_approx < float("inf")]
        top1_acc = sum(r.top1_match for r in results) / len(results)
        mean_top5 = np.mean([r.top5_overlap for r in results])
        mean_mass = np.mean([r.mass_captured for r in results])
        exact_times = [r.exact_time_ms for r in results]
        approx_times = [r.approx_time_ms for r in results]
        mean_speedup = (
            np.mean(exact_times) / np.mean(approx_times)
            if np.mean(approx_times) > 0
            else float("inf")
        )

        report["beam_widths"][str(bw)] = {
            "mean_jsd": float(np.mean(jsds)),
            "max_jsd": float(np.max(jsds)),
            "median_jsd": float(np.median(jsds)),
            "mean_kl": float(np.mean(kls)) if kls else None,
            "top1_accuracy": top1_acc,
            "mean_top5_overlap": float(mean_top5),
            "mean_mass_captured": float(mean_mass),
            "mean_speedup": float(mean_speedup),
            "mean_exact_time_ms": float(np.mean(exact_times)),
            "mean_approx_time_ms": float(np.mean(approx_times)),
            "n_steps": len(results),
            "per_step": [
                {
                    "step": r.step,
                    "jsd": r.jsd,
                    "kl": r.kl_exact_approx,
                    "top1_match": r.top1_match,
                    "top5_overlap": r.top5_overlap,
                    "mass_captured": r.mass_captured,
                    "approx_time_ms": r.approx_time_ms,
                    "n_symbols_exact": r.n_symbols_exact,
                    "n_symbols_approx": r.n_symbols_approx,
                }
                for r in results
            ],
        }

    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
