"""
LM calls are currently very slow, so this benchmark is not very useful.

Minimal PTB benchmark: full next-symbol distributions over 10 wikitext paragraphs.

Uses RustPeekaboo (from rust_bridge.py) for per-symbol decomposition and
CachedByteLM for autoregressive scoring. Computes the exhaustive conditional
distribution p(z | target) at every step.

Run:
    python -m transduction.benchmarking.run_ptb_benchmark
    python -m transduction.benchmarking.run_ptb_benchmark --steps 5 --paragraphs 3
"""

import asyncio
import json
import numpy as np
import time
import tqdm
from scipy.special import logsumexp
from typing import Dict, Tuple

from transduction.benchmarking.fsts.ptb_pynini import build_ptb_fst_pynini
from transduction.rust_bridge import RustPeekaboo, RustDecomp
from transduction.benchmarking.config.constants import NEG_INF
from transduction.benchmarking.lm.cached_byte_lm import CachedByteLM
from transduction.benchmarking.decomp.path_enumeration import enumerate_fsa_paths_bfs
from transduction.benchmarking.fst_utils import load_paragraphs


async def compute_full_distribution(
    peekaboo: RustPeekaboo,
    fst,
    target: Tuple[str, ...],
    lm: CachedByteLM,
    max_depth: int = 30,
    max_paths: int = 100,
) -> Dict[str, float]:
    """Compute p(z | target) for every reachable symbol z.

    Uses RustPeekaboo for per-symbol decomposition and CachedByteLM for
    autoregressive path scoring.

    Returns normalized log-probabilities {symbol: logp}.
    """
    uniform = np.full(257, 0.0)

    # 1. Decompose target prefix to get logp(target)
    target_decomp = RustDecomp(fst, target)
    q_paths = [
        p
        for p, _, _ in enumerate_fsa_paths_bfs(
            target_decomp.quotient, uniform, max_depth=max_depth, max_paths=max_paths
        )
    ]
    r_paths = [
        p
        for p, _, _ in enumerate_fsa_paths_bfs(
            target_decomp.remainder, uniform, max_depth=max_depth, max_paths=max_paths
        )
    ]

    q_scores = [await lm.score_path(p) for p in q_paths]
    r_scores = [await lm.score_path_with_eos(p) for p in r_paths]
    all_target = [s for s in q_scores + r_scores if s > NEG_INF]
    if not all_target:
        return {}
    logp_target = logsumexp(all_target)

    # EOS conditional
    r_logp = logsumexp(r_scores) if r_scores else NEG_INF
    eos_cond = r_logp - logp_target if r_logp > NEG_INF else NEG_INF

    # 2. Per-symbol decomposition via RustPeekaboo
    decomps = peekaboo(target)  # Dict[symbol, PrecoverDecomp]
    scores = {}
    for z in sorted(decomps):
        d = decomps[z]
        Q_z, R_z = d.quotient, d.remainder
        if not Q_z.states and not R_z.states:
            continue

        qp = [
            p
            for p, _, _ in enumerate_fsa_paths_bfs(
                Q_z, uniform, max_depth=max_depth, max_paths=max_paths
            )
        ]
        rp = [
            p
            for p, _, _ in enumerate_fsa_paths_bfs(
                R_z, uniform, max_depth=max_depth, max_paths=max_paths
            )
        ]
        if not qp and not rp:
            continue

        qs = [await lm.score_path(p) for p in qp]
        rs = [await lm.score_path_with_eos(p) for p in rp]
        all_s = [s for s in qs + rs if s > NEG_INF]
        if not all_s:
            continue
        scores[z] = logsumexp(all_s) - logp_target

    # 3. Normalize (including EOS)
    all_scores = dict(scores)
    if eos_cond > NEG_INF:
        all_scores["EOS"] = eos_cond

    if not all_scores:
        return {}
    vals = np.array(list(all_scores.values()))
    logZ = logsumexp(vals)
    return {k: float(v - logZ) for k, v in all_scores.items()}


async def run(args):
    # Build FST
    print("Building PTB FST...")
    t0 = time.perf_counter()
    fst = build_ptb_fst_pynini()
    print(f"  {len(fst.states)} states, {time.perf_counter()-t0:.1f}s")

    # Build RustPeekaboo (builds Rust FST once)
    print("Building RustPeekaboo...")
    t0 = time.perf_counter()
    peekaboo = RustPeekaboo(fst)
    n_symbols = len(peekaboo.target_alphabet)
    print(f"  {n_symbols} output symbols, {time.perf_counter()-t0:.1f}s")

    # Load paragraphs
    print(f"Loading {args.paragraphs} wikitext paragraphs...")
    paragraphs = load_paragraphs(fst, n=args.paragraphs)
    print(f"  loaded {len(paragraphs)} paragraphs")
    for i, p in enumerate(paragraphs):
        print(f"  [{i}] {len(p['output'])} symbols: {p['text'][:60]}...")

    # Initialize LM
    print("Initializing LM...")
    lm = await CachedByteLM.create(
        model_name="gpt2",
        K=args.beam_k,
        cache_size=50000,
    )
    print("  ready")

    # Run benchmark
    all_results = []
    for para_idx, para in enumerate(paragraphs):
        output = para["output"]
        n_steps = min(args.steps, len(output)) if args.steps else len(output)
        print(f"\nParagraph {para_idx}: {n_steps} steps, {len(output)} total symbols")
        print(f"  {para['text'][:80]}...")

        para_results = []
        pbar = tqdm.tqdm(range(n_steps), desc=f"  para {para_idx}")
        for step in pbar:
            target = output[:step]
            t0 = time.perf_counter()
            dist = await compute_full_distribution(
                peekaboo,
                fst,
                target,
                lm,
                max_depth=args.max_depth,
                max_paths=args.max_paths,
            )
            elapsed = time.perf_counter() - t0

            actual = output[step] if step < len(output) else None
            actual_logp = dist.get(actual, NEG_INF) if actual else NEG_INF
            top_sym = max(dist, key=dist.get) if dist else None

            para_results.append(
                {
                    "step": step,
                    "target": list(target),
                    "distribution": dist,
                    "n_symbols": len(dist),
                    "time_s": elapsed,
                    "actual_next": actual,
                    "actual_logp": actual_logp,
                    "top_symbol": top_sym,
                    "top_logp": dist[top_sym] if top_sym else NEG_INF,
                }
            )
            pbar.set_postfix(
                syms=len(dist),
                t=f"{elapsed:.1f}s",
                actual_p=f"{np.exp(actual_logp):.3f}" if actual_logp > NEG_INF else "0",
            )

        all_results.append(
            {
                "paragraph_idx": para_idx,
                "text": para["text"],
                "output": list(output),
                "steps": para_results,
            }
        )

        # Summary for this paragraph
        times = [r["time_s"] for r in para_results]
        actual_logps = [
            r["actual_logp"] for r in para_results if r["actual_logp"] > NEG_INF
        ]
        top1_correct = sum(
            1 for r in para_results if r["actual_next"] == r["top_symbol"]
        )
        print(f"  mean time: {np.mean(times):.2f}s/step")
        if actual_logps:
            print(f"  mean logp(actual): {np.mean(actual_logps):.3f}")
        print(f"  top-1 accuracy: {top1_correct}/{len(para_results)}")

    # Save results
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Cache stats
    print(
        f"LM cache: {lm.beam_hits} hits, {lm.beam_misses} misses, {len(lm._beams)} beams"
    )
    await lm.cleanup()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Minimal PTB benchmark")
    parser.add_argument("--paragraphs", "-p", type=int, default=10)
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=None,
        help="Steps per paragraph (default: all)",
    )
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--max-paths", type=int, default=100)
    parser.add_argument(
        "--beam-k", type=int, default=5, help="LM beam width (genlm ByteBeamState K)"
    )
    parser.add_argument("--output", "-o", default="ptb_full_benchmark.json")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
