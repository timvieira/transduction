"""Minimal PTB benchmark (no LM): path counts and uniform-weight distributions.

Uses RustPeekaboo (from rust_bridge.py) for per-symbol decomposition. Instead of
scoring paths with a language model, assigns uniform weight to each accepting path.
The resulting distribution reflects FST structure only.

No genlm / torch dependency required.

Run:
    python -m transduction.benchmarking.run_ptb_benchmark_nolm
    python -m transduction.benchmarking.run_ptb_benchmark_nolm --steps 5 --paragraphs 3
"""

import json
import numpy as np
import time
import tqdm
from scipy.special import logsumexp
from typing import Dict, Tuple

from transduction.benchmarking.fsts.ptb_pynini import build_ptb_fst_pynini
from transduction.rust_bridge import RustPeekaboo, RustDecomp
from transduction.benchmarking.config.constants import NEG_INF
from transduction.benchmarking.decomp.path_enumeration import enumerate_fsa_paths_bfs
from transduction.benchmarking.fst_utils import load_paragraphs


def compute_full_distribution(
    peekaboo: RustPeekaboo,
    fst,
    target: Tuple[str, ...],
    max_depth: int = 30,
    max_paths: int = 100,
) -> Dict[str, float]:
    """Compute p(z | target) for every reachable symbol z using uniform path weights.

    Each accepting path gets equal weight (logp = 0). The distribution reflects
    the number of FSA paths per symbol, not LM probabilities.

    Returns normalized log-probabilities {symbol: logp}.
    """
    uniform = np.full(257, 0.0)

    # 1. Count paths for the target prefix
    target_decomp = RustDecomp(fst, target)
    n_q = sum(
        1
        for _ in enumerate_fsa_paths_bfs(
            target_decomp.quotient, uniform, max_depth=max_depth, max_paths=max_paths
        )
    )
    n_r = sum(
        1
        for _ in enumerate_fsa_paths_bfs(
            target_decomp.remainder, uniform, max_depth=max_depth, max_paths=max_paths
        )
    )
    n_target = n_q + n_r
    if n_target == 0:
        return {}
    logp_target = np.log(n_target)

    # EOS: remainder paths count
    eos_cond = np.log(n_r) - logp_target if n_r > 0 else NEG_INF

    # 2. Per-symbol decomposition via RustPeekaboo
    decomps = peekaboo(target)  # Dict[symbol, PrecoverDecomp]
    scores = {}
    for z in sorted(decomps):
        d = decomps[z]
        Q_z, R_z = d.quotient, d.remainder
        if not Q_z.states and not R_z.states:
            continue

        nq = sum(
            1
            for _ in enumerate_fsa_paths_bfs(
                Q_z, uniform, max_depth=max_depth, max_paths=max_paths
            )
        )
        nr = sum(
            1
            for _ in enumerate_fsa_paths_bfs(
                R_z, uniform, max_depth=max_depth, max_paths=max_paths
            )
        )
        n_total = nq + nr
        if n_total == 0:
            continue
        scores[z] = np.log(n_total) - logp_target

    # 3. Normalize (including EOS)
    all_scores = dict(scores)
    if eos_cond > NEG_INF:
        all_scores["EOS"] = eos_cond

    if not all_scores:
        return {}
    vals = np.array(list(all_scores.values()))
    logZ = logsumexp(vals)
    return {k: float(v - logZ) for k, v in all_scores.items()}


def run(args):
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
            dist = compute_full_distribution(
                peekaboo,
                fst,
                target,
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
            pbar.set_postfix(syms=len(dist), t=f"{elapsed:.1f}s")

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
        print(f"  mean time: {np.mean(times):.2f}s/step")

    # Save results
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Minimal PTB benchmark (no LM)")
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
    parser.add_argument("--output", "-o", default="ptb_nolm_benchmark.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
