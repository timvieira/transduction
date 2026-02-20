"""
Benchmark: FactoredDecomp vs NonrecursiveDFADecomp.

Measures wall-clock time for decomposition on example FSTs with
various target lengths.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transduction.util import set_memory_limit
set_memory_limit(4)

import time
from transduction import examples, EPSILON
from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp
from transduction.factored_decompose import FactoredDecomp


def generate_targets(fst, max_len):
    """Generate all target strings up to max_len."""
    target_alpha = sorted(fst.B - {EPSILON})
    targets = [()]
    for length in range(1, max_len + 1):
        new = []
        for t in targets:
            if len(t) == length - 1:
                for sym in target_alpha:
                    new.append(t + (sym,))
        targets.extend(new)
    return [t for t in targets if t]


def benchmark_nonrecursive(fst, targets):
    """Time NonrecursiveDFADecomp on all targets."""
    t0 = time.perf_counter()
    for target in targets:
        try:
            d = NonrecursiveDFADecomp(fst, target)
            _ = d.quotient
            _ = d.remainder
        except Exception:
            pass
    return time.perf_counter() - t0


def benchmark_factored(fst, targets):
    """Time FactoredDecomp on all targets."""
    t0 = time.perf_counter()
    for target in targets:
        try:
            d = FactoredDecomp(fst, target)
            _ = d.quotient
            _ = d.remainder
        except Exception:
            pass
    return time.perf_counter() - t0


def benchmark_factored_incremental(fst, targets_by_prefix):
    """Time FactoredDecomp using >> for incremental extension."""
    t0 = time.perf_counter()
    target_alpha = sorted(fst.B - {EPSILON})
    # Build by extending each symbol from the root
    root = FactoredDecomp(fst, ())
    _ = root.quotient
    _ = root.remainder
    for y in target_alpha:
        s1 = FactoredDecomp(fst, (y,))
        _ = s1.quotient
        _ = s1.remainder
        for y2 in target_alpha:
            s2 = FactoredDecomp(fst, (y, y2))
            _ = s2.quotient
            _ = s2.remainder
            for y3 in target_alpha:
                s3 = FactoredDecomp(fst, (y, y2, y3))
                _ = s3.quotient
                _ = s3.remainder
    return time.perf_counter() - t0


def main():
    print("FactoredDecomp vs NonrecursiveDFADecomp Benchmark")
    print("=" * 80)

    results = []

    test_cases = [
        ('replace(xyz)', lambda: examples.replace([('a', 'x'), ('b', 'y'), ('c', 'z')]), 4),
        ('delete_b', examples.delete_b, 4),
        ('samuel_example', examples.samuel_example, 4),
        ('doom(K=3)', lambda: examples.doom({'a', 'b'}, 3), 4),
        ('mystery1', examples.mystery1, 4),
        ('mystery7', examples.mystery7, 4),
        ('newspeak2', examples.newspeak2, 3),
        ('anbn', examples.anbn, 4),
        ('backticks_to_quote', examples.backticks_to_quote, 3),
        ('parity_copy', examples.parity_copy, 3),
    ]

    for name, fst_fn, max_len in test_cases:
        fst = fst_fn()
        targets = generate_targets(fst, max_len)

        t_std = benchmark_nonrecursive(fst, targets)
        t_fac = benchmark_factored(fst, targets)
        speedup = t_std / t_fac if t_fac > 0 else float('inf')

        results.append((name, len(targets), t_std, t_fac, speedup))
        print(f"  {name:<25s}  targets={len(targets):>5d}  "
              f"std={t_std:.3f}s  fac={t_fac:.3f}s  "
              f"speedup={speedup:.2f}x")

    print()
    print("=" * 80)
    print(f"{'FST':<25s}  {'Targets':>7s}  {'Standard':>9s}  {'Factored':>9s}  {'Speedup':>8s}")
    print("-" * 80)
    for name, n_targets, t_std, t_fac, speedup in results:
        print(f"{name:<25s}  {n_targets:>7d}  {t_std:>8.3f}s  {t_fac:>8.3f}s  {speedup:>7.2f}x")

    # Geometric mean speedup
    from math import exp, log
    geo_mean = exp(sum(log(s) for _, _, _, _, s in results) / len(results))
    print("-" * 80)
    print(f"{'Geometric mean':<25s}  {'':>7s}  {'':>9s}  {'':>9s}  {geo_mean:>7.2f}x")


if __name__ == '__main__':
    main()
