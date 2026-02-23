#!/usr/bin/env python3
"""Run all benchmark scripts and optionally regenerate plots.

Usage:
    python reports/run_all.py                  # run all benchmarks (full)
    python reports/run_all.py --quick           # run all benchmarks (fast)
    python reports/run_all.py --only vec run    # run only bench_vectorization + run_benchmarks
    python reports/run_all.py --list            # list available benchmarks
    python reports/run_all.py --quick --no-plots  # skip plot generation

Available benchmark names (short aliases):
    vec          bench_vectorization.py        BPE vocab scaling (FusedLM, CharacterBeam)
    run          run_benchmarks.py             PTB + BPE end-to-end LM comparison
    gen          bench_generalized_beam.py     GeneralizedBeam on BPE + PTB
    trie         bench_trie_dispatch.py        Trie dispatch vs standard decomposition
    bpe_ptb      bpe_ptb_benchmark.py          Backend comparison (Standard/Pynini/Rust)
"""

import argparse
import os
import subprocess
import sys
import time

REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(REPORTS_DIR)

# (alias, script filename, description)
BENCHMARKS = [
    ('vec',     'bench_vectorization.py',    'BPE vocab scaling (FusedLM, CharacterBeam)'),
    ('run',     'run_benchmarks.py',         'PTB + BPE end-to-end LM comparison'),
    ('gen',     'bench_generalized_beam.py', 'GeneralizedBeam on BPE + PTB'),
    ('trie',    'bench_trie_dispatch.py',    'Trie dispatch decomposition'),
    ('bpe_ptb', 'bpe_ptb_benchmark.py',      'Backend comparison (Standard/Pynini/Rust)'),
]

ALIAS_MAP = {alias: script for alias, script, _ in BENCHMARKS}


def list_benchmarks():
    print("Available benchmarks:\n")
    for alias, script, desc in BENCHMARKS:
        print(f"  {alias:<10s}  {script:<32s}  {desc}")
    print(f"\nAll scripts support --quick for faster runs.")


def run_script(script, quick=False):
    """Run a benchmark script as a subprocess. Returns (elapsed, returncode)."""
    script_path = os.path.join(REPORTS_DIR, script)
    cmd = [sys.executable, script_path]
    if quick:
        cmd.append('--quick')

    print(f"\n{'='*70}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*70}\n", flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0

    return elapsed, result.returncode


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--quick', action='store_true',
                        help='Pass --quick to all benchmark scripts')
    parser.add_argument('--only', nargs='+', metavar='NAME',
                        help='Run only the specified benchmarks (use aliases)')
    parser.add_argument('--list', action='store_true',
                        help='List available benchmarks and exit')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation at the end')
    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    # Determine which benchmarks to run
    if args.only:
        scripts = []
        for name in args.only:
            if name in ALIAS_MAP:
                scripts.append(ALIAS_MAP[name])
            else:
                # Try prefix match
                matches = [a for a in ALIAS_MAP if a.startswith(name)]
                if len(matches) == 1:
                    scripts.append(ALIAS_MAP[matches[0]])
                else:
                    print(f"Unknown benchmark: {name!r}")
                    print(f"Available: {', '.join(ALIAS_MAP.keys())}")
                    sys.exit(1)
    else:
        scripts = [script for _, script, _ in BENCHMARKS]

    mode = "QUICK" if args.quick else "FULL"
    print(f"{'='*70}")
    print(f"  BENCHMARK SUITE ({mode} mode)")
    print(f"  Running {len(scripts)} benchmark(s)")
    print(f"{'='*70}")

    results = []
    total_t0 = time.perf_counter()

    for script in scripts:
        elapsed, rc = run_script(script, quick=args.quick)
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        results.append((script, elapsed, status))
        print(f"\n  >> {script}: {status} in {elapsed:.1f}s")

    # Optionally regenerate plots
    if not args.no_plots:
        plots_script = os.path.join(REPORTS_DIR, 'dashboard_plots.py')
        if os.path.exists(plots_script):
            print(f"\n{'='*70}")
            print(f"  Regenerating plots...")
            print(f"{'='*70}\n", flush=True)
            t0 = time.perf_counter()
            rc = subprocess.run([sys.executable, plots_script], cwd=PROJECT_ROOT).returncode
            elapsed = time.perf_counter() - t0
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            results.append(('dashboard_plots.py', elapsed, status))

    total_elapsed = time.perf_counter() - total_t0

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY ({mode} mode)")
    print(f"{'='*70}\n")
    print(f"  {'Script':<36s}  {'Time':>8s}  {'Status'}")
    print(f"  {'-'*60}")
    for script, elapsed, status in results:
        print(f"  {script:<36s}  {elapsed:>7.1f}s  {status}")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<36s}  {total_elapsed:>7.1f}s")

    failed = [s for s, _, st in results if not st.startswith("OK")]
    if failed:
        print(f"\n  WARNING: {len(failed)} benchmark(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n  All benchmarks completed successfully.")


if __name__ == '__main__':
    main()
