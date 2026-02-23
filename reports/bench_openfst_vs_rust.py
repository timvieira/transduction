"""
Benchmark: OpenFST C++ backend vs Rust backend.

Compares three decomposition modes:
  1. One-shot decomposition (OpenFstDecomp vs RustDecomp)
  2. Incremental dirty peekaboo (OpenFstDirtyPeekaboo vs RustDirtyPeekaboo)
  3. Lazy DFA classify/arcs (OpenFstLazyPeekabooDFA vs RustLazyPeekabooDFA)

Tests on synthetic BPE-like FSTs at various vocab sizes and on standard
example FSTs.

Usage:
    python reports/bench_openfst_vs_rust.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transduction.util import set_memory_limit
set_memory_limit(4)

import gc
import json
import time
import resource
from collections import defaultdict

import numpy as np

from transduction import examples
from transduction.fst import FST
from transduction.fsa import EPSILON

# ---- Backends ----
from transduction.rust_bridge import RustDecomp, RustDirtyPeekaboo, to_rust_fst
from transduction.openfst_bridge import (
    OpenFstDecomp, OpenFstDirtyPeekaboo, OpenFstLazyPeekabooDFA
)
import transduction_core

OUTFILE = os.path.join(os.path.dirname(__file__), 'bench_openfst_vs_rust.json')
REPORT = os.path.join(os.path.dirname(__file__), 'OPENFST_VS_RUST.md')


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# =====================================================================
# Section 1: One-shot decomposition
# =====================================================================

def bench_oneshot(fst, targets, n_runs=3, label=""):
    """Benchmark one-shot decomposition on a list of targets."""
    results = {}

    for backend_name, DecompClass in [("rust", RustDecomp), ("openfst", OpenFstDecomp)]:
        times = []
        for _ in range(n_runs):
            gc.collect()
            t0 = time.perf_counter()
            for target in targets:
                d = DecompClass(fst, target)
                _ = d.quotient
                _ = d.remainder
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        avg_ms = np.median(times) * 1000
        results[backend_name] = avg_ms

    speedup = results["rust"] / results["openfst"] if results["openfst"] > 0 else float('inf')
    print(f"  One-shot {label}: rust={results['rust']:.1f}ms  openfst={results['openfst']:.1f}ms  "
          f"speedup={speedup:.2f}x")
    return results


# =====================================================================
# Section 2: Incremental dirty peekaboo
# =====================================================================

def bench_dirty_peekaboo(fst, target_sequence, n_runs=3, label=""):
    """Benchmark incremental dirty peekaboo over a target sequence."""
    results = {}

    for backend_name, PeekabooClass in [("rust", RustDirtyPeekaboo), ("openfst", OpenFstDirtyPeekaboo)]:
        times = []
        for _ in range(n_runs):
            gc.collect()
            t0 = time.perf_counter()
            state = PeekabooClass(fst, ())
            for y in target_sequence:
                children = state.decompose_next()
                state = children[y]
                _ = state.quotient
                _ = state.remainder
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        avg_ms = np.median(times) * 1000
        results[backend_name] = avg_ms

    speedup = results["rust"] / results["openfst"] if results["openfst"] > 0 else float('inf')
    print(f"  Dirty peekaboo {label}: rust={results['rust']:.1f}ms  openfst={results['openfst']:.1f}ms  "
          f"speedup={speedup:.2f}x")
    return results


# =====================================================================
# Section 3: Lazy DFA (classify + arcs)
# =====================================================================

def bench_lazy_dfa(fst, target_sequence, n_runs=3, label=""):
    """Benchmark lazy DFA new_step + full BFS expansion at each step."""
    results = {}

    for backend_name in ["rust", "openfst"]:
        times = []
        for _ in range(n_runs):
            gc.collect()

            if backend_name == "rust":
                rust_fst, sym_map, state_map = to_rust_fst(fst)
                helper = transduction_core.RustLazyPeekabooDFA(rust_fst)
            else:
                helper = OpenFstLazyPeekabooDFA(fst)
                sym_map = helper._sym_map

            target_u32 = []
            t0 = time.perf_counter()
            for y in target_sequence:
                target_u32.append(sym_map(y))
                helper.new_step(list(target_u32))

                if backend_name == "openfst":
                    # Layer-by-layer BFS with batch expansion
                    frontier = list(helper.start_ids())
                    visited = set(frontier)
                    while frontier:
                        batch = helper.expand_batch(frontier)
                        frontier = []
                        for sid, cr, arcs in batch:
                            if cr.quotient_sym is not None:
                                continue
                            for lbl, dst in arcs:
                                if dst not in visited:
                                    visited.add(dst)
                                    frontier.append(dst)
                else:
                    # Rust: per-state BFS (PyO3 overhead is minimal)
                    from collections import deque
                    visited = set()
                    q = deque(helper.start_ids())
                    visited.update(q)
                    while q:
                        sid = q.popleft()
                        cr = helper.classify(sid)
                        if cr.quotient_sym is not None:
                            continue
                        for lbl, dst in helper.arcs(sid):
                            if dst not in visited:
                                visited.add(dst)
                                q.append(dst)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg_ms = np.median(times) * 1000
        results[backend_name] = avg_ms

    speedup = results["rust"] / results["openfst"] if results["openfst"] > 0 else float('inf')
    print(f"  Lazy DFA {label}: rust={results['rust']:.1f}ms  openfst={results['openfst']:.1f}ms  "
          f"speedup={speedup:.2f}x")
    return results


# =====================================================================
# BPE vocab scaling
# =====================================================================

def bench_bpe_scaling():
    """Benchmark across BPE-like FSTs at various vocab sizes."""
    print("\n=== BPE Vocab Scaling ===")
    VOCAB_SIZES = [50, 100, 200, 500, 1000, 2000]
    ALPHABET = tuple("abcdefghij")
    MAX_LEN = 5
    N_RUNS = 3

    all_results = []

    for V in VOCAB_SIZES:
        print(f"\nVocab size = {V}")
        fst = examples.bpe_like(vocab_size=V, alphabet=ALPHABET, max_len=MAX_LEN)
        n_states = len(list(fst.states))
        target_alpha = sorted(fst.B - {EPSILON})
        # Use first few target symbols for target sequence
        target_seq = target_alpha[:min(5, len(target_alpha))]

        # Targets for one-shot: 5 target prefixes of increasing length
        oneshot_targets = [tuple(target_seq[:i+1]) for i in range(len(target_seq))]

        row = {"vocab_size": V, "fst_states": n_states}

        # One-shot
        r = bench_oneshot(fst, oneshot_targets, n_runs=N_RUNS, label=f"V={V}")
        row["oneshot_rust_ms"] = r["rust"]
        row["oneshot_openfst_ms"] = r["openfst"]

        # Dirty peekaboo
        r = bench_dirty_peekaboo(fst, target_seq, n_runs=N_RUNS, label=f"V={V}")
        row["dirty_rust_ms"] = r["rust"]
        row["dirty_openfst_ms"] = r["openfst"]

        # Lazy DFA
        r = bench_lazy_dfa(fst, target_seq, n_runs=N_RUNS, label=f"V={V}")
        row["lazy_rust_ms"] = r["rust"]
        row["lazy_openfst_ms"] = r["openfst"]

        all_results.append(row)

    return all_results


# =====================================================================
# Standard example FSTs
# =====================================================================

def bench_example_fsts():
    """Benchmark on standard example FSTs from the test suite."""
    print("\n=== Standard Example FSTs ===")

    test_cases = [
        ("small", examples.small(), ('a',)),
        ("lowercase", examples.lowercase(), ('h', 'e', 'l')),
        ("delete_b", examples.delete_b(), ('A', 'A', 'A')),
        ("samuel", examples.samuel_example(), ('c', 'x', 'y')),
        ("duplicate_K2", examples.duplicate(['a', 'b'], K=2), ('a', 'a', 'b')),
        ("togglecase", examples.togglecase(), ('A', 'b', 'C')),
        ("parity_copy", examples.parity_copy(), ('b', 'c', 'b')),
        ("backticks_to_quote", examples.backticks_to_quote(), ('b', '"', 'b')),
        ("infinite_quotient",
         examples.infinite_quotient(alphabet=('a',), separators=('#',)),
         ('1',)),
        ("bpe_like_30", examples.bpe_like(vocab_size=30, alphabet=tuple("abc"), max_len=3),
         ('a', 'b', 'c', 'a')),
    ]

    all_results = []
    N_RUNS = 5

    for name, fst, target_seq in test_cases:
        print(f"\n{name} (states={len(list(fst.states))})")

        row = {"name": name, "fst_states": len(list(fst.states))}

        # One-shot targets
        oneshot_targets = [tuple(target_seq[:i+1]) for i in range(len(target_seq))]
        r = bench_oneshot(fst, oneshot_targets, n_runs=N_RUNS, label=name)
        row["oneshot_rust_ms"] = r["rust"]
        row["oneshot_openfst_ms"] = r["openfst"]

        # Dirty peekaboo
        r = bench_dirty_peekaboo(fst, target_seq, n_runs=N_RUNS, label=name)
        row["dirty_rust_ms"] = r["rust"]
        row["dirty_openfst_ms"] = r["openfst"]

        # Lazy DFA
        r = bench_lazy_dfa(fst, target_seq, n_runs=N_RUNS, label=name)
        row["lazy_rust_ms"] = r["rust"]
        row["lazy_openfst_ms"] = r["openfst"]

        all_results.append(row)

    return all_results


# =====================================================================
# Main
# =====================================================================

def main():
    print("OpenFST vs Rust Backend Benchmark")
    print("=" * 60)
    print(f"Peak RSS at start: {peak_rss_mb():.0f} MB")

    bpe_results = bench_bpe_scaling()
    example_results = bench_example_fsts()

    print(f"\nPeak RSS at end: {peak_rss_mb():.0f} MB")

    # Save raw results
    output = {
        "bpe_scaling": bpe_results,
        "example_fsts": example_results,
    }
    with open(OUTFILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTFILE}")

    # Generate markdown report
    generate_report(bpe_results, example_results)
    print(f"Report saved to {REPORT}")


def generate_report(bpe_results, example_results):
    """Generate a markdown report from benchmark results."""
    lines = []
    lines.append("# OpenFST vs Rust Backend Benchmark")
    lines.append("")
    lines.append("Comparison of the new OpenFST C++ backend against the existing Rust backend")
    lines.append("across three decomposition modes: one-shot, incremental dirty peekaboo,")
    lines.append("and lazy DFA (classify + arcs BFS).")
    lines.append("")
    lines.append("Speedup > 1 means OpenFST is faster; < 1 means Rust is faster.")
    lines.append("")

    # BPE scaling table
    lines.append("## BPE Vocab Scaling")
    lines.append("")
    lines.append("Synthetic BPE-like FSTs with 10-symbol output alphabet, max token length 5.")
    lines.append("Each row shows median time over 3 runs for 5 target prefixes (one-shot),")
    lines.append("5-step incremental sequence (dirty peekaboo), or 5-step full BFS (lazy DFA).")
    lines.append("")
    lines.append("| Vocab | States | One-shot Rust (ms) | One-shot OpenFST (ms) | Speedup | Dirty Rust (ms) | Dirty OpenFST (ms) | Speedup | Lazy Rust (ms) | Lazy OpenFST (ms) | Speedup |")
    lines.append("|------:|-------:|-------------------:|----------------------:|--------:|----------------:|--------------------:|--------:|---------------:|-------------------:|--------:|")

    for r in bpe_results:
        os_r = r["oneshot_rust_ms"]
        os_o = r["oneshot_openfst_ms"]
        os_sp = os_r / os_o if os_o > 0 else 0
        d_r = r["dirty_rust_ms"]
        d_o = r["dirty_openfst_ms"]
        d_sp = d_r / d_o if d_o > 0 else 0
        l_r = r["lazy_rust_ms"]
        l_o = r["lazy_openfst_ms"]
        l_sp = l_r / l_o if l_o > 0 else 0
        lines.append(
            f"| {r['vocab_size']:>5} | {r['fst_states']:>6} "
            f"| {os_r:>18.1f} | {os_o:>21.1f} | {os_sp:>7.2f} "
            f"| {d_r:>15.1f} | {d_o:>19.1f} | {d_sp:>7.2f} "
            f"| {l_r:>14.1f} | {l_o:>18.1f} | {l_sp:>7.2f} |"
        )

    lines.append("")

    # Example FSTs table
    lines.append("## Standard Example FSTs")
    lines.append("")
    lines.append("Assorted FSTs from the test suite. Times are median over 5 runs.")
    lines.append("")
    lines.append("| FST | States | One-shot Rust (ms) | One-shot OpenFST (ms) | Speedup | Dirty Rust (ms) | Dirty OpenFST (ms) | Speedup | Lazy Rust (ms) | Lazy OpenFST (ms) | Speedup |")
    lines.append("|-----|-------:|-------------------:|----------------------:|--------:|----------------:|--------------------:|--------:|---------------:|-------------------:|--------:|")

    for r in example_results:
        os_r = r["oneshot_rust_ms"]
        os_o = r["oneshot_openfst_ms"]
        os_sp = os_r / os_o if os_o > 0 else 0
        d_r = r["dirty_rust_ms"]
        d_o = r["dirty_openfst_ms"]
        d_sp = d_r / d_o if d_o > 0 else 0
        l_r = r["lazy_rust_ms"]
        l_o = r["lazy_openfst_ms"]
        l_sp = l_r / l_o if l_o > 0 else 0
        lines.append(
            f"| {r['name']:<20} | {r['fst_states']:>6} "
            f"| {os_r:>18.1f} | {os_o:>21.1f} | {os_sp:>7.2f} "
            f"| {d_r:>15.1f} | {d_o:>19.1f} | {d_sp:>7.2f} "
            f"| {l_r:>14.1f} | {l_o:>18.1f} | {l_sp:>7.2f} |"
        )

    lines.append("")

    # Summary
    all_oneshot_speedups = []
    all_dirty_speedups = []
    all_lazy_speedups = []
    for r in bpe_results + example_results:
        if r["oneshot_openfst_ms"] > 0:
            all_oneshot_speedups.append(r["oneshot_rust_ms"] / r["oneshot_openfst_ms"])
        if r["dirty_openfst_ms"] > 0:
            all_dirty_speedups.append(r["dirty_rust_ms"] / r["dirty_openfst_ms"])
        if r["lazy_openfst_ms"] > 0:
            all_lazy_speedups.append(r["lazy_rust_ms"] / r["lazy_openfst_ms"])

    lines.append("## Summary")
    lines.append("")
    if all_oneshot_speedups:
        lines.append(f"- **One-shot decomposition**: median speedup = {np.median(all_oneshot_speedups):.2f}x "
                     f"(range {min(all_oneshot_speedups):.2f}x - {max(all_oneshot_speedups):.2f}x)")
    if all_dirty_speedups:
        lines.append(f"- **Dirty peekaboo**: median speedup = {np.median(all_dirty_speedups):.2f}x "
                     f"(range {min(all_dirty_speedups):.2f}x - {max(all_dirty_speedups):.2f}x)")
    if all_lazy_speedups:
        lines.append(f"- **Lazy DFA**: median speedup = {np.median(all_lazy_speedups):.2f}x "
                     f"(range {min(all_lazy_speedups):.2f}x - {max(all_lazy_speedups):.2f}x)")
    lines.append("")
    lines.append("Speedup > 1 means OpenFST is faster than Rust.")
    lines.append("")

    with open(REPORT, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
