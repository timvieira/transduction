"""
Benchmark: TrieDispatchDFADecomp on BPE and PTB FSTs.

Measures decomposition time for the trie-dispatch optimization and captures
dispatch stats (trie vs fallback states/arcs). Saves results as JSON for
comparison against NonrecursiveDFADecomp / RustDecomp data.

Two benchmarks:
1. Vocab scaling: fixed target length=8, increasing BPE vocab sizes
   (comparable to bench_vectorization.py / dashboard curves).
2. Target-length sweep: fixed vocab sizes × varying target lengths
   (same format as bpe_ptb_benchmark.py).

Usage:
    python reports/bench_trie_dispatch.py          # full (~20 min)
    python reports/bench_trie_dispatch.py --quick   # fast (~3 min)
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transduction.util import set_memory_limit
set_memory_limit(8)

parser = argparse.ArgumentParser(description=__doc__.strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--quick', action='store_true', help='Fast run: fewer vocab sizes and target lengths')
args = parser.parse_args()

import json
import signal
import time
from transduction.fst import FST
from transduction.fsa import EPSILON
from transduction.trie_dispatch import TrieDispatchDFADecomp, get_stats, reset_stats
from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp

try:
    from transduction.rust_bridge import RustDecomp
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


# --- Per-call timeout ---
CALL_TIMEOUT = 60  # seconds

class CallTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise CallTimeout()

signal.signal(signal.SIGALRM, _alarm_handler)


# --- Target strings (same as bpe_ptb_benchmark.py) ---
TARGET_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "To be, or not to be, that is the question.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "Call me Ishmael.",
    "It is a truth universally acknowledged.",
    "The Morse code was developed in the 1830s.",
    "A 3.14-percent increase was noted by Dr. Smith (2024).",
    "He said, \"Hello!\" She replied: 'Hi there.'",
]


def timed_call(fn, timeout=CALL_TIMEOUT):
    """Run fn() with a SIGALRM timeout. Returns (elapsed, True) or (elapsed, False) on timeout."""
    signal.alarm(timeout)
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        signal.alarm(0)
        return elapsed, True
    except CallTimeout:
        elapsed = time.perf_counter() - t0
        return elapsed, False
    except Exception:
        elapsed = time.perf_counter() - t0
        signal.alarm(0)
        return elapsed, True  # count errors as completed (they're fast)
    finally:
        signal.alarm(0)


def build_subsampled_bpe(vocab_size, _cache={}):
    """Build a subsampled BPE FST with the given vocab size (cached)."""
    if vocab_size in _cache:
        return _cache[vocab_size]
    from transformers import AutoTokenizer
    from transduction.lm.statelm import HfTokenizerVocab

    if '_tok' not in _cache:
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
        _decode = HfTokenizerVocab(tokenizer).decode
        drop = {x.encode() for x in tokenizer.all_special_tokens}
        all_token_ids = sorted(i for i in range(len(_decode)) if _decode[i] not in drop)
        _cache['_tok'] = (_decode, drop, all_token_ids)
    _decode, drop, all_token_ids = _cache['_tok']

    used_ids = all_token_ids[:vocab_size]
    m = FST()
    m.add_start(())
    for i in used_ids:
        x = _decode[i]
        if x in drop:
            continue
        bx = tuple(x)
        for j in range(len(bx)):
            m.add_arc(bx[:j], EPSILON, bx[j], bx[:j+1])
        m.add_arc(bx, i, EPSILON, ())
    m.add_stop(())
    fst = m.renumber()
    _cache[vocab_size] = fst
    return fst


def make_byte_targets(texts, lengths, output_alpha):
    """Generate byte-sequence targets from text, grouped by length."""
    targets_by_length = {}
    for length in lengths:
        tgts = []
        for text in texts:
            bs = tuple(text.encode('utf-8'))[:length]
            if len(bs) == length and set(bs) <= output_alpha:
                tgts.append(bs)
        if tgts:
            targets_by_length[length] = tgts
    return targets_by_length


def time_method(decomp_cls, fst, targets):
    """Time a decomposition class on a list of targets."""
    total = 0.0
    count = 0
    timeouts = 0
    for target in targets:
        def run():
            d = decomp_cls(fst, target)
            _ = d.quotient
            _ = d.remainder
        elapsed, ok = timed_call(run)
        total += elapsed
        if ok:
            count += 1
        else:
            timeouts += 1
    return total, count, timeouts


def fmt_time(t, n, total, timeouts):
    """Format time with optional count annotation."""
    s = f"{t:>7.3f}s"
    if timeouts > 0:
        s += f" ({timeouts}T)"
    elif n < total:
        s += f" ({n}/{total})"
    return s


def benchmark_fst(name, fst, targets_by_length):
    """Run TrieDispatch (+ baselines) on an FST with grouped targets."""
    print(f"\n{'='*110}")
    print(f"  {name}")
    print(f"{'='*110}")
    print(f"  FST: {len(fst.states)} states, |A|={len(fst.A)}, |B|={len(fst.B)}")
    print(f"  Per-call timeout: {CALL_TIMEOUT}s")
    print()

    hdr = (f"  {'Len':<5s} {'#Tgt':>5s}  "
           f"{'Standard':>14s}  {'TrieDispatch':>14s}  ")
    if HAS_RUST:
        hdr += f"{'Rust':>14s}  "
    hdr += (f"{'TD/Std':>8s} ")
    if HAS_RUST:
        hdr += f"{'TD/Rust':>9s} "
    hdr += f"{'TrieStates':>12s}  {'TrieArcs':>12s}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    results = []
    for length, targets in sorted(targets_by_length.items()):
        sys.stdout.write(f"  {length:<5d} {len(targets):>5d}  ")
        sys.stdout.flush()

        # Baseline: NonrecursiveDFADecomp
        t_std, n_std, to_std = time_method(NonrecursiveDFADecomp, fst, targets)
        sys.stdout.write(f"{fmt_time(t_std, n_std, len(targets), to_std):>14s}  ")
        sys.stdout.flush()

        # TrieDispatchDFADecomp with stats
        reset_stats()
        t_td, n_td, to_td = time_method(TrieDispatchDFADecomp, fst, targets)
        stats = get_stats()
        sys.stdout.write(f"{fmt_time(t_td, n_td, len(targets), to_td):>14s}  ")
        sys.stdout.flush()

        # Rust baseline
        if HAS_RUST:
            t_rust, n_rust, to_rust = time_method(RustDecomp, fst, targets)
            sys.stdout.write(f"{fmt_time(t_rust, n_rust, len(targets), to_rust):>14s}  ")
            sys.stdout.flush()

        # Speedup ratios (>1 means TrieDispatch is faster)
        td_vs_std = t_std / t_td if t_td > 0 else float('inf')
        line = f"{td_vs_std:>7.2f}x "
        if HAS_RUST:
            td_vs_rust = t_rust / t_td if t_td > 0 else float('inf')
            line += f"{td_vs_rust:>8.2f}x "

        # Stats
        total_states = stats['trie_states'] + stats['fallback_states']
        total_arcs = stats['trie_arcs'] + stats['fallback_arcs']
        trie_state_pct = 100 * stats['trie_states'] / total_states if total_states > 0 else 0
        trie_arc_pct = 100 * stats['trie_arcs'] / total_arcs if total_arcs > 0 else 0
        line += f"{trie_state_pct:>5.1f}%/{total_states:<5d}  {trie_arc_pct:>5.1f}%/{total_arcs:<5d}"
        print(line)

        row = {
            'length': length, 'n_targets': len(targets),
            't_std': t_std, 't_trie_dispatch': t_td,
            'n_std': n_std, 'n_td': n_td,
            'to_std': to_std, 'to_td': to_td,
            'td_vs_std': td_vs_std,
            'stats': stats,
        }
        if HAS_RUST:
            row['t_rust'] = t_rust
            row['td_vs_rust'] = td_vs_rust
        results.append(row)

    return results


def vocab_scaling_benchmark():
    """BPE vocab scaling: fixed target_length=8, increasing vocab sizes."""
    print(f"\n{'='*110}")
    print(f"  BPE VOCAB SCALING (target_length=8)")
    print(f"{'='*110}")

    vocab_sizes = [257, 1000, 5000] if args.quick else [100, 257, 500, 1000, 2000, 3000, 5000]
    target_length = 8

    hdr = (f"  {'Vocab':>7s} {'States':>7s} {'#Tgt':>5s}  "
           f"{'Standard':>14s}  {'TrieDispatch':>14s}  ")
    if HAS_RUST:
        hdr += f"{'Rust':>14s}  "
    hdr += f"{'TD/Std':>8s} "
    if HAS_RUST:
        hdr += f"{'TD/Rust':>9s} "
    hdr += f"{'Trie%':>6s}"
    print(f"\n{hdr}")
    print(f"  {'-'*len(hdr)}")

    scaling_rows = []
    for vocab_size in vocab_sizes:
        fst = build_subsampled_bpe(vocab_size)
        output_alpha = fst.B - {EPSILON}
        targets_by_length = make_byte_targets(TARGET_TEXTS, [target_length], output_alpha)
        targets = targets_by_length.get(target_length, [])
        if not targets:
            print(f"  {vocab_size:>7d} {len(fst.states):>7d}   (no valid targets)")
            continue

        sys.stdout.write(f"  {vocab_size:>7d} {len(fst.states):>7d} {len(targets):>5d}  ")
        sys.stdout.flush()

        # Standard
        t_std, n_std, to_std = time_method(NonrecursiveDFADecomp, fst, targets)
        sys.stdout.write(f"{fmt_time(t_std, n_std, len(targets), to_std):>14s}  ")
        sys.stdout.flush()

        # TrieDispatch
        reset_stats()
        t_td, n_td, to_td = time_method(TrieDispatchDFADecomp, fst, targets)
        stats = get_stats()
        sys.stdout.write(f"{fmt_time(t_td, n_td, len(targets), to_td):>14s}  ")
        sys.stdout.flush()

        # Rust
        t_rust = None
        if HAS_RUST:
            t_rust_v, n_rust, to_rust = time_method(RustDecomp, fst, targets)
            t_rust = t_rust_v
            sys.stdout.write(f"{fmt_time(t_rust, n_rust, len(targets), to_rust):>14s}  ")
            sys.stdout.flush()

        td_vs_std = t_std / t_td if t_td > 0 else float('inf')
        line = f"{td_vs_std:>7.2f}x "
        if HAS_RUST:
            td_vs_rust = t_rust / t_td if t_td > 0 else float('inf')
            line += f"{td_vs_rust:>8.2f}x "

        total_states = stats['trie_states'] + stats['fallback_states']
        trie_pct = 100 * stats['trie_states'] / total_states if total_states > 0 else 0
        line += f"{trie_pct:>5.1f}%"
        print(line)

        # Compute avg ms per target
        avg_std_ms = (t_std / len(targets) * 1000) if n_std > 0 else None
        avg_td_ms = (t_td / len(targets) * 1000) if n_td > 0 else None
        avg_rust_ms = (t_rust / len(targets) * 1000) if t_rust is not None else None

        scaling_rows.append({
            'vocab_size': vocab_size,
            'fst_states': len(fst.states),
            'n_targets': len(targets),
            't_std': t_std, 't_trie_dispatch': t_td, 't_rust': t_rust,
            'avg_std_ms': avg_std_ms, 'avg_td_ms': avg_td_ms, 'avg_rust_ms': avg_rust_ms,
            'td_vs_std': td_vs_std,
            'td_vs_rust': td_vs_rust if HAS_RUST else None,
            'to_std': to_std, 'to_td': to_td,
            'stats': stats,
        })

    return scaling_rows


def main():
    from math import exp, log

    print("TrieDispatchDFADecomp Benchmark")
    print("=" * 110)
    print(f"Memory limit: 8 GB, per-call timeout: {CALL_TIMEOUT}s")
    print(f"Rust backend: {'available' if HAS_RUST else 'NOT available'}")

    all_results = {}

    # === Part 1: Vocab scaling (for dashboard) ===
    scaling_rows = vocab_scaling_benchmark()
    all_results['vocab_scaling'] = scaling_rows

    # === Part 2: Target-length sweep at select vocab sizes ===
    if args.quick:
        bpe_vocab_sizes = [500, 2000]
        target_lengths = [3, 8]
    else:
        bpe_vocab_sizes = [500, 1000, 2000]
        target_lengths = [3, 5, 8, 10, 15]

    for vocab_size in bpe_vocab_sizes:
        print(f"\nBuilding BPE FST (vocab={vocab_size})...", end=" ", flush=True)
        fst = build_subsampled_bpe(vocab_size)
        print(f"done ({len(fst.states)} states)")
        output_alpha = fst.B - {EPSILON}
        targets_by_length = make_byte_targets(TARGET_TEXTS, target_lengths, output_alpha)
        name = f"BPE (vocab={vocab_size})"
        results = benchmark_fst(name, fst, targets_by_length)
        all_results[name] = results

    # --- PTB benchmark ---
    try:
        print(f"\nBuilding PTB FST...", end=" ", flush=True)
        from transduction.applications.ptb import build_ptb_fst_pynini
        ptb_fst = build_ptb_fst_pynini()
        print(f"done ({len(ptb_fst.states)} states)")

        output_alpha = ptb_fst.B - {EPSILON}
        ptb_lengths = [3, 8] if args.quick else [3, 5, 8, 10, 15]
        targets_by_length = make_byte_targets(TARGET_TEXTS, ptb_lengths, output_alpha)
        name = "PTB"
        results = benchmark_fst(name, ptb_fst, targets_by_length)
        all_results[name] = results
    except Exception as e:
        import traceback
        print(f"\n  PTB benchmark failed: {e}")
        traceback.print_exc()

    # === Summary ===
    print(f"\n\n{'='*110}")
    print(f"  SUMMARY")
    print(f"{'='*110}")

    # Vocab scaling summary
    print(f"\n  Vocab Scaling (target_length=8, avg ms per decomposition):")
    print(f"  {'Vocab':>7s}  {'Standard':>10s}  {'TrieDisp':>10s}  {'Rust':>10s}  {'TD/Std':>8s}")
    print(f"  {'-'*52}")
    for r in scaling_rows:
        std_s = f"{r['avg_std_ms']:.0f} ms" if r['avg_std_ms'] is not None else "TIMEOUT"
        td_s = f"{r['avg_td_ms']:.0f} ms" if r['avg_td_ms'] is not None else "TIMEOUT"
        rust_s = f"{r['avg_rust_ms']:.0f} ms" if r['avg_rust_ms'] is not None else "N/A"
        ratio_s = f"{r['td_vs_std']:.2f}x" if r['td_vs_std'] < float('inf') else "N/A"
        print(f"  {r['vocab_size']:>7d}  {std_s:>10s}  {td_s:>10s}  {rust_s:>10s}  {ratio_s:>8s}")

    # Per-FST geomean summary
    for name, results in all_results.items():
        if name == 'vocab_scaling' or not results:
            continue
        print(f"\n  {name}:")
        for key, label in [
            ('td_vs_std', 'TrieDispatch vs Standard'),
            ('td_vs_rust', 'TrieDispatch vs Rust'),
        ]:
            vals = [r[key] for r in results if key in r and 0 < r[key] < float('inf')]
            if vals:
                geo = exp(sum(log(s) for s in vals) / len(vals))
                print(f"    {label:<32s} geomean: {geo:.2f}x")

    # --- Save JSON ---
    out_path = os.path.join(os.path.dirname(__file__), 'bench_trie_dispatch_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
