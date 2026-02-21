"""
Benchmark: decomposition algorithms on BPE and PTB FSTs.

Tests on real tokenizer FSTs with byte-sequence targets at various lengths.
Compares:
  - NonrecursiveDFADecomp (fresh build per target)
  - PyniniNonrecursiveDecomp (pynini composition backend)
  - RustDecomp (Rust-accelerated decomposition)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transduction.util import set_memory_limit
set_memory_limit(8)

import signal
import time
from collections import deque
from transduction.fst import FST
from transduction.fsa import EPSILON
from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp
from transduction.pynini_ops import PyniniNonrecursiveDecomp, PyniniPrecover
from transduction.precover_nfa import PrecoverNFA

try:
    from transduction.rust_bridge import RustDecomp
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


# --- Per-call timeout ---
CALL_TIMEOUT = 30  # seconds

class CallTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise CallTimeout()

signal.signal(signal.SIGALRM, _alarm_handler)


# --- Target strings (representative English text) ---
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


def time_nonrecursive_fresh(fst, targets):
    """Time NonrecursiveDFADecomp on a list of byte-sequence targets."""
    total = 0.0
    count = 0
    timeouts = 0
    for target in targets:
        def run():
            d = NonrecursiveDFADecomp(fst, target)
            _ = d.quotient
            _ = d.remainder
        elapsed, ok = timed_call(run)
        total += elapsed
        if ok:
            count += 1
        else:
            timeouts += 1
    return total, count, timeouts


def time_pynini_fresh(fst, targets, backend=None):
    """Time PyniniNonrecursiveDecomp on a list of targets."""
    total = 0.0
    count = 0
    timeouts = 0
    for target in targets:
        def run():
            d = PyniniNonrecursiveDecomp(fst, target, backend=backend)
            _ = d.quotient
            _ = d.remainder
        elapsed, ok = timed_call(run)
        total += elapsed
        if ok:
            count += 1
        else:
            timeouts += 1
    return total, count, timeouts


def time_rust_fresh(fst, targets):
    """Time RustDecomp on a list of targets."""
    total = 0.0
    count = 0
    timeouts = 0
    for target in targets:
        def run():
            d = RustDecomp(fst, target)
            _ = d.quotient
            _ = d.remainder
        elapsed, ok = timed_call(run)
        total += elapsed
        if ok:
            count += 1
        else:
            timeouts += 1
    return total, count, timeouts


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


def analyze_spm_regime(fst, targets, name):
    """Quick SPM regime analysis on a set of targets."""
    total_states = 0
    uniform_states = 0
    for target in targets:
        try:
            def run_analysis():
                nonlocal total_states, uniform_states
                dfa = PrecoverNFA(fst, target).det()
                visited = set()
                worklist = deque()
                for i in dfa.start():
                    worklist.append(i)
                    visited.add(i)
                while worklist:
                    i = worklist.popleft()
                    positions = {len(buf) for (q, buf) in i}
                    total_states += 1
                    if len(positions) == 1:
                        uniform_states += 1
                    for a, j in dfa.arcs(i):
                        if j not in visited:
                            worklist.append(j)
                            visited.add(j)
            _, ok = timed_call(run_analysis)
            if not ok:
                print(f"    (timeout on target len={len(target)})")
        except Exception as e:
            print(f"    (error: {e})")

    pct = 100 * uniform_states / total_states if total_states > 0 else 0
    print(f"  {name:<30s}  states={total_states:>8d}  "
          f"uniform={uniform_states:>8d}  ({pct:.1f}%)")
    return total_states, uniform_states


def fmt_time(t, n, total, timeouts):
    """Format time with optional count annotation."""
    s = f"{t:>7.3f}s"
    if timeouts > 0:
        s += f" ({timeouts}T)"
    elif n < total:
        s += f" ({n}/{total})"
    return s


def benchmark_fst(name, fst, targets_by_length):
    """Run all methods on an FST with grouped targets."""
    print(f"\n{'='*100}")
    print(f"  {name}")
    print(f"{'='*100}")
    print(f"  FST: {len(fst.states)} states, |A|={len(fst.A)}, |B|={len(fst.B)}")
    print(f"  Per-call timeout: {CALL_TIMEOUT}s")

    # Pre-init pynini backend (shared across targets for this FST)
    t0 = time.perf_counter()
    pynini_backend = PyniniPrecover(fst)
    t_pynini_init = time.perf_counter() - t0
    print(f"  Pynini init: {t_pynini_init*1000:.1f}ms")
    print()

    hdr = (f"  {'Len':<5s} {'#Tgt':>5s}  "
           f"{'Standard':>14s}  {'Pynini':>14s}  ")
    if HAS_RUST:
        hdr += f"{'Rust':>14s}  "
    hdr += f"{'Pyn/Std':>8s} "
    if HAS_RUST:
        hdr += f"{'Pyn/Rust':>9s} "
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    results = []
    for length, targets in sorted(targets_by_length.items()):
        sys.stdout.write(f"  {length:<5d} {len(targets):>5d}  ")
        sys.stdout.flush()

        t_std, n_std, to_std = time_nonrecursive_fresh(fst, targets)
        sys.stdout.write(f"{fmt_time(t_std, n_std, len(targets), to_std):>14s}  ")
        sys.stdout.flush()

        t_pyn, n_pyn, to_pyn = time_pynini_fresh(fst, targets, backend=pynini_backend)
        sys.stdout.write(f"{fmt_time(t_pyn, n_pyn, len(targets), to_pyn):>14s}  ")
        sys.stdout.flush()

        if HAS_RUST:
            t_rust, n_rust, to_rust = time_rust_fresh(fst, targets)
            sys.stdout.write(f"{fmt_time(t_rust, n_rust, len(targets), to_rust):>14s}  ")
            sys.stdout.flush()

        pyn_vs_std = t_std / t_pyn if t_pyn > 0 else float('inf')

        line = f"{pyn_vs_std:>7.2f}x "
        if HAS_RUST:
            pyn_vs_rust = t_rust / t_pyn if t_pyn > 0 else float('inf')
            line += f"{pyn_vs_rust:>8.2f}x "
        print(line)

        row = {
            'length': length, 'n_targets': len(targets),
            't_std': t_std, 't_pyn': t_pyn,
            'pyn_vs_std': pyn_vs_std,
        }
        if HAS_RUST:
            row['t_rust'] = t_rust
            row['pyn_vs_rust'] = pyn_vs_rust
        results.append(row)

    return results


def main():
    print("BPE and PTB Decomposition Benchmark")
    print("=" * 100)
    print(f"Memory limit: 8 GB, per-call timeout: {CALL_TIMEOUT}s")
    print(f"Rust backend: {'available' if HAS_RUST else 'NOT available'}")

    all_results = {}

    # --- BPE benchmarks at various vocab sizes ---
    bpe_vocab_sizes = [100, 500, 1000]
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

    # --- SPM regime analysis for BPE ---
    print(f"\n{'='*80}")
    print(f"  SPM Regime Analysis")
    print(f"{'='*80}")

    for vocab_size in bpe_vocab_sizes:
        fst = build_subsampled_bpe(vocab_size)
        output_alpha = fst.B - {EPSILON}
        tgts = []
        for text in TARGET_TEXTS[:5]:
            bs = tuple(text.encode('utf-8'))[:8]
            if set(bs) <= output_alpha and len(bs) == 8:
                tgts.append(bs)
        if tgts:
            analyze_spm_regime(fst, tgts[:3], f"BPE (vocab={vocab_size})")

    # --- PTB benchmark ---
    try:
        print(f"\nBuilding PTB FST...", end=" ", flush=True)
        from transduction.applications.ptb import build_ptb_fst_pynini
        ptb_fst = build_ptb_fst_pynini()
        print(f"done ({len(ptb_fst.states)} states)")

        output_alpha = ptb_fst.B - {EPSILON}
        targets_by_length = make_byte_targets(TARGET_TEXTS, [3, 5, 8, 10, 15], output_alpha)
        name = "PTB"
        results = benchmark_fst(name, ptb_fst, targets_by_length)
        all_results[name] = results

        # SPM analysis for PTB
        tgts = []
        for text in TARGET_TEXTS[:5]:
            bs = tuple(text.encode('utf-8'))[:8]
            if set(bs) <= output_alpha and len(bs) == 8:
                tgts.append(bs)
        if tgts:
            analyze_spm_regime(ptb_fst, tgts[:3], "PTB")
    except Exception as e:
        import traceback
        print(f"\n  PTB benchmark failed: {e}")
        traceback.print_exc()

    # --- Summary ---
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY (geomean speedup vs Standard)")
    print(f"{'='*100}")
    from math import exp, log
    for name, results in all_results.items():
        if not results:
            continue
        print(f"\n  {name}:")
        for key, label in [
            ('pyn_vs_std', 'Pynini'),
            ('pyn_vs_rust', 'Pynini vs Rust'),
        ]:
            vals = [r[key] for r in results if key in r and 0 < r[key] < float('inf')]
            if vals:
                geo = exp(sum(log(s) for s in vals) / len(vals))
                print(f"    {label:<28s} geomean: {geo:.2f}x")

    print("\nDone.")


if __name__ == '__main__':
    main()
