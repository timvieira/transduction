"""
Benchmark: FactoredDecomp vs NonrecursiveDFADecomp on BPE and PTB FSTs.

Tests on real tokenizer FSTs with byte-sequence targets at various lengths.
Compares:
  - NonrecursiveDFADecomp (fresh build per target)
  - FactoredDecomp (fresh build per target)
  - FactoredDecomp incremental (>> operator, sequential extension)
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
from transduction.factored_decompose import FactoredDecomp
from transduction.precover_nfa import PrecoverNFA


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


def time_factored_fresh(fst, targets):
    """Time FactoredDecomp (fresh, non-incremental) on targets."""
    total = 0.0
    count = 0
    timeouts = 0
    for target in targets:
        def run():
            d = FactoredDecomp(fst, target)
            _ = d.quotient
            _ = d.remainder
        elapsed, ok = timed_call(run)
        total += elapsed
        if ok:
            count += 1
        else:
            timeouts += 1
    return total, count, timeouts


def time_factored_incremental(fst, targets):
    """Time FactoredDecomp using >> for incremental extension."""
    total = 0.0
    count = 0
    timeouts = 0
    for target in targets:
        def run():
            state = FactoredDecomp(fst, ())
            for b in target:
                state = state >> b
            _ = state.quotient
            _ = state.remainder
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
    from transduction.lm.statelm import decode_hf_tokenizer

    if '_tok' not in _cache:
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
        _, _, _decode, _ = decode_hf_tokenizer(tokenizer)
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
    """Run all three methods on an FST with grouped targets."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")
    print(f"  FST: {len(fst.states)} states, |A|={len(fst.A)}, |B|={len(fst.B)}")
    print(f"  Per-call timeout: {CALL_TIMEOUT}s")
    print()
    print(f"  {'Len':<5s} {'#Tgt':>5s}  "
          f"{'Standard':>14s}  {'Fac(fresh)':>14s}  {'Fac(>>)':>14s}  "
          f"{'Fac/Std':>8s} {'>>/Std':>8s}")
    print(f"  {'-'*80}")

    results = []
    for length, targets in sorted(targets_by_length.items()):
        sys.stdout.write(f"  {length:<5d} {len(targets):>5d}  ")
        sys.stdout.flush()

        t_std, n_std, to_std = time_nonrecursive_fresh(fst, targets)
        sys.stdout.write(f"{fmt_time(t_std, n_std, len(targets), to_std):>14s}  ")
        sys.stdout.flush()

        t_fac, n_fac, to_fac = time_factored_fresh(fst, targets)
        sys.stdout.write(f"{fmt_time(t_fac, n_fac, len(targets), to_fac):>14s}  ")
        sys.stdout.flush()

        t_inc, n_inc, to_inc = time_factored_incremental(fst, targets)

        fac_vs_std = t_std / t_fac if t_fac > 0 else float('inf')
        inc_vs_std = t_std / t_inc if t_inc > 0 else float('inf')

        print(f"{fmt_time(t_inc, n_inc, len(targets), to_inc):>14s}  "
              f"{fac_vs_std:>7.2f}x {inc_vs_std:>7.2f}x")

        results.append((length, len(targets), t_std, t_fac, t_inc, fac_vs_std, inc_vs_std))

    return results


def main():
    print("BPE and PTB Benchmark: FactoredDecomp vs NonrecursiveDFADecomp")
    print("=" * 80)
    print(f"Memory limit: 8 GB, per-call timeout: {CALL_TIMEOUT}s")

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
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    for name, results in all_results.items():
        if not results:
            continue
        print(f"\n  {name}:")
        from math import exp, log
        fac_speedups = [r[5] for r in results if 0 < r[5] < float('inf')]
        inc_speedups = [r[6] for r in results if 0 < r[6] < float('inf')]
        if fac_speedups:
            geo_fac = exp(sum(log(s) for s in fac_speedups) / len(fac_speedups))
            print(f"    Factored (fresh) geomean speedup: {geo_fac:.2f}x")
        if inc_speedups:
            geo_inc = exp(sum(log(s) for s in inc_speedups) / len(inc_speedups))
            print(f"    Factored (incremental >>) geomean speedup: {geo_inc:.2f}x")

    print("\nDone.")


if __name__ == '__main__':
    main()
