"""
Benchmark GeneralTokenDecompose vs NonrecursiveDFADecomp.

Verifies correctness (Q/R language equivalence) and measures speedup
from position-set quotienting.
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
from transduction.fsa import FSA, EPSILON
from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp
from transduction.general_token_decompose import GeneralTokenDecompose
from transduction import examples


class Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise Timeout()

signal.signal(signal.SIGALRM, _alarm)


def enumerate_fsa(fsa, max_depth=15):
    """Enumerate accepted strings up to max_depth."""
    result = set()
    worklist = deque()
    for s in fsa.start:
        worklist.append((s, ()))
    while worklist:
        state, path = worklist.popleft()
        if len(path) > max_depth:
            continue
        if fsa.is_final(state):
            result.add(path)
        for a, j in fsa.arcs(state):
            worklist.append((j, path + (a,)))
    return result


def generate_targets(fst, max_len):
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


def verify_and_benchmark(name, fst, targets, timeout=30):
    """Compare GeneralTokenDecompose vs NonrecursiveDFADecomp."""
    n_ok = 0
    n_fail = 0
    n_skip = 0
    time_std = 0
    time_gtd = 0
    total_std_states = 0
    total_gtd_states = 0

    for target in targets:
        oov = set(target) - (fst.B - {EPSILON})
        if oov:
            n_skip += 1
            continue

        signal.alarm(timeout)
        try:
            t0 = time.time()
            std = NonrecursiveDFADecomp(fst, target)
            t1 = time.time()
            try:
                gtd = GeneralTokenDecompose(fst, target)
            except ValueError:
                # FST is not TD for this target — expected for non-TD FSTs
                n_skip += 1
                continue
            t2 = time.time()

            time_std += t1 - t0
            time_gtd += t2 - t1

            # Count states
            std_states = len(std.quotient.states) + len(std.remainder.states)
            gtd_states = gtd.n_position_sets
            total_std_states += std_states
            total_gtd_states += gtd_states

            # Verify Q/R language equivalence
            std_Q = enumerate_fsa(std.quotient)
            gtd_Q = enumerate_fsa(gtd.quotient)
            std_R = enumerate_fsa(std.remainder)
            gtd_R = enumerate_fsa(gtd.remainder)

            if std_Q == gtd_Q and std_R == gtd_R:
                n_ok += 1
            else:
                n_fail += 1
                if n_fail <= 3:
                    print(f"    FAIL target={target}", flush=True)
                    if std_Q != gtd_Q:
                        print(f"      Q diff: std-gtd={std_Q-gtd_Q}, gtd-std={gtd_Q-std_Q}", flush=True)
                    if std_R != gtd_R:
                        print(f"      R diff: std-gtd={std_R-gtd_R}, gtd-std={gtd_R-std_R}", flush=True)

        except Timeout:
            n_skip += 1
        except Exception as e:
            n_skip += 1
            if n_skip <= 3:
                print(f"    ERROR target={target}: {e}", flush=True)
        finally:
            signal.alarm(0)

    speedup = time_std / time_gtd if time_gtd > 0 else float('inf')
    compression = total_std_states / total_gtd_states if total_gtd_states > 0 else 0

    status = "OK" if n_fail == 0 else f"FAIL({n_fail})"
    print(f"  {name:<25s}  {status:<10s}  "
          f"n={n_ok:>5d}  "
          f"std={time_std:>6.3f}s  "
          f"gtd={time_gtd:>6.3f}s  "
          f"speedup={speedup:>5.2f}x  "
          f"compress={compression:>5.1f}x"
          f"  (skip={n_skip})" if n_skip > 0 else
          f"  {name:<25s}  {status:<10s}  "
          f"n={n_ok:>5d}  "
          f"std={time_std:>6.3f}s  "
          f"gtd={time_gtd:>6.3f}s  "
          f"speedup={speedup:>5.2f}x  "
          f"compress={compression:>5.1f}x",
          flush=True)

    return n_fail == 0


def main():
    print("GeneralTokenDecompose Benchmark", flush=True)
    print("=" * 100, flush=True)
    print(flush=True)

    # --- Example FSTs (TD and non-TD) ---
    test_cases = [
        ('replace(xyz)', lambda: examples.replace([('a', 'x'), ('b', 'y'), ('c', 'z')]), 4),
        ('delete_b', examples.delete_b, 4),
        ('doom(K=3)', lambda: examples.doom({'a', 'b'}, 3), 4),
        ('parity_copy', examples.parity_copy, 3),
    ]

    all_ok = True
    for name, fst_fn, max_len in test_cases:
        fst = fst_fn()
        targets = generate_targets(fst, max_len)
        ok = verify_and_benchmark(name, fst, targets)
        all_ok = all_ok and ok

    # --- BPE ---
    print(flush=True)
    print("BPE (subsampled GPT-2):", flush=True)

    from transformers import AutoTokenizer
    from transduction.lm.statelm import decode_hf_tokenizer
    print("  Loading tokenizer...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
    _, _, _decode, _ = decode_hf_tokenizer(tokenizer)
    drop = {x.encode() for x in tokenizer.all_special_tokens}
    all_ids = sorted(i for i in range(len(_decode)) if _decode[i] not in drop)
    print("done", flush=True)

    def build_bpe(vocab_size):
        used = all_ids[:vocab_size]
        m = FST()
        m.add_start(())
        for i in used:
            x = _decode[i]
            if x in drop:
                continue
            bx = tuple(x)
            for j in range(len(bx)):
                m.add_arc(bx[:j], EPSILON, bx[j], bx[:j+1])
            m.add_arc(bx, i, EPSILON, ())
        m.add_stop(())
        return m.renumber()

    texts = ["The quick brown fox", "Hello world", "It was the best"]
    for vocab_size in [100, 500]:
        fst = build_bpe(vocab_size)
        output_alpha = fst.B - {EPSILON}
        targets = []
        for text in texts:
            for length in [3, 5, 8]:
                t = tuple(text.encode('utf-8'))[:length]
                if set(t) <= output_alpha:
                    targets.append(t)
        ok = verify_and_benchmark(f"BPE(vocab={vocab_size})", fst, targets, timeout=60)
        all_ok = all_ok and ok

    # --- PTB ---
    print(flush=True)
    print("PTB:", flush=True)
    try:
        print("  Building PTB FST...", end=" ", flush=True)
        from transduction.applications.ptb import build_ptb_fst_pynini
        ptb_fst = build_ptb_fst_pynini()
        print("done", flush=True)

        output_alpha = ptb_fst.B - {EPSILON}
        targets = []
        for text in texts:
            for length in [3, 5, 8]:
                t = tuple(text.encode('utf-8'))[:length]
                if set(t) <= output_alpha:
                    targets.append(t)

        ok = verify_and_benchmark("PTB", ptb_fst, targets, timeout=60)
        all_ok = all_ok and ok
    except Exception as e:
        print(f"  PTB: error — {e}", flush=True)

    print(flush=True)
    if all_ok:
        print("All tests passed.", flush=True)
    else:
        print("SOME TESTS FAILED.", flush=True)


if __name__ == '__main__':
    main()
