"""Benchmark BPE vocab scaling for FusedTransducedLM.

Measures avg ms/step and peak memory at each vocabulary size.
Writes results to reports/bench_vectorization_results.json for
consumption by dashboard_plots.py.

Usage:
    python reports/bench_vectorization.py          # full (~30 min)
    python reports/bench_vectorization.py --quick   # fast (~3 min)
"""

import argparse
import json
import os
import resource
import sys
import time
import gc

import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from transduction.fst import FST
from transduction.fsa import EPSILON
from transduction.lm.huggingface_lm import HfTokenizerVocab
from transduction.lm.ngram import CharNgramLM
from transduction.lm.fused_transduced import FusedTransducedLM
from transduction.lm.character_beam import CharacterBeam
from transduction.util import Timeout, timelimit, set_memory_limit

set_memory_limit(16)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--quick', action='store_true', help='Fast run: fewer vocab sizes, 1 run, skip quality check')
args = parser.parse_args()

OUTFILE = os.path.join(os.path.dirname(__file__), 'bench_vectorization_results.json')

# ---- Setup ----
tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False, local_files_only=True)
hf_vocab = HfTokenizerVocab(tokenizer)
_decode = hf_vocab.decode
drop = {x.encode() for x in tokenizer.all_special_tokens}
all_token_ids = sorted(i for i in range(len(_decode)) if _decode[i] not in drop)

eos_bytes = tokenizer.eos_token.encode()
eos_id = hf_vocab.encode[eos_bytes]

train_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A stitch in time saves nine.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "Actions speak louder than words.",
    "Practice makes perfect.",
    "Where there is a will, there is a way.",
] * 3
train_ids = [tokenizer.encode(s) for s in train_sentences]
train_used = sorted(set(tid for seq in train_ids for tid in seq))


def subsampled_bpe_fst(decode, token_ids, drop=frozenset()):
    m = FST()
    m.add_start(())
    for i in token_ids:
        x = decode[i]
        if x in drop:
            continue
        bx = tuple(x)
        for j in range(len(bx)):
            m.add_arc(bx[:j], EPSILON, bx[j], bx[:j+1])
        m.add_arc(bx, i, EPSILON, ())
    m.add_stop(())
    return m


def peak_rss_mb():
    """Current peak RSS in MB (high-water mark for the process)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def current_rss_mb():
    """Current RSS in MB (not high-water mark) via /proc/self/statm."""
    with open('/proc/self/statm') as f:
        pages = int(f.read().split()[1])  # second field = resident pages
    return pages * resource.getpagesize() / (1024 * 1024)


# Target byte sequence: generated per vocab size (must be transducible)
text_short = "The quick brown fox"
token_ids_short = tokenizer.encode(text_short)
NUM_TARGET_BYTES = 8
print(f"Text: {text_short!r}, Token IDs: {token_ids_short}")

# ---- Vocab sizes to benchmark ----
# --quick keeps the full range of sizes (fast methods like CharacterBeam still
# reach large vocabs) but reduces runs to 1 and shortens per-method timeout.
if args.quick:
    VOCAB_SIZES = [257, 1000, 5000, 10000, 15000]
    SCALE_TIMEOUT = 60
    N_RUNS = 1
else:
    VOCAB_SIZES = [257, 500, 1000, 2000, 5000, 7000, 10000, 12000, 15000, 20000, 30000, 50256]
    SCALE_TIMEOUT = 300
    N_RUNS = 1

def make_character_beam(lm, used):
    """Build a CharacterBeam from an LM and a set of token IDs."""
    vocab = {tid: _decode[tid] for tid in used
             if _decode[tid] is not None and _decode[tid] not in drop}
    if eos_id not in vocab:
        vocab[eos_id] = eos_bytes
    return CharacterBeam(lm, vocab, K=10, eos_token=eos_id)

methods = [
    ('CharacterBeam', lambda lm, fst, used: make_character_beam(lm, used)),
    ('FusedLM_rust', lambda lm, fst, used: FusedTransducedLM(lm, fst, max_steps=200, max_beam=10, helper='rust')),
]

rows = []  # collected for JSON output

print(f"\n{'='*70}")
print(f"BPE Vocab Scaling Benchmark ({N_RUNS} runs, {NUM_TARGET_BYTES} decode steps)")
print(f"{'='*70}\n")

baseline_rss = peak_rss_mb()
print(f"Baseline peak RSS: {baseline_rss:.0f} MB\n")

for vs in VOCAB_SIZES:
    used = sorted(set(all_token_ids[:vs]) | set(train_used))
    fst_v = subsampled_bpe_fst(_decode, used, drop)
    source_alpha_v = fst_v.A - {EPSILON}
    inner_v = CharNgramLM.train(train_ids, n=3, alpha=0.5, alphabet=source_alpha_v)

    # Generate target bytes using tokens that are in this FST's vocab
    used_set = set(used)
    target_bytes = None
    for seq in train_ids:
        if all(tid in used_set for tid in seq):
            try:
                target_bytes = list(next(fst_v.transduce(seq)))[:NUM_TARGET_BYTES]
                break
            except ValueError:
                continue
    if target_bytes is None:
        target_bytes = []
        for tid in train_used:
            if tid in used_set:
                try:
                    target_bytes.extend(next(fst_v.transduce([tid])))
                except ValueError:
                    continue
                if len(target_bytes) >= NUM_TARGET_BYTES:
                    break
        target_bytes = target_bytes[:NUM_TARGET_BYTES]

    row = {
        'vocab_size': len(used),
        'fst_states': len(fst_v.states),
        'n_target_bytes': len(target_bytes),
    }

    print(f"V={len(used):>5d}  ({len(fst_v.states)} FST states, {len(target_bytes)} target bytes)")

    for method_name, make_tlm in methods:
        gc.collect()
        rss_before = current_rss_mb()
        run_avgs = []
        tlm_v = None
        state_v = None
        for run in range(N_RUNS):
            step_times = []
            try:
                with timelimit(SCALE_TIMEOUT):
                    tlm_v = make_tlm(inner_v, fst_v, used)
                    state_v = tlm_v.initial()
                    for yb in target_bytes:
                        t0 = time.perf_counter()
                        _ = state_v.logp_next[yb]
                        state_v = state_v >> yb
                        step_times.append(time.perf_counter() - t0)
            except (Timeout, MemoryError, ValueError) as e:
                print(f"  {method_name}: {type(e).__name__} after {len(step_times)} steps (run {run+1})")
                break

            if step_times:
                run_avgs.append(np.mean(step_times) * 1000)

        rss_after = current_rss_mb()

        key = method_name
        if run_avgs:
            avg = np.median(run_avgs)
            row[f'{key}_avg_ms'] = round(float(avg))
            row[f'{key}_runs_ms'] = [round(float(r)) for r in run_avgs]
            row[f'{key}_rss_mb'] = round(rss_after)
            print(f"  {method_name:<35s}  avg={avg:7.0f} ms/step  RSS={rss_after:.0f} MB (delta={rss_after - rss_before:.0f} MB)")
        else:
            row[f'{key}_avg_ms'] = None
            row[f'{key}_runs_ms'] = []
            row[f'{key}_rss_mb'] = round(rss_after) if rss_after > rss_before else None
            print(f"  {method_name:<35s}  TIMEOUT/OOM  RSS={rss_after:.0f} MB")

        # Clean up method objects before next method
        del tlm_v, state_v
        gc.collect()

        # Backward compat: still record process peak
        row['peak_rss_mb'] = round(peak_rss_mb())

    rows.append(row)
    print()

# ---- Write JSON ----
output = {
    'description': 'BPE vocab scaling benchmark for FusedTransducedLM',
    'config': {
        'n_runs': N_RUNS,
        'n_target_bytes': NUM_TARGET_BYTES,
        'scale_timeout_s': SCALE_TIMEOUT,
        'text': text_short,
        'memory_limit_gb': 16,
    },
    'baseline_rss_mb': round(baseline_rss),
    'rows': rows,
}

with open(OUTFILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Wrote {OUTFILE}")

# ---- Summary ----
print(f"\n{'='*70}")
print("Summary Table")
print(f"{'='*70}\n")

method_keys = [name for name, _ in methods]
header = f"{'|V|':>6s}"
for key in method_keys:
    header += f"  {key + ' (ms)':>20s}  {key + ' RSS':>10s}"
print(header)
print("-" * len(header))
for row in rows:
    line = f"{row['vocab_size']:>6d}"
    for key in method_keys:
        avg = row.get(f'{key}_avg_ms')
        rss = row.get(f'{key}_rss_mb')
        line += f"  {(f'{avg}' if avg is not None else 'TIMEOUT'):>20s}"
        line += f"  {(f'{rss} MB' if rss is not None else ''):>10s}"
    print(line)

# ---- Quality check: logp agreement at V=5000 (skip in --quick mode) ----
if args.quick:
    print(f"\n(Skipping quality check in --quick mode)")
    print(f"\nDone.")
    sys.exit(0)

print(f"\n{'='*70}")
print("Quality Check: max |logp_full - logp_topk| at V=5000")
print(f"{'='*70}\n")

quality_vs = 5000
used_q = sorted(set(all_token_ids[:quality_vs]) | set(train_used))
fst_q = subsampled_bpe_fst(_decode, used_q, drop)
source_alpha_q = fst_q.A - {EPSILON}
inner_q = CharNgramLM.train(train_ids, n=3, alpha=0.5, alphabet=source_alpha_q)

used_set_q = set(used_q)
target_bytes_q = None
for seq in train_ids:
    if all(tid in used_set_q for tid in seq):
        try:
            target_bytes_q = list(next(fst_q.transduce(seq)))[:NUM_TARGET_BYTES]
            break
        except ValueError:
            continue

if target_bytes_q:
    tlm_full = FusedTransducedLM(inner_q, fst_q, max_steps=200, max_beam=10, helper='rust')
    topk_vals = [50, 100, 500]
    tlm_topks = {k: FusedTransducedLM(inner_q, fst_q, max_steps=200, max_beam=10, helper='rust', top_k=k)
                 for k in topk_vals}

    state_full = tlm_full.initial()
    states_topk = {k: tlm.initial() for k, tlm in tlm_topks.items()}

    for step, yb in enumerate(target_bytes_q):
        logp_full = state_full.logp_next
        for k in topk_vals:
            logp_tk = states_topk[k].logp_next
            # Compare on keys present in the top_k distribution
            common_keys = set(logp_full.keys()) & set(logp_tk.keys())
            if common_keys:
                diffs = [abs(float(logp_full[key]) - float(logp_tk[key])) for key in common_keys]
                max_diff = max(diffs)
            else:
                max_diff = float('inf')
            print(f"  step {step}, top_k={k:>3d}: max|diff|={max_diff:.4f}  ({len(common_keys)}/{len(logp_full)} keys in common)")

        state_full = state_full >> yb
        states_topk = {k: s >> yb for k, s in states_topk.items()}
else:
    print("  (skipped: could not generate target bytes for quality check)")

print(f"\nDone.")
