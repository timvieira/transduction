"""Benchmark GeneralizedBeam on BPE vocab scaling and PTB.

Runs only GeneralizedBeam to collect timing data quickly.
Results saved to reports/bench_generalized_beam_results.json for
dashboard integration.

Usage:
    python reports/bench_generalized_beam.py          # full (~20 min)
    python reports/bench_generalized_beam.py --quick   # fast (~3 min)
"""

import argparse
import json
import os
import resource
import time
import gc

import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from transduction.fst import FST
from transduction.fsa import EPSILON
from transduction.lm.huggingface_lm import HfTokenizerVocab
from transduction.lm.ngram import CharNgramLM
from transduction.lm.generalized_beam import GeneralizedBeam
from transduction.util import Timeout, timelimit, set_memory_limit

set_memory_limit(12)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--quick', action='store_true', help='Fast run: fewer vocab sizes, 1 run, shorter PTB decode')
args = parser.parse_args()

OUTFILE = os.path.join(os.path.dirname(__file__), 'bench_generalized_beam_results.json')

# ---- Setup ----
os.environ['TRANSFORMERS_OFFLINE'] = '1'
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
            m.add_arc(bx[:j], EPSILON, bx[j], bx[:j + 1])
        m.add_arc(bx, i, EPSILON, ())
    m.add_stop(())
    return m


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


text_short = "The quick brown fox"
token_ids_short = tokenizer.encode(text_short)
NUM_TARGET_BYTES = 8

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: BPE Vocab Scaling
# ═══════════════════════════════════════════════════════════════════════════════

if args.quick:
    VOCAB_SIZES = [257, 1000, 5000, 15000, 50256]
    SCALE_TIMEOUT = 60
    N_RUNS = 1
else:
    VOCAB_SIZES = [257, 500, 1000, 2000, 5000, 7000, 10000, 12000, 15000, 20000, 30000, 50256]
    SCALE_TIMEOUT = 300
    N_RUNS = 1

bpe_rows = []

print(f"\n{'='*70}")
print(f"GeneralizedBeam BPE Vocab Scaling ({N_RUNS} runs, {NUM_TARGET_BYTES} decode steps)")
print(f"{'='*70}\n")

baseline_rss = peak_rss_mb()
print(f"Baseline peak RSS: {baseline_rss:.0f} MB\n")

for vs in VOCAB_SIZES:
    used = sorted(set(all_token_ids[:vs]) | set(train_used))
    fst_v = subsampled_bpe_fst(_decode, used, drop)
    source_alpha_v = fst_v.A - {EPSILON}
    inner_v = CharNgramLM.train(train_ids, n=3, alpha=0.5, alphabet=source_alpha_v)

    # Generate target bytes
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

    run_avgs = []
    init_times = []
    for run in range(N_RUNS):
        step_times = []
        try:
            with timelimit(SCALE_TIMEOUT):
                t_init_0 = time.perf_counter()
                gb = GeneralizedBeam(inner_v, fst_v, K=10, max_beam=10,
                                     max_steps=200, eos='<EOS>',
                                     helper='python')
                t_init_1 = time.perf_counter()
                init_times.append((t_init_1 - t_init_0) * 1000)

                state_v = gb.initial()
                for yb in target_bytes:
                    t0 = time.perf_counter()
                    _ = state_v.logp_next[yb]
                    state_v = state_v >> yb
                    step_times.append(time.perf_counter() - t0)
        except (Timeout, MemoryError, ValueError) as e:
            print(f"  GeneralizedBeam: {type(e).__name__} after {len(step_times)} steps (run {run + 1})")
            break

        if step_times:
            run_avgs.append(np.mean(step_times) * 1000)
        gc.collect()

    rss_after = peak_rss_mb()

    if run_avgs:
        avg = np.median(run_avgs)
        row['GeneralizedBeam_avg_ms'] = round(float(avg))
        row['GeneralizedBeam_runs_ms'] = [round(float(r)) for r in run_avgs]
        row['GeneralizedBeam_init_ms'] = round(float(np.median(init_times)))
        print(f"  GeneralizedBeam                     avg={avg:7.0f} ms/step  "
              f"(runs: {', '.join(f'{r:.0f}' for r in run_avgs)})  "
              f"init={np.median(init_times):.0f} ms")
    else:
        row['GeneralizedBeam_avg_ms'] = None
        row['GeneralizedBeam_runs_ms'] = []
        row['GeneralizedBeam_init_ms'] = None
        print(f"  GeneralizedBeam                     TIMEOUT/OOM")

    row['peak_rss_mb'] = round(rss_after)
    bpe_rows.append(row)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: PTB
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"GeneralizedBeam PTB Benchmark")
print(f"{'='*70}\n")

ptb_result = {}

try:
    from transduction.applications.ptb import build_ptb_fst_pynini

    print("Building PTB FST...", flush=True)
    t0 = time.perf_counter()
    ptb_fst = build_ptb_fst_pynini()
    ptb_build_time = time.perf_counter() - t0
    print(f"PTB FST built in {ptb_build_time:.1f}s: {len(ptb_fst.states)} states")

    ptb_text = "The quick brown fox jumps over the lazy dog."
    ptb_target_seq = list(next(ptb_fst.transduce(ptb_text.encode('utf-8'))))
    ptb_source_alpha = ptb_fst.A - {EPSILON}
    ptb_inner_lm = CharNgramLM.train(
        [list(s.encode('utf-8')) for s in train_sentences],
        n=3, alpha=0.5, alphabet=ptb_source_alpha)

    PTB_MAX_DECODE = 10 if args.quick else 45
    PTB_MAX_SEARCH = 200
    PTB_MAX_BEAM = 20
    PTB_LM_TIMEOUT = 60

    print(f"PTB target: {len(ptb_target_seq)} symbols")
    print(f"Config: K={PTB_MAX_BEAM}, max_steps={PTB_MAX_SEARCH}, timeout={PTB_LM_TIMEOUT}s\n")

    step_data = []
    try:
        with timelimit(PTB_LM_TIMEOUT):
            t_init_0 = time.perf_counter()
            gb_ptb = GeneralizedBeam(ptb_inner_lm, ptb_fst,
                                     K=PTB_MAX_BEAM, max_beam=PTB_MAX_BEAM,
                                     max_steps=PTB_MAX_SEARCH,
                                     eos='<EOS>', helper='python')
            t_init_1 = time.perf_counter()
            ptb_init_ms = (t_init_1 - t_init_0) * 1000
            print(f"  Constructor: {ptb_init_ms:.0f} ms")
            print(f"  Hubs: {len(gb_ptb._hub_tries)}, needs particles: {gb_ptb._fused_helper is not None}")

        state = gb_ptb.initial()
        for i in range(min(PTB_MAX_DECODE, len(ptb_target_seq))):
            y = ptb_target_seq[i]
            try:
                with timelimit(PTB_LM_TIMEOUT):
                    t0 = time.perf_counter()
                    lp = state.logp_next[y]
                    state = state >> y
                    t1 = time.perf_counter()
            except Timeout:
                print(f"  step {i + 1} TIMEOUT ({PTB_LM_TIMEOUT}s)")
                break
            except MemoryError:
                print(f"  step {i + 1} OOM")
                break
            except Exception as e:
                print(f"  step {i + 1} ERROR: {type(e).__name__}: {e}")
                break
            elapsed = t1 - t0
            step_data.append((i + 1, elapsed, float(lp)))
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {i + 1:3d}: {elapsed * 1000:8.1f} ms  logp={lp:.4f}")

    except (Timeout, MemoryError, ValueError) as e:
        print(f"  Constructor failed: {type(e).__name__}: {e}")

    if step_data:
        total = sum(t for _, t, _ in step_data)
        avg = total / len(step_data) * 1000
        print(f"\n  Total: {total:.2f}s, Avg: {avg:.1f} ms/step, Steps: {len(step_data)}")
        ptb_result = {
            'total_s': round(total, 2),
            'avg_ms': round(avg, 1),
            'steps': len(step_data),
            'init_ms': round(ptb_init_ms),
            'n_hubs': len(gb_ptb._hub_tries),
            'needs_particles': gb_ptb._fused_helper is not None,
            'step_data': step_data,
        }
    else:
        print("\n  No steps completed.")
        ptb_result = {'error': 'no steps completed'}

except ImportError as e:
    print(f"Skipping PTB: {e}")
    ptb_result = {'error': f'import: {e}'}

# ═══════════════════════════════════════════════════════════════════════════════
# Write results
# ═══════════════════════════════════════════════════════════════════════════════

output = {
    'description': 'GeneralizedBeam benchmark (BPE vocab scaling + PTB)',
    'config': {
        'n_runs': N_RUNS,
        'n_target_bytes': NUM_TARGET_BYTES,
        'scale_timeout_s': SCALE_TIMEOUT,
        'text': text_short,
        'memory_limit_gb': 12,
    },
    'baseline_rss_mb': round(baseline_rss),
    'bpe_rows': bpe_rows,
    'ptb': ptb_result,
}

with open(OUTFILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nWrote {OUTFILE}")

# ── Summary ──
print(f"\n{'='*70}")
print("BPE Summary Table")
print(f"{'='*70}\n")

header = f"{'|V|':>6s}  {'GeneralizedBeam (ms/step)':>25s}  {'Init (ms)':>10s}  {'Peak RSS (MB)':>14s}"
print(header)
print("-" * len(header))
for row in bpe_rows:
    avg = row.get('GeneralizedBeam_avg_ms')
    init = row.get('GeneralizedBeam_init_ms')
    rss = row.get('peak_rss_mb', '')
    line = f"{row['vocab_size']:>6d}"
    line += f"  {(f'{avg} ms' if avg is not None else 'TIMEOUT/OOM'):>25s}"
    line += f"  {(f'{init}' if init is not None else '—'):>10s}"
    line += f"  {rss:>14}"
    print(line)

print(f"\nDone.")
