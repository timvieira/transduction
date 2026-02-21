"""Run all dashboard benchmarks: BPE TransducedLM, PTB backend comparison,
logp agreement, and PyniniTransducedLM debugging.

Usage:
    python reports/run_benchmarks.py
"""

import time, gc, json, os, sys
from math import exp, log
from collections import defaultdict

from transduction.util import set_memory_limit, Timeout, timelimit
set_memory_limit(8)

import numpy as np

# ── Shared setup ──────────────────────────────────────────────────────────────

print('='*70, flush=True)
print('BENCHMARK RUNNER', flush=True)
print('='*70, flush=True)

# ── BPE FST setup ─────────────────────────────────────────────────────────────

print('Loading GPT-2 tokenizer...', flush=True)
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoTokenizer
from transduction.fst import FST
from transduction.fsa import EPSILON
from transduction.lm.statelm import HfTokenizerVocab
from transduction.lm.ngram import CharNgramLM

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False, local_files_only=True)
_decode = HfTokenizerVocab(tokenizer).decode
drop = {x.encode() for x in tokenizer.all_special_tokens}

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

bpe_fst = subsampled_bpe_fst(_decode, train_used, drop)
bpe_source_alpha = bpe_fst.A - {EPSILON}
bpe_inner_lm = CharNgramLM.train(train_ids, n=3, alpha=0.5, alphabet=bpe_source_alpha)

text = "The quick brown fox jumps over the lazy dog."
bpe_token_ids = tokenizer.encode(text)
bpe_target_seq = list(bpe_fst.transduce(bpe_token_ids))

print(f'\nBPE FST: {len(bpe_fst.states)} states, |A|={len(bpe_fst.A)}, |B|={len(bpe_fst.B)}')
print(f'BPE target: {len(bpe_target_seq)} symbols')

# ── PTB FST setup ─────────────────────────────────────────────────────────────

print('\nBuilding PTB FST...', flush=True)
from transduction.applications.ptb import build_ptb_fst_pynini
t0 = time.perf_counter()
ptb_fst = build_ptb_fst_pynini()
ptb_build_time = time.perf_counter() - t0
print(f'PTB FST built in {ptb_build_time:.1f}s: {len(ptb_fst.states)} states')

ptb_text = "The quick brown fox jumps over the lazy dog."
ptb_target_seq = list(ptb_fst.transduce(ptb_text.encode('utf-8')))
ptb_source_alpha = ptb_fst.A - {EPSILON}
ptb_inner_lm = CharNgramLM.train(
    [list(s.encode('utf-8')) for s in train_sentences],
    n=3, alpha=0.5, alphabet=ptb_source_alpha)

print(f'PTB target: {len(ptb_target_seq)} symbols')

# ── Collect results ───────────────────────────────────────────────────────────

results = {}

# ══════════════════════════════════════════════════════════════════════════════
# Section 1: PTB Decomposition Backend Comparison (Issue #2)
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PTB DECOMPOSITION BACKEND COMPARISON')
print('='*70)

from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp
from transduction.rust_bridge import RustDecomp
from transduction.general_token_decompose import GeneralTokenDecompose

ptb_target = tuple(ptb_target_seq[:10])

ptb_decomp_methods = [
    ('Standard', lambda t: NonrecursiveDFADecomp(ptb_fst, t)),
    ('Rust',     lambda t: RustDecomp(ptb_fst, t)),
    ('PositionSet', lambda t: GeneralTokenDecompose(ptb_fst, t)),
]

ptb_prefix_lengths = [3, 5, 8, 10]
ptb_decomp_results = {name: [] for name, _ in ptb_decomp_methods}

print(f'\nPTB FST: {len(ptb_fst.states)} states')
print(f'{"len":>5s}', end='')
for name, _ in ptb_decomp_methods:
    print(f'  {name:>12s}', end='')
print()
print('-' * (5 + 14 * len(ptb_decomp_methods)))

for length in ptb_prefix_lengths:
    target = tuple(ptb_target_seq[:length])
    print(f'{length:5d}', end='', flush=True)
    for name, fn in ptb_decomp_methods:
        try:
            with timelimit(30):
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    d = fn(target)
                    _ = d.quotient
                    _ = d.remainder
                    times.append(time.perf_counter() - t0)
                best = min(times)
                ptb_decomp_results[name].append(best)
                print(f'  {best*1000:10.1f}ms', end='')
        except (Timeout, Exception) as e:
            ptb_decomp_results[name].append(None)
            print(f'  {"FAIL":>10s}  ', end='')
    print(flush=True)

std_times = ptb_decomp_results.get('Standard', [])
print(f'\nSpeedup vs Standard (geomean):')
for name in ptb_decomp_results:
    if name == 'Standard':
        continue
    ratios = []
    for s, o in zip(std_times, ptb_decomp_results[name]):
        if s is not None and o is not None and o > 0:
            ratios.append(s / o)
    if ratios:
        geo = exp(sum(log(r) for r in ratios) / len(ratios))
        print(f'  {name}: {geo:.1f}x')

results['ptb_decomp'] = ptb_decomp_results

# ══════════════════════════════════════════════════════════════════════════════
# Section 2: BPE TransducedLM Benchmark (Issue #1)
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('BPE TRANSDUCED LM BENCHMARK')
print('='*70)

from transduction.lm.transduced import TransducedLM
from transduction.lm.fused_transduced import FusedTransducedLM
from transduction.position_set_peekaboo import PositionSetPeekabooState, _PositionSetPeekabooUniv
from transduction.rust_bridge import RustPositionSetPeekabooState, _RustPositionSetPeekabooUniv

MAX_DECODE = 44    # length of target
MAX_SEARCH = 200
MAX_BEAM = 10
LM_TIMEOUT = 30

bpe_lm_results = defaultdict(list)

bpe_configs = [
    ('TransducedLM', lambda: TransducedLM(
        bpe_inner_lm, bpe_fst, K=MAX_BEAM, max_expansions=MAX_SEARCH)),
    ('TransducedLM+PosSet', lambda: TransducedLM(
        bpe_inner_lm, bpe_fst, K=MAX_BEAM, max_expansions=MAX_SEARCH,
        decomp_state_cls=PositionSetPeekabooState,
        univ_cls=_PositionSetPeekabooUniv)),
    # RustPosSet excluded from BPE: BPE FSTs are not token-decomposable
    ('FusedTransducedLM', lambda: FusedTransducedLM(
        bpe_inner_lm, bpe_fst, max_steps=MAX_SEARCH, max_beam=MAX_BEAM)),
]

for name, make_tlm in bpe_configs:
    print(f'\n{name} (K={MAX_BEAM}, max_expansions={MAX_SEARCH}):')
    try:
        with timelimit(LM_TIMEOUT):
            tlm = make_tlm()
            state = tlm.initial()
    except (Timeout, MemoryError, ValueError) as e:
        print(f'  initial() failed: {type(e).__name__}: {e}')
        continue
    except Exception as e:
        print(f'  initial() failed: {type(e).__name__}: {e}')
        continue
    for i in range(min(MAX_DECODE, len(bpe_target_seq))):
        y = bpe_target_seq[i]
        try:
            with timelimit(LM_TIMEOUT):
                t0 = time.perf_counter()
                lp = state.logp_next[y]
                state = state >> y
                t1 = time.perf_counter()
        except Timeout:
            print(f'  step {i+1} TIMEOUT ({LM_TIMEOUT}s)')
            break
        except MemoryError:
            print(f'  step {i+1} OOM')
            break
        except Exception as e:
            print(f'  step {i+1} ERROR: {type(e).__name__}: {e}')
            break
        elapsed = t1 - t0
        bpe_lm_results[name].append((i + 1, elapsed, lp))
        if (i+1) % 10 == 0 or i == 0:
            print(f'  {i+1:3d}: {elapsed*1000:8.1f} ms  logp={lp:.4f}')
    gc.collect()

# BPE Summary
print(f'\n{"Algorithm":<25s} {"Total (s)":>10s} {"Avg/step (ms)":>14s} {"Steps":>6s}')
print('-' * 57)
for name, data in sorted(bpe_lm_results.items()):
    total = sum(t for _, t, _ in data)
    avg = total / len(data) * 1000
    print(f'{name:<25s} {total:10.2f} {avg:14.1f} {len(data):6d}')

# BPE logp agreement
names = sorted(bpe_lm_results.keys())
if len(names) >= 2:
    ref_name = names[0]
    ref_data = bpe_lm_results[ref_name]
    print(f'\nMax |logp| diff vs {ref_name}:')
    for name in names[1:]:
        data = bpe_lm_results[name]
        n = min(len(ref_data), len(data))
        if n > 0:
            diffs = [abs(ref_data[i][2] - data[i][2]) for i in range(n)]
            max_diff = max(diffs)
            print(f'  {name}: {max_diff:.6f}')

results['bpe_lm'] = {name: [(s, t, lp) for s, t, lp in data]
                      for name, data in bpe_lm_results.items()}

# ══════════════════════════════════════════════════════════════════════════════
# Section 3: PTB TransducedLM (with fix, re-check logp agreement)
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PTB TRANSDUCED LM BENCHMARK (with FusedTransducedLM drain fix)')
print('='*70)

PTB_MAX_DECODE = 45
PTB_MAX_SEARCH = 200
PTB_MAX_BEAM = 20
PTB_LM_TIMEOUT = 60

ptb_lm_results = defaultdict(list)

ptb_configs = [
    ('TransducedLM', lambda: TransducedLM(
        ptb_inner_lm, ptb_fst, K=PTB_MAX_BEAM, max_expansions=PTB_MAX_SEARCH)),
    ('TransducedLM+PosSet', lambda: TransducedLM(
        ptb_inner_lm, ptb_fst, K=PTB_MAX_BEAM, max_expansions=PTB_MAX_SEARCH,
        decomp_state_cls=PositionSetPeekabooState,
        univ_cls=_PositionSetPeekabooUniv)),
    ('TransducedLM+RustPosSet', lambda: TransducedLM(
        ptb_inner_lm, ptb_fst, K=PTB_MAX_BEAM, max_expansions=PTB_MAX_SEARCH,
        decomp_state_cls=RustPositionSetPeekabooState,
        univ_cls=_RustPositionSetPeekabooUniv)),
    ('FusedTransducedLM', lambda: FusedTransducedLM(
        ptb_inner_lm, ptb_fst, max_steps=PTB_MAX_SEARCH, max_beam=PTB_MAX_BEAM)),
]

for name, make_tlm in ptb_configs:
    print(f'\n{name} (K={PTB_MAX_BEAM}, max_expansions={PTB_MAX_SEARCH}):')
    try:
        with timelimit(PTB_LM_TIMEOUT):
            tlm = make_tlm()
            state = tlm.initial()
    except (Timeout, MemoryError, ValueError) as e:
        print(f'  initial() failed: {type(e).__name__}: {e}')
        continue
    except Exception as e:
        print(f'  initial() failed: {type(e).__name__}: {e}')
        continue
    for i in range(min(PTB_MAX_DECODE, len(ptb_target_seq))):
        y = ptb_target_seq[i]
        try:
            with timelimit(PTB_LM_TIMEOUT):
                t0 = time.perf_counter()
                lp = state.logp_next[y]
                state = state >> y
                t1 = time.perf_counter()
        except Timeout:
            print(f'  step {i+1} TIMEOUT ({PTB_LM_TIMEOUT}s)')
            break
        except MemoryError:
            print(f'  step {i+1} OOM')
            break
        except Exception as e:
            print(f'  step {i+1} ERROR: {type(e).__name__}: {e}')
            break
        elapsed = t1 - t0
        ptb_lm_results[name].append((i + 1, elapsed, lp))
        if (i+1) % 10 == 0 or i == 0:
            print(f'  {i+1:3d}: {elapsed*1000:8.1f} ms  logp={lp:.4f}')
    gc.collect()

# PTB Summary
print(f'\n{"Algorithm":<25s} {"Total (s)":>10s} {"Avg/step (ms)":>14s} {"Steps":>6s}')
print('-' * 57)
for name, data in sorted(ptb_lm_results.items()):
    total = sum(t for _, t, _ in data)
    avg = total / len(data) * 1000
    print(f'{name:<25s} {total:10.2f} {avg:14.1f} {len(data):6d}')

# PTB logp agreement
names = sorted(ptb_lm_results.keys())
if len(names) >= 2:
    ref_name = names[0]
    ref_data = ptb_lm_results[ref_name]
    print(f'\nMax |logp| diff vs {ref_name}:')
    for name in names[1:]:
        data = ptb_lm_results[name]
        n = min(len(ref_data), len(data))
        if n > 0:
            diffs = [abs(ref_data[i][2] - data[i][2]) for i in range(n)]
            max_diff = max(diffs)
            print(f'  {name}: {max_diff:.6f}')

results['ptb_lm'] = {name: [(s, t, lp) for s, t, lp in data]
                      for name, data in ptb_lm_results.items()}

# ══════════════════════════════════════════════════════════════════════════════
# Section 4: PyniniTransducedLM Debugging (Issue #4)
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PYNINI TRANSDUCED LM DEBUG')
print('='*70)

try:
    from transduction.lm.pynini_transduced import PyniniTransducedLM
    from transduction.pynini_ops import PyniniPrecover

    # Test on BPE first (smaller, should work)
    print('\nPyniniTransducedLM on BPE:')
    t0 = time.perf_counter()
    pynini_bpe = PyniniTransducedLM(bpe_inner_lm, bpe_fst, K=10, max_expansions=200)
    t1 = time.perf_counter()
    print(f'  Constructor: {(t1-t0)*1000:.0f} ms')

    t0 = time.perf_counter()
    pstate = pynini_bpe.initial()
    t1 = time.perf_counter()
    print(f'  initial(): {(t1-t0)*1000:.0f} ms')

    for i in range(min(5, len(bpe_target_seq))):
        y = bpe_target_seq[i]
        t0 = time.perf_counter()
        lp = pstate.logp_next[y]
        pstate = pstate >> y
        t1 = time.perf_counter()
        print(f'  step {i+1}: {(t1-t0)*1000:.0f} ms  logp={lp:.4f}')

    # Test on PTB — SKIP: C++ code blocks SIGALRM, making step 1 hang
    # indefinitely. _compute_logp_next builds 257 per-symbol pynini precover
    # DFAs (one per output symbol). Each involves C++ composition which
    # blocks signal delivery, so timelimit() cannot interrupt it.
    print('\nPyniniTransducedLM on PTB:')
    print(f'  SKIPPED — step 1 requires {len(ptb_fst.B - {EPSILON})} pynini compositions')
    print(f'  (one per output symbol), each involving C++ code that blocks SIGALRM.')
    print(f'  Known to hang indefinitely (>10 min with no output).')
    print(f'  Root cause: PyniniTransducedState._compute_logp_next iterates over')
    print(f'  all {len(ptb_fst.B - {EPSILON})} target symbols calling pd.precover(extended)')
    print(f'  which is O(|B| * composition_cost) per step — prohibitive for PTB.')

    results['pynini_debug'] = 'ptb_skipped_known_hang'

except ImportError:
    print('  pynini not available, skipping')
    results['pynini_debug'] = 'import_error'

# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════

outdir = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'benchmark_results.json')

# Convert results to serializable form
def _serialize(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f'Not serializable: {type(obj)}')

with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, default=_serialize)
print(f'\nResults saved to {outpath}')
print('\nDone.')
