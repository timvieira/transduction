"""Benchmark CompiledBeam vs GeneralizedBeam.

Compares timing, region classification, and per-step cost on a range of
FSTs from the examples module plus BPE vocab scaling.

Usage:
    python reports/bench_compiled_beam.py
"""

import time
import gc
import numpy as np

from transduction import examples, FST
from transduction.fst import EPSILON
from transduction.lm.base import LM, LMState
from transduction.lm.ngram import CharNgramLM
from transduction.lm.compiled_beam import CompiledBeam, RegionAnalyzer
from transduction.lm.fused_transduced import FusedTransducedLM
from transduction.lm.generalized_beam import GeneralizedBeam
from transduction.util import LogDistr, set_memory_limit

set_memory_limit(4)


# ---------------------------------------------------------------------------
# Test LM
# ---------------------------------------------------------------------------

class TinyState(LMState):
    def __init__(self, probs, logprefix=0.0):
        self._probs = probs
        self.logprefix = logprefix
    @property
    def logp_next(self):
        return LogDistr(self._probs)
    @property
    def eos(self):
        return '<EOS>'
    def __rshift__(self, token):
        lp = self._probs.get(token, -np.inf)
        return TinyState(self._probs, self.logprefix + lp)

class TinyLM(LM):
    def __init__(self):
        self.eos = '<EOS>'
    def initial(self):
        return TinyState({'a': np.log(0.6), 'b': np.log(0.3), '<EOS>': np.log(0.1)})


def copy_fst(alphabet):
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    for x in alphabet:
        fst.add_arc(0, x, x, 0)
    return fst


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def benchmark_decode(alg, name, n_steps=5, n_warmup=1, n_runs=3):
    """Time n_steps decode steps, return (avg_init_ms, avg_step_ms, total_ms)."""
    init_times = []
    step_times_all = []

    for run in range(n_warmup + n_runs):
        t0 = time.perf_counter()
        state = alg.initial()
        t_init = time.perf_counter() - t0

        step_times = []
        for i in range(n_steps):
            t0 = time.perf_counter()
            lp = state.logp_next
            sym = lp.argmax()
            if sym == state.eos:
                break
            state = state >> sym
            step_times.append(time.perf_counter() - t0)

        if run >= n_warmup:
            init_times.append(t_init * 1000)
            step_times_all.append(step_times)

    avg_init = np.median(init_times)
    # average per-step across all runs
    all_steps = [t * 1000 for run_times in step_times_all for t in run_times]
    avg_step = np.mean(all_steps) if all_steps else 0
    total = avg_init + sum(np.mean([t * 1000 for t in run_times]) * len(run_times)
                           for run_times in step_times_all) / n_runs

    return {
        'name': name,
        'init_ms': round(float(avg_init), 3),
        'avg_step_ms': round(float(avg_step), 3),
        'total_ms': round(float(total), 3),
        'n_steps': len(all_steps) // n_runs if n_runs else 0,
    }


# ---------------------------------------------------------------------------
# Part 1: Small FSTs — CompiledBeam vs FusedTransducedLM (all-wild baseline)
# ---------------------------------------------------------------------------

print("=" * 70)
print("Part 1: Small FSTs — CompiledBeam vs FusedTransducedLM (all-wild)")
print("=" * 70)
print()

SMALL_FSTS = [
    ("copy_ab",           lambda: copy_fst(['a', 'b'])),
    ("small",             examples.small),
    ("lowercase",         examples.lowercase),
    ("delete_b",          examples.delete_b),
    ("triplets_of_doom",  examples.triplets_of_doom),
    ("two_hub_alt",       examples.two_hub_alternating),
    ("hub_with_escape",   examples.hub_with_escape),
    ("multi_hub_chain_3", lambda: examples.multi_hub_chain(n=3)),
    ("no_hub",            examples.no_hub_transducer),
    ("bpe_like_20",       lambda: examples.bpe_like(vocab_size=20)),
    ("newspeak2",         examples.newspeak2),
    ("togglecase",        examples.togglecase),
    ("weird_copy",        examples.weird_copy),
]

K = 50
MAX_BEAM = 100
MAX_STEPS = 1000
N_DECODE_STEPS = 5

header = f"{'FST':<22s}  {'Regions':>8s}  {'CB init':>8s}  {'CB step':>8s}  {'Fused init':>10s}  {'Fused step':>10s}  {'Speedup':>8s}"
print(header)
print("-" * len(header))

results = []

for name, fst_fn in SMALL_FSTS:
    inner_lm = TinyLM()
    fst = fst_fn()

    # Region analysis
    rmap = RegionAnalyzer(fst, inner_lm).analyze()
    n_hubs = len(rmap.hub_regions)
    has_wild = rmap.wild_region is not None
    region_str = f"{n_hubs}H" + ("+W" if has_wild else "")

    # Benchmark CompiledBeam
    cb = CompiledBeam(inner_lm, fst, K=K, max_beam=MAX_BEAM, max_steps=MAX_STEPS)
    cb_result = benchmark_decode(cb, 'CompiledBeam', n_steps=N_DECODE_STEPS)

    # Benchmark FusedTransducedLM (all-wild baseline)
    fused = FusedTransducedLM(inner_lm, fst, max_steps=MAX_STEPS, max_beam=MAX_BEAM,
                              helper="python")
    fused_result = benchmark_decode(fused, 'Fused', n_steps=N_DECODE_STEPS)

    speedup = fused_result['avg_step_ms'] / cb_result['avg_step_ms'] if cb_result['avg_step_ms'] > 0 else float('inf')

    print(f"{name:<22s}  {region_str:>8s}  "
          f"{cb_result['init_ms']:7.1f}ms  {cb_result['avg_step_ms']:7.3f}ms  "
          f"{fused_result['init_ms']:9.1f}ms  {fused_result['avg_step_ms']:9.3f}ms  "
          f"{speedup:7.2f}x")

    results.append({
        'fst': name,
        'regions': region_str,
        'n_hubs': n_hubs,
        'has_wild': has_wild,
        **{f'cb_{k}': v for k, v in cb_result.items()},
        **{f'fused_{k}': v for k, v in fused_result.items()},
        'speedup': round(speedup, 2),
    })

    gc.collect()

print()


# ---------------------------------------------------------------------------
# Part 2: BPE Vocab Scaling
# ---------------------------------------------------------------------------

print("=" * 70)
print("Part 2: BPE Vocab Scaling")
print("=" * 70)
print()

VOCAB_SIZES = [10, 20, 50, 100, 200]

header2 = f"{'|V|':>5s}  {'CB init':>8s}  {'CB step':>8s}  {'Fused init':>10s}  {'Fused step':>10s}  {'Speedup':>8s}  {'Regions':>8s}"
print(header2)
print("-" * len(header2))

bpe_results = []

for vs in VOCAB_SIZES:
    fst = examples.bpe_like(vocab_size=vs)
    source_alpha = fst.A - {EPSILON}
    inner_lm = CharNgramLM.train(
        [list("aabbaabb" * 3)], n=2, alpha=0.5, alphabet=source_alpha
    )

    rmap = RegionAnalyzer(fst, inner_lm).analyze()
    n_hubs = len(rmap.hub_regions)
    has_wild = rmap.wild_region is not None
    region_str = f"{n_hubs}H" + ("+W" if has_wild else "")

    cb = CompiledBeam(inner_lm, fst, K=10, max_beam=10, max_steps=200)
    cb_r = benchmark_decode(cb, 'CB', n_steps=5, n_warmup=1, n_runs=3)

    fused = FusedTransducedLM(inner_lm, fst, max_steps=200, max_beam=10,
                              helper="python")
    fused_r = benchmark_decode(fused, 'Fused', n_steps=5, n_warmup=1, n_runs=3)

    speedup = fused_r['avg_step_ms'] / cb_r['avg_step_ms'] if cb_r['avg_step_ms'] > 0 else float('inf')

    print(f"{vs:>5d}  "
          f"{cb_r['init_ms']:7.1f}ms  {cb_r['avg_step_ms']:7.3f}ms  "
          f"{fused_r['init_ms']:7.1f}ms  {fused_r['avg_step_ms']:7.3f}ms  "
          f"{speedup:7.2f}x  {region_str:>8s}")

    bpe_results.append({
        'vocab_size': vs,
        'regions': region_str,
        'cb_init_ms': cb_r['init_ms'],
        'cb_step_ms': cb_r['avg_step_ms'],
        'fused_init_ms': fused_r['init_ms'],
        'fused_step_ms': fused_r['avg_step_ms'],
        'speedup': round(speedup, 2),
    })

    gc.collect()

print()


# ---------------------------------------------------------------------------
# Part 3: Region analysis summary across all example FSTs
# ---------------------------------------------------------------------------

print("=" * 70)
print("Part 3: Region Analysis Across All Example FSTs")
print("=" * 70)
print()

ALL_FSTS = [
    ("copy_ab",           lambda: copy_fst(['a', 'b'])),
    ("small",             examples.small),
    ("lowercase",         examples.lowercase),
    ("delete_b",          examples.delete_b),
    ("triplets_of_doom",  examples.triplets_of_doom),
    ("two_hub_alt",       examples.two_hub_alternating),
    ("hub_with_escape",   examples.hub_with_escape),
    ("multi_hub_chain_3", lambda: examples.multi_hub_chain(n=3)),
    ("multi_hub_chain_5", lambda: examples.multi_hub_chain(n=5)),
    ("no_hub",            examples.no_hub_transducer),
    ("bpe_like_10",       lambda: examples.bpe_like(vocab_size=10)),
    ("bpe_like_50",       lambda: examples.bpe_like(vocab_size=50)),
    ("bpe_like_200",      lambda: examples.bpe_like(vocab_size=200)),
    ("weird_copy",        examples.weird_copy),
    ("togglecase",        examples.togglecase),
    ("newspeak2",         examples.newspeak2),
]

header3 = f"{'FST':<22s}  {'States':>6s}  {'Arcs':>6s}  {'Hubs':>5s}  {'Wild':>5s}  {'Hub tokens':>10s}"
print(header3)
print("-" * len(header3))

for fname, fst_fn in ALL_FSTS:
    inner_lm = TinyLM()
    fst = fst_fn()

    rmap = RegionAnalyzer(fst, inner_lm).analyze()

    n_states = len(fst.states)
    n_arcs = sum(len(fst.arcs(s)) for s in fst.states)
    n_hubs = len(rmap.hub_regions)
    has_wild = 'Y' if rmap.wild_region is not None else 'N'

    hub_tokens = sum(len(r.trie._source_syms) for r in rmap.hub_regions.values())
    hub_tok_str = str(hub_tokens) if hub_tokens > 0 else '-'

    print(f"{fname:<22s}  {n_states:>6d}  {n_arcs:>6d}  {n_hubs:>5d}  {has_wild:>5s}  {hub_tok_str:>10s}")

print()
print("Done.")
