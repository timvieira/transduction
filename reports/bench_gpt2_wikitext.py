"""Benchmark CharacterBeam and GeneralizedBeam with GPT-2 on WikiText.

Compares prefetch (batched forward passes) vs no-prefetch (sequential)
for both methods on a real HuggingFace LM with full BPE vocab.

Usage:
    python reports/bench_gpt2_wikitext.py
"""

import json
import os
import resource
import time
import gc

import numpy as np
from transduction.fst import FST
from transduction.fsa import EPSILON
from transduction.lm.huggingface_lm import HfTokenizerVocab, load_model_by_name
from transduction.lm.character_beam import CharacterBeam
from transduction.lm.generalized_beam import GeneralizedBeam
from transduction.applications.bpe import bpe_wfst
from transduction.util import set_memory_limit

set_memory_limit(10)

OUTFILE = os.path.join(os.path.dirname(__file__), 'bench_gpt2_wikitext_results.json')
NUM_BYTES = 200
K = 10
REPORT_EVERY = 20


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class _NoPrefetchWrapper:
    """Thin wrapper that disables prefetch for A/B comparison."""
    def __init__(self, real_lm):
        self._real = real_lm
    def __getattr__(self, name):
        return getattr(self._real, name)
    def prefetch(self, states):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading GPT-2...")
t0 = time.perf_counter()
lm = load_model_by_name('gpt2')
print(f"  Loaded in {time.perf_counter() - t0:.1f}s on {lm.device}")

tokenizer = lm.tokenizer
hf_vocab = HfTokenizerVocab(tokenizer)
drop = {x.encode() for x in tokenizer.all_special_tokens}
eos_bytes = tokenizer.eos_token.encode()
eos_id = lm._encode[eos_bytes]

# ---- WikiText text ----
print("Loading WikiText...")
try:
    from transduction.applications.wikitext import load_wikitext, wikitext_detokenize
    ds = load_wikitext('test')
    raw = ' '.join(row['text'] for row in ds if row['text'].strip())
    text = wikitext_detokenize(raw)
except Exception as e:
    print(f"  WikiText load failed ({e}), using fallback text")
    text = (
        "The tower is 324 metres (1,063 ft) tall, about the same height as "
        "an 81-storey building, and the tallest structure in Paris. Its base "
        "is square, measuring 125 metres (410 ft) on each side. During its "
        "construction, the Eiffel Tower surpassed the Washington Monument to "
        "become the tallest man-made structure in the world."
    )

target_bytes = list(text.encode('utf-8')[:NUM_BYTES])
print(f"  Target: {NUM_BYTES} bytes")
print(f"  Text: {text[:60]!r}...")

# ---- CB vocab (full GPT-2 vocab) ----
cb_vocab: dict = {}
for i, bs in enumerate(hf_vocab.decode):
    if bs is not None and bs not in drop:
        cb_vocab[i] = bs
if eos_id not in cb_vocab:
    cb_vocab[eos_id] = eos_bytes
print(f"  Vocab: {len(cb_vocab)} tokens")

# ---- BPE FST (for GeneralizedBeam) ----
print("Building BPE WFST...")
t0 = time.perf_counter()
fst = bpe_wfst(tokenizer)
print(f"  Built in {time.perf_counter() - t0:.1f}s: {len(fst.states)} states")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(name, make_alg):
    gc.collect()
    lm._calls = 0

    t_init = time.perf_counter()
    alg = make_alg()
    init_ms = (time.perf_counter() - t_init) * 1000

    calls_after_init = lm._calls
    print(f"\n  {name}")
    print(f"  Init: {init_ms:.0f} ms, LM calls: {calls_after_init}")

    state = alg.initial()
    step_times = []
    step_calls = []

    for i, yb in enumerate(target_bytes):
        calls_before = lm._calls
        t0 = time.perf_counter()
        try:
            lp = state.logp_next
            if yb not in lp:
                print(f"  byte {yb} not in logp_next at step {i}, stopping")
                break
            state = state >> yb
        except Exception as e:
            print(f"  step {i}: {type(e).__name__}: {e}")
            break
        elapsed_ms = (time.perf_counter() - t0) * 1000
        step_times.append(elapsed_ms)
        step_calls.append(lm._calls - calls_before)

        if (i + 1) % REPORT_EVERY == 0:
            avg = np.mean(step_times)
            total_calls = lm._calls - calls_after_init
            print(f"    step {i+1:3d}: avg {avg:.1f} ms/step, {total_calls} LM calls")

    total_ms = sum(step_times)
    total_calls = lm._calls - calls_after_init
    avg_ms = float(np.mean(step_times)) if step_times else 0

    print(f"  Done: {len(step_times)} steps, {total_ms/1000:.2f}s total, "
          f"{avg_ms:.1f} ms/step, {total_calls} LM calls "
          f"({total_calls/max(len(step_times),1):.1f}/step)")

    return {
        'name': name,
        'init_ms': round(init_ms),
        'steps': len(step_times),
        'total_ms': round(total_ms),
        'avg_ms': round(avg_ms, 1),
        'lm_calls': total_calls,
        'calls_per_step': round(total_calls / max(len(step_times), 1), 2),
        'peak_rss_mb': round(peak_rss_mb()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"CharacterBeam + GPT-2 ({NUM_BYTES} bytes, K={K})")
print(f"{'='*60}")

results = []

results.append(run_benchmark(
    'CharacterBeam (prefetch)',
    lambda: CharacterBeam(lm, cb_vocab, K=K, eos_token=eos_id),
))

results.append(run_benchmark(
    'CharacterBeam (no prefetch)',
    lambda: CharacterBeam(_NoPrefetchWrapper(lm), cb_vocab, K=K, eos_token=eos_id),
))

print(f"\n{'='*60}")
print(f"GeneralizedBeam + GPT-2 ({NUM_BYTES} bytes, K={K})")
print(f"{'='*60}")

results.append(run_benchmark(
    'GeneralizedBeam (prefetch)',
    lambda: GeneralizedBeam(lm, fst, K=K, max_beam=K, max_steps=200),
))

results.append(run_benchmark(
    'GeneralizedBeam (no prefetch)',
    lambda: GeneralizedBeam(_NoPrefetchWrapper(lm), fst, K=K, max_beam=K, max_steps=200),
))


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}\n")

header = f"{'Method':<35s} {'ms/step':>8s} {'Total':>7s} {'Calls/step':>11s}"
print(header)
print("─" * len(header))
for r in results:
    print(f"{r['name']:<35s} {r['avg_ms']:>8.1f} {r['total_ms']/1000:>6.1f}s "
          f"{r['calls_per_step']:>11.1f}")

output = {
    'description': f'GPT-2 + WikiText benchmark ({NUM_BYTES} bytes)',
    'config': {'num_bytes': NUM_BYTES, 'K': K, 'device': str(lm.device)},
    'results': results,
}
with open(OUTFILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nWrote {OUTFILE}")
