"""Benchmark CharacterBeam and GeneralizedBeam with GPT-2 on WikiText.

Compares prefetch (batched forward passes) vs no-prefetch (sequential)
for both methods on a real HuggingFace LM with full BPE vocab.

CharacterBeam is tested with extend_threshold=0.1 (which reduces LM calls
from ~7.7/step to ~0.27/step) and without threshold.

Usage:
    python reports/bench_gpt2_wikitext.py
"""

import json
import os
import resource
import time
import gc

import numpy as np
from transduction.lm.huggingface_lm import HfTokenizerVocab, load_model_by_name
from transduction.lm.character_beam import CharacterBeam
from transduction.lm.generalized_beam import GeneralizedBeam
from transduction.applications.bpe import bpe_wfst
from transduction.util import set_memory_limit

set_memory_limit(10)

OUTFILE = os.path.join(os.path.dirname(__file__), 'bench_gpt2_wikitext_results.json')
NUM_BYTES = 200
K = 10
REPORT_EVERY = 10


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

print("Loading GPT-2...", flush=True)
t0 = time.perf_counter()
lm = load_model_by_name('gpt2')
print(f"  Loaded in {time.perf_counter() - t0:.1f}s on {lm.device}", flush=True)

tokenizer = lm.tokenizer
hf_vocab = HfTokenizerVocab(tokenizer)
drop = {x.encode() for x in tokenizer.all_special_tokens}
eos_bytes = tokenizer.eos_token.encode()
eos_id = lm._encode[eos_bytes]

# ---- WikiText text ----
print("Loading WikiText...", flush=True)
try:
    from transduction.applications.wikitext import load_wikitext, wikitext_detokenize
    ds = load_wikitext('test')
    raw = ' '.join(row['text'] for row in ds if row['text'].strip())
    text = wikitext_detokenize(raw)
except Exception as e:
    print(f"  WikiText load failed ({e}), using fallback text", flush=True)
    text = (
        "The tower is 324 metres (1,063 ft) tall, about the same height as "
        "an 81-storey building, and the tallest structure in Paris. Its base "
        "is square, measuring 125 metres (410 ft) on each side. During its "
        "construction, the Eiffel Tower surpassed the Washington Monument to "
        "become the tallest man-made structure in the world."
    )

target_bytes = list(text.encode('utf-8')[:NUM_BYTES])
preview = repr(text[:60])
print(f"  Target: {NUM_BYTES} bytes", flush=True)
print(f"  Text: {preview}...", flush=True)

# ---- CB vocab (full GPT-2 vocab) ----
cb_vocab: dict = {}
for i, bs in enumerate(hf_vocab.decode):
    if bs is not None and bs not in drop:
        cb_vocab[i] = bs
if eos_id not in cb_vocab:
    cb_vocab[eos_id] = eos_bytes
print(f"  Vocab: {len(cb_vocab)} tokens", flush=True)

# ---- BPE FST (for GeneralizedBeam) ----
print("Building BPE WFST...", flush=True)
t0 = time.perf_counter()
fst = bpe_wfst(tokenizer)
print(f"  Built in {time.perf_counter() - t0:.1f}s: {len(fst.states)} states", flush=True)


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
    print(f"\n=== {name} ===", flush=True)
    print(f"  Init: {init_ms:.0f} ms, LM calls: {calls_after_init}", flush=True)

    state = alg.initial()
    step_times = []

    for i, yb in enumerate(target_bytes):
        t0 = time.perf_counter()
        try:
            lp = state.logp_next
            if yb not in lp:
                print(f"  byte {yb} not in logp_next at step {i}, stopping", flush=True)
                break
            state = state >> yb
        except Exception as e:
            print(f"  step {i}: {type(e).__name__}: {e}", flush=True)
            break
        elapsed_ms = (time.perf_counter() - t0) * 1000
        step_times.append(elapsed_ms)

        if (i + 1) % REPORT_EVERY == 0:
            avg = np.mean(step_times)
            total_calls = lm._calls - calls_after_init
            print(f"  {i+1:3d}/{NUM_BYTES}  avg={avg:5.0f} ms/step  "
                  f"calls={total_calls} ({total_calls/(i+1):.1f}/step)", flush=True)

    total_ms = sum(step_times)
    total_calls = lm._calls - calls_after_init
    avg_ms = float(np.mean(step_times)) if step_times else 0

    print(f"  TOTAL: {len(step_times)} steps, {total_ms/1000:.1f}s, "
          f"{avg_ms:.1f} ms/step, {total_calls} calls "
          f"({total_calls/max(len(step_times),1):.2f}/step)", flush=True)

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

print(f"\n{'='*60}", flush=True)
print(f"CharacterBeam + GPT-2 ({NUM_BYTES} bytes, K={K})", flush=True)
print(f"{'='*60}", flush=True)

results = []

results.append(run_benchmark(
    'CharacterBeam (prefetch, threshold=0.1)',
    lambda: CharacterBeam(lm, cb_vocab, K=K, eos_token=eos_id, extend_threshold=0.1),
))

results.append(run_benchmark(
    'CharacterBeam (no prefetch, threshold=0.1)',
    lambda: CharacterBeam(_NoPrefetchWrapper(lm), cb_vocab, K=K, eos_token=eos_id,
                          extend_threshold=0.1),
))

print(f"\n{'='*60}", flush=True)
print(f"GeneralizedBeam + GPT-2 ({NUM_BYTES} bytes, K={K})", flush=True)
print(f"{'='*60}", flush=True)

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

print(f"\n{'='*60}", flush=True)
print("Summary", flush=True)
print(f"{'='*60}\n", flush=True)

header = f"{'Method':<45s} {'ms/step':>8s} {'Total':>7s} {'Calls/step':>11s}"
print(header, flush=True)
print("─" * len(header), flush=True)
for r in results:
    print(f"{r['name']:<45s} {r['avg_ms']:>8.1f} {r['total_ms']/1000:>6.1f}s "
          f"{r['calls_per_step']:>11.2f}", flush=True)

output = {
    'description': f'GPT-2 + WikiText benchmark ({NUM_BYTES} bytes)',
    'config': {'num_bytes': NUM_BYTES, 'K': K, 'device': str(lm.device)},
    'results': results,
}
with open(OUTFILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nWrote {OUTFILE}", flush=True)
