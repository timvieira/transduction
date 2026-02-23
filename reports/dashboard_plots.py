"""Generate plots for BENCHMARK_DASHBOARD.md — focused on scaling curves.

Reads BPE vocab scaling data from bench_vectorization_results.json
(produced by bench_vectorization.py) and factored arena comparison data
from bench_factored_results.json.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)
DATAFILE = os.path.join(os.path.dirname(__file__), 'bench_vectorization_results.json')
FACTORED_FILE = os.path.join(os.path.dirname(__file__), 'bench_factored_results.json')

# Color palette
C_FUSED = '#5B8DB8'
C_FUSED_MEM = '#2E6B9E'
C_RUST_TOKEN = '#E07B54'
C_TRANSDUCED = '#6BBF6B'
C_PTB = '#D4A843'
C_EXTRAPOLATE = '#999999'
C_CHARBEAM = '#9B59B6'   # character beam (purple)
C_FLAT = '#B85B5B'       # flat arena (old)
C_FACTORED = '#5BB88D'   # factored arena (new)


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {path}')


# ── Load benchmark data ──

with open(DATAFILE) as f:
    bench_data = json.load(f)

rows = bench_data['rows']
baseline_rss = bench_data['baseline_rss_mb']

# ── Extract per-method data ──

# Discover method keys from the first row
method_keys = [k.replace('_avg_ms', '') for k in rows[0] if k.endswith('_avg_ms')]

method_data = {}  # key -> (vs, ms)
for key in method_keys:
    vs = [r['vocab_size'] for r in rows if r.get(f'{key}_avg_ms') is not None]
    ms = [r[f'{key}_avg_ms'] for r in rows if r.get(f'{key}_avg_ms') is not None]
    method_data[key] = (vs, ms)

# Primary method (rust) for memory + extrapolation
primary = method_keys[0]
vs_ok, ms_ok = method_data[primary]
rss_ok = [r['peak_rss_mb'] for r in rows if r.get(f'{primary}_avg_ms') is not None]


# ── Plot 1: BPE Vocab Scaling — time + memory (side by side) ──

# Compute delta memory (benchmark overhead above baseline)
delta_mb = [rss - baseline_rss for rss in rss_ok]

# Log-log fits for primary method
log_vs = np.log(vs_ok)
log_ts = np.log(ms_ok)
time_slope, time_intercept = np.polyfit(log_vs, log_ts, 1)

# For memory fit, skip tiny deltas where baseline dominates
mem_fit_mask = [d > 10 for d in delta_mb]
log_vs_mem = np.log([v for v, m in zip(vs_ok, mem_fit_mask) if m])
log_dm = np.log([d for d, m in zip(delta_mb, mem_fit_mask) if m])
mem_slope, mem_intercept = np.polyfit(log_vs_mem, log_dm, 1)

extrap_vs = np.array([v for v in [15000, 20000, 30000, 50257] if v > max(vs_ok)])
extrap_ts = np.exp(time_intercept + time_slope * np.log(extrap_vs))
extrap_dm = np.exp(mem_intercept + mem_slope * np.log(extrap_vs))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left panel: time — all methods
method_styles = {
    'FusedLM_rust':       ('o-', C_FUSED, 2.5, 8, 'helper="rust"'),
    'FusedLM_rust_token': ('s-', C_RUST_TOKEN, 2, 7, 'helper="rust_token"'),
    'CharacterBeam':      ('D-', C_CHARBEAM, 2, 7, 'CharacterBeam (K=10)'),
}
for key in method_keys:
    vs, ms = method_data[key]
    style = method_styles.get(key, ('o-', C_EXTRAPOLATE, 1.5, 6, key))
    ax1.plot(vs, ms, style[0], color=style[1], linewidth=style[2],
             markersize=style[3], label=style[4], zorder=5)

if len(extrap_vs) > 0:
    ax1.plot(extrap_vs, extrap_ts, 'x--', color=C_EXTRAPOLATE, linewidth=1.5,
             markersize=10, label=f'Extrapolation (|V|$^{{{time_slope:.2f}}}$)', zorder=2)
ax1.axvline(50257, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
ax1.text(50257, 3, 'GPT-2\n50,257', ha='right', fontsize=8, color='red', alpha=0.8)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Vocabulary size |V|')
ax1.set_ylabel('Avg time per step (ms)')
ax1.set_title('Time scaling')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(200, 100000)
ax1.set_ylim(3, 50000)

# Right panel: memory (delta above baseline)
ax2.plot(vs_ok, delta_mb, 's-', color=C_FUSED_MEM, linewidth=2.5,
         markersize=8, label='FusedLM (peak RSS - baseline)', zorder=5)
if len(extrap_vs) > 0:
    ax2.plot(extrap_vs, extrap_dm, 'x--', color=C_EXTRAPOLATE, linewidth=1.5,
             markersize=10, label=f'Extrapolation (|V|$^{{{mem_slope:.2f}}}$)', zorder=2)

# CharacterBeam memory (tracemalloc peak allocation)
cb_mem_vs = [r['vocab_size'] for r in rows if r.get('CharacterBeam_peak_mb') is not None]
cb_mem_mb = [r['CharacterBeam_peak_mb'] for r in rows if r.get('CharacterBeam_peak_mb') is not None]
if cb_mem_vs:
    ax2.plot(cb_mem_vs, cb_mem_mb, 'D-', color=C_CHARBEAM, linewidth=2,
             markersize=7, label='CharacterBeam (tracemalloc)', zorder=5)
mem_limit_mb = bench_data['config']['memory_limit_gb'] * 1024
ax2.axhline(mem_limit_mb - baseline_rss, color='orange', linestyle=':', alpha=0.6, linewidth=1.5)
ax2.text(250, (mem_limit_mb - baseline_rss) * 1.1,
         f'{bench_data["config"]["memory_limit_gb"]} GB limit ({mem_limit_mb - baseline_rss:.0f} MB avail)',
         fontsize=7, color='orange', alpha=0.8)
ax2.axvline(50257, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
ax2.text(50257, 3, 'GPT-2\n50,257', ha='right', fontsize=8, color='red', alpha=0.8)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Vocabulary size |V|')
ax2.set_ylabel('Peak memory delta (MB)')
ax2.set_title('Memory scaling')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(200, 100000)
ax2.set_ylim(1, 100000)

fig.suptitle('BPE Vocab Scaling: FusedTransducedLM (8 decode steps, CharNgramLM)',
             fontsize=11, y=1.02)
fig.tight_layout()
save(fig, 'bpe_vocab_scaling.png')

# Print extrapolation summary
print(f'\n  Time ({primary}): |V|^{time_slope:.2f}')
print(f'  Memory (delta): |V|^{mem_slope:.2f}')
print(f'  Baseline RSS: {baseline_rss} MB')
if len(extrap_vs) > 0:
    for v, t, dm in zip(extrap_vs, extrap_ts, extrap_dm):
        total = dm + baseline_rss
        print(f'    V={v:>6.0f}:  ~{t:,.0f} ms/step,  ~{dm:,.0f} MB delta  (~{total/1024:.1f} GB total)')



# ── Plot 2: PTB TransducedLM variants ──

variants = ['TransducedLM\n(Rust peekaboo)', 'FusedTransducedLM\n(single-pass)']
ms_per_step = [129, 66]

fig, ax = plt.subplots(figsize=(5, 3.5))
colors_lm = [C_TRANSDUCED, C_FUSED]
bars = ax.bar(range(2), ms_per_step, color=colors_lm, width=0.5)
for i, v in enumerate(ms_per_step):
    ax.text(i, v + 3, f'{v} ms', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(2))
ax.set_xticklabels(variants, fontsize=8)
ax.set_ylabel('ms / step')
ax.set_title('PTB TransducedLM (K=20, CharNgramLM, 45 steps)')
ax.set_ylim(0, 170)
ax.grid(axis='y', alpha=0.3)
ax.annotate('2.0x', xy=(1, ms_per_step[1]), xytext=(0.5, 120),
            fontsize=11, fontweight='bold', color=C_FUSED,
            arrowprops=dict(arrowstyle='->', color=C_FUSED, lw=1.5))
fig.tight_layout()
save(fig, 'ptb_transduced_lm.png')


# ── Plot 3: PTB Decomposition backends ──

methods = ['Standard\n(Python)', 'Rust']
times = [1651, 110]  # ms (approx geomean)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
colors3 = [C_TRANSDUCED, C_FUSED]
bars = ax.bar(methods, times, color=colors3, width=0.45)
ax.set_ylabel('Time (ms)')
ax.set_title('PTB Decomposition Backend (geomean)')
ax.set_yscale('log')
ax.set_ylim(10, 5000)
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.3,
            f'{val} ms', ha='center', fontsize=10, fontweight='bold')
ax.text(1, 30, '15x', ha='center', fontsize=10, fontweight='bold', color=C_FUSED)
ax.grid(axis='y', alpha=0.3, which='both')
fig.tight_layout()
save(fig, 'ptb_backends.png')


# ── Plot 4: Factored Arena — memory + time comparison ──

if os.path.exists(FACTORED_FILE):
    with open(FACTORED_FILE) as f:
        factored_data = json.load(f)

    factored_rows = [r for r in factored_data['rows'] if r['avg_ms'] is not None]
    factored_baseline = factored_data['baseline_rss_mb']

    # Factored arena data
    fv = [r['vocab_size'] for r in factored_rows]
    f_ms = [r['avg_ms'] for r in factored_rows]
    f_rss = [r['peak_rss_mb'] for r in factored_rows]
    f_delta = [rss - factored_baseline for rss in f_rss]

    # Flat arena data (from main benchmark — use FusedLM_rust which includes
    # decomposition; memory is dominated by arena so delta RSS is comparable)
    flat_vs = vs_ok
    flat_delta = delta_mb
    flat_ms = ms_ok

    # Log-log fits for memory
    f_mem_mask = [d > 5 for d in f_delta]
    if sum(f_mem_mask) >= 2:
        f_log_vs = np.log([v for v, m in zip(fv, f_mem_mask) if m])
        f_log_dm = np.log([d for d, m in zip(f_delta, f_mem_mask) if m])
        f_mem_slope, f_mem_intercept = np.polyfit(f_log_vs, f_log_dm, 1)
    else:
        f_mem_slope = None

    flat_mem_mask = [d > 5 for d in flat_delta]
    if sum(flat_mem_mask) >= 2:
        o_log_vs = np.log([v for v, m in zip(flat_vs, flat_mem_mask) if m])
        o_log_dm = np.log([d for d, m in zip(flat_delta, flat_mem_mask) if m])
        o_mem_slope, o_mem_intercept = np.polyfit(o_log_vs, o_log_dm, 1)
    else:
        o_mem_slope = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: memory comparison
    flat_label = f'Flat arena (|V|$^{{{o_mem_slope:.2f}}}$)' if o_mem_slope else 'Flat arena'
    fac_label = f'Factored arena (|V|$^{{{f_mem_slope:.2f}}}$)' if f_mem_slope else 'Factored arena'

    ax1.plot(flat_vs, flat_delta, 's-', color=C_FLAT, linewidth=2.5,
             markersize=8, label=flat_label, zorder=5)
    ax1.plot(fv, f_delta, 'o-', color=C_FACTORED, linewidth=2.5,
             markersize=8, label=fac_label, zorder=5)

    # Extrapolation lines
    extrap_x = np.linspace(np.log(200), np.log(60000), 100)
    if o_mem_slope:
        ax1.plot(np.exp(extrap_x), np.exp(o_mem_intercept + o_mem_slope * extrap_x),
                 '--', color=C_FLAT, linewidth=1, alpha=0.4, zorder=2)
    if f_mem_slope:
        ax1.plot(np.exp(extrap_x), np.exp(f_mem_intercept + f_mem_slope * extrap_x),
                 '--', color=C_FACTORED, linewidth=1, alpha=0.4, zorder=2)

    ax1.axvline(50257, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    ax1.text(50257, 3, 'GPT-2\n50,257', ha='right', fontsize=8, color='red', alpha=0.8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Vocabulary size |V|')
    ax1.set_ylabel('Peak memory delta (MB)')
    ax1.set_title('Memory scaling')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(200, 100000)
    ax1.set_ylim(1, 100000)

    # Right panel: time comparison
    # Note: flat data is FusedTransducedLM (decomp + LM), factored is decomp-only.
    # Both use CharNgramLM which is ~free, so they're roughly comparable.
    ax2.plot(flat_vs, flat_ms, 's-', color=C_FLAT, linewidth=2.5,
             markersize=8, label='Flat arena (FusedLM)', zorder=5)
    ax2.plot(fv, f_ms, 'o-', color=C_FACTORED, linewidth=2.5,
             markersize=8, label='Factored arena (decomp)', zorder=5)

    ax2.axvline(50257, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    ax2.text(50257, 3, 'GPT-2\n50,257', ha='right', fontsize=8, color='red', alpha=0.8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Vocabulary size |V|')
    ax2.set_ylabel('Avg time per step (ms)')
    ax2.set_title('Time scaling')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(200, 100000)
    ax2.set_ylim(3, 50000)

    fig.suptitle('Factored Arena vs Flat Arena (GPT-2 BPE, 8 decode steps)',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, 'factored_arena_scaling.png')

    # Print summary
    print(f'\n  Factored arena memory: |V|^{f_mem_slope:.2f}' if f_mem_slope else '')
    print(f'  Flat arena memory:     |V|^{o_mem_slope:.2f}' if o_mem_slope else '')

    # Compute reduction at common vocab sizes
    print('\n  Memory reduction at measured points:')
    for r in factored_rows:
        v = r['vocab_size']
        f_d = r['peak_rss_mb'] - factored_baseline
        # Find closest flat point
        closest = min(rows, key=lambda x: abs(x['vocab_size'] - v))
        o_d = closest['peak_rss_mb'] - baseline_rss
        if o_d > 10 and f_d > 0:
            pct = (1 - f_d / o_d) * 100
            print(f'    V={v:>6d}: {o_d:>5.0f} MB (flat) -> {f_d:>5.0f} MB (factored)  = {pct:+.0f}%')

else:
    print(f'\n  No factored arena data found at {FACTORED_FILE} — skipping Plot 4')


print('\nAll plots generated.')
