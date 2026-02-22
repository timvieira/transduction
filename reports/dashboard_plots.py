"""Generate plots for BENCHMARK_DASHBOARD.md — focused on scaling curves."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# Color palette
C_FUSED = '#5B8DB8'
C_RUST_TOKEN = '#E07B54'
C_TRANSDUCED = '#6BBF6B'
C_PTB = '#D4A843'
C_EXTRAPOLATE = '#999999'


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {path}')


# ── Plot 1: BPE Vocab Scaling — THE key plot ──

# Data from notes/bpe-lm-benchmark.ipynb vocab scaling cell
vocab_sizes =   [297,  529,  1023,  2020, 5011]
fused_avg_ms =  [7,    13,   31,    82,   291]
rust_token_ms = [25,   115,  618,   3862, None]  # 5011 timed out

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(vocab_sizes, fused_avg_ms, 'o-', color=C_FUSED, linewidth=2.5,
        markersize=8, label='FusedTransducedLM (rust)', zorder=5)

rt_vs = [v for v, t in zip(vocab_sizes, rust_token_ms) if t is not None]
rt_ts = [t for t in rust_token_ms if t is not None]
ax.plot(rt_vs, rt_ts, 's-', color=C_RUST_TOKEN, linewidth=2,
        markersize=7, label='FusedTransducedLM (rust_token)', zorder=4)

# Linear extrapolation from fused data to 50k
log_vs = np.log(vocab_sizes)
log_ts = np.log(fused_avg_ms)
slope, intercept = np.polyfit(log_vs, log_ts, 1)
extrap_vs = [10000, 20000, 50257]
extrap_ts = [np.exp(intercept + slope * np.log(v)) for v in extrap_vs]
ax.plot(extrap_vs, extrap_ts, 'x--', color=C_EXTRAPOLATE, linewidth=1.5,
        markersize=10, label=f'Extrapolation (slope={slope:.2f})', zorder=3)

# Mark GPT-2 full vocab
ax.axvline(50257, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
ax.text(50257, 2, 'GPT-2\n50,257', ha='right', fontsize=8, color='red', alpha=0.8)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Vocabulary size |V|')
ax.set_ylabel('Avg time per step (ms)')
ax.set_title('BPE Vocab Scaling: FusedTransducedLM (8 decode steps, CharNgramLM)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(200, 100000)
ax.set_ylim(3, 50000)
fig.tight_layout()
save(fig, 'bpe_vocab_scaling.png')


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


print('\nAll plots generated.')
