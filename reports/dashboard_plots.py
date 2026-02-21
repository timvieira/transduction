"""Generate plots for BENCHMARK_DASHBOARD.md"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# Color palette
C_STANDARD = '#5B8DB8'
C_PYNINI = '#E07B54'
C_RUST = '#6BBF6B'
C_FUSED = '#D4A843'


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {path}')


# ── Plot 1: BPE decomposition backends across prefix lengths ──

lengths = [3, 5, 8, 10, 15, 20, 30, 40]
standard = [3.8, 3.7, 3.3, 5.2, 3.6, 5.7, 4.6, 6.8]
pynini   = [0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
rust     = [1.0, 1.5, 0.8, 0.9, 0.8, 0.9, 0.9, 0.9]

fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(lengths))
w = 0.25
ax.bar(x - w, standard, w, label='Standard (Python)', color=C_STANDARD)
ax.bar(x,     rust,     w, label='Rust',              color=C_RUST)
ax.bar(x + w, pynini,   w, label='Pynini',            color=C_PYNINI)
ax.set_xticks(x)
ax.set_xticklabels(lengths)
ax.set_xlabel('Target prefix length')
ax.set_ylabel('Time (ms, best of 3)')
ax.set_title('BPE Decomposition: Backend Comparison')
ax.legend(fontsize=8)
ax.set_ylim(0, max(standard) * 1.2)
ax.grid(axis='y', alpha=0.3)
save(fig, 'bpe_backends.png')


# ── Plot 2: BPE backends — log scale to show Pynini advantage ──

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(lengths, standard, 'o-', label='Standard (Python)', color=C_STANDARD, linewidth=2)
ax.plot(lengths, rust,     's-', label='Rust',              color=C_RUST,     linewidth=2)
ax.plot(lengths, pynini,   '^-', label='Pynini',            color=C_PYNINI,   linewidth=2)
ax.set_yscale('log')
ax.set_xlabel('Target prefix length')
ax.set_ylabel('Time (ms, log scale)')
ax.set_title('BPE Decomposition: Backend Comparison (log scale)')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3, which='both')
ax.set_ylim(0.05, 15)
save(fig, 'bpe_backends_log.png')


# ── Plot 3: PTB decomposition — Standard vs Rust ──

methods = ['Standard', 'Rust']
times = [1651, 110]  # ms (approx geomean from benchmark)
colors3 = [C_STANDARD, C_RUST]

fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.bar(methods, times, color=colors3, width=0.5)
ax.set_ylabel('Time (ms)')
ax.set_title('PTB Decomposition: Backend Comparison')
ax.set_yscale('log')
ax.set_ylim(1, 5000)
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.3,
            f'{val} ms', ha='center', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3, which='both')
# speedup annotation
ax.text(1, 30, '15x', ha='center', fontsize=10, fontweight='bold', color=C_RUST)
fig.tight_layout()
save(fig, 'ptb_backends.png')


# ── Plot 4: TransducedLM variants on PTB (updated numbers) ──

variants = ['TransducedLM\n(Rust peekaboo)', 'FusedTransducedLM\n(single-pass)']
ms_per_step = [129, 66]
total_s = [5.80, 2.95]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
colors_lm = [C_STANDARD, C_FUSED]

# ms/step
ax = axes[0]
bars = ax.bar(range(2), ms_per_step, color=colors_lm, width=0.5)
for i, v in enumerate(ms_per_step):
    ax.text(i, v + 3, f'{v} ms', ha='center', fontsize=9)
ax.set_xticks(range(2))
ax.set_xticklabels(variants, fontsize=7.5)
ax.set_ylabel('ms / step')
ax.set_title('PTB TransducedLM: Per-Step Latency')
ax.set_ylim(0, 170)
ax.grid(axis='y', alpha=0.3)
ax.annotate('2.0x', xy=(1, ms_per_step[1]), xytext=(0.5, 120),
            fontsize=11, fontweight='bold', color=C_FUSED,
            arrowprops=dict(arrowstyle='->', color=C_FUSED, lw=1.5))

# total time
ax = axes[1]
bars = ax.bar(range(2), total_s, color=colors_lm, width=0.5)
for i, v in enumerate(total_s):
    ax.text(i, v + 0.15, f'{v} s', ha='center', fontsize=9)
ax.set_xticks(range(2))
ax.set_xticklabels(variants, fontsize=7.5)
ax.set_ylabel('Total time (s)')
ax.set_title('PTB TransducedLM: Total Time')
ax.set_ylim(0, 7)
ax.grid(axis='y', alpha=0.3)

fig.suptitle('PTB TransducedLM (K=20, 3-gram CharNgramLM)', fontsize=10, fontweight='bold')
fig.tight_layout()
save(fig, 'ptb_transduced_lm.png')


# ── Plot 5: BPE TransducedLM variants ──

bpe_variants = ['TransducedLM', 'FusedTransducedLM', 'PyniniTransducedLM']
bpe_ms = [0.9, 0.7, 15]
bpe_colors = [C_STANDARD, C_FUSED, C_PYNINI]

fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(range(3), bpe_ms, color=bpe_colors, width=0.55)
for i, v in enumerate(bpe_ms):
    label = f'{v} ms' if v >= 1 else f'{v:.1f} ms'
    ax.text(i, v * 1.1 + 2, label, ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(3))
ax.set_xticklabels(bpe_variants, fontsize=8)
ax.set_ylabel('Avg ms / step')
ax.set_title('BPE TransducedLM: Per-Step Latency')
ax.set_yscale('log')
ax.set_ylim(0.3, 300)
ax.grid(axis='y', alpha=0.3, which='both')
fig.tight_layout()
save(fig, 'bpe_transduced_lm.png')


# ── Plot 6: Speedup summary (updated) ──

fig, ax = plt.subplots(figsize=(6, 3.5))
labels = [
    'Pynini decomp\n(BPE)',
    'Rust decomp\n(BPE)',
    'Rust decomp\n(PTB)',
    'FusedLM vs TransducedLM\n(PTB)',
]
speedups = [41.1, 4.7, 12.5, 2.0]
colors_sp = [C_PYNINI, C_RUST, C_RUST, C_FUSED]

bars = ax.barh(range(len(labels)), speedups, color=colors_sp, height=0.5)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Speedup (x)')
ax.set_xscale('log')
ax.set_xlim(1, 100)
ax.set_title('Speedup Summary vs Standard Python Baseline')
for bar, val in zip(bars, speedups):
    ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
            f'{val}x', va='center', fontsize=10, fontweight='bold')
ax.axvline(1, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.grid(axis='x', alpha=0.3, which='both')
fig.tight_layout()
save(fig, 'speedup_summary.png')


# ── Plot 7: logp agreement (before/after fix) ──

fig, ax = plt.subplots(figsize=(5, 3))
labels = ['Before fix', 'After fix']
diffs = [2.03, 0.000287]
colors_fix = ['#D9534F', '#5CB85C']
bars = ax.bar(labels, diffs, color=colors_fix, width=0.4)
ax.set_ylabel('Max |logp| diff')
ax.set_title('FusedTransducedLM vs TransducedLM (PTB)')
ax.set_yscale('log')
ax.set_ylim(0.0001, 5)
for bar, val in zip(bars, diffs):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5,
            f'{val}', ha='center', fontsize=10, fontweight='bold')
ax.axhline(0.01, color='gray', linestyle='--', alpha=0.5, label='~floating point')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3, which='both')
fig.tight_layout()
save(fig, 'logp_agreement.png')


# ── Plot 8: FST characteristics comparison ──

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
fsts = ['BPE', 'PTB']

# States
ax = axes[0]
vals = [140, 296]
bars = ax.bar(fsts, vals, color=[C_STANDARD, C_PYNINI], width=0.45)
ax.bar_label(bars, fontsize=10, padding=3)
ax.set_ylabel('Count')
ax.set_title('States')
ax.set_ylim(0, max(vals) * 1.3)
ax.grid(axis='y', alpha=0.3)

# Arcs
ax = axes[1]
vals = [180, 23723]
bars = ax.bar(fsts, vals, color=[C_STANDARD, C_PYNINI], width=0.45)
ax.bar_label(bars, fontsize=10, padding=3)
ax.set_ylabel('Count')
ax.set_title('Arcs')
ax.set_ylim(0, max(vals) * 1.3)
ax.grid(axis='y', alpha=0.3)

# Build time
ax = axes[2]
vals = [0.001, 36.4]
bars = ax.bar(fsts, vals, color=[C_STANDARD, C_PYNINI], width=0.45)
ax.bar_label(bars, fmt='%.3f s', fontsize=10, padding=3)
ax.set_ylabel('Seconds')
ax.set_title('Build Time')
ax.set_yscale('log')
ax.set_ylim(0.0005, 100)
ax.grid(axis='y', alpha=0.3, which='both')

fig.suptitle('FST Characteristics: BPE vs PTB', fontsize=12, fontweight='bold')
fig.tight_layout()
save(fig, 'fst_characteristics.png')


print('\nAll plots generated.')
