"""
SPM Regime Analysis: measure how often the decomposition is "locally SPM".

For each DFA state encountered during NonrecursiveDFADecomp's BFS, compute:
  - |S|: powerset size (number of NFA elements)
  - |positions(S)|: number of distinct target positions
  - |fst_states(S)|: number of distinct FST states
  - position-uniform: |positions(S)| == 1 (pure SPM -- single position)
  - position-factored: all FST states share the same position set
  - factoring ratio: |S| / (|positions(S)| * |fst_states(S)|)

Tests on example FSTs from transduction/examples.py with exhaustive short strings,
and optionally on BPE and PTB FSTs with WikiText data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transduction.util import set_memory_limit
set_memory_limit(4)

from collections import defaultdict, deque
from transduction import EPSILON
from transduction.fsa import FSA
from transduction.precover_nfa import PrecoverNFA
from transduction.universality import UniversalityFilter
from transduction import examples


def analyze_dfa_states(fst, target):
    """Run NonrecursiveDFADecomp-style BFS and collect SPM metrics per DFA state.

    Returns a list of dicts, one per visited DFA state.
    """
    target = tuple(target)
    oov = set(target) - (fst.B - {EPSILON})
    if oov:
        return []

    dfa = PrecoverNFA(fst, target).det()

    metrics = []
    worklist = deque()
    visited = set()

    for i in dfa.start():
        worklist.append(i)
        visited.add(i)

    while worklist:
        i = worklist.popleft()

        # i is a frozenset of (fst_state, buffer) NFA states
        nfa_states = i
        S = len(nfa_states)
        positions = set()
        fst_states = set()
        # Build position -> set of fst_states mapping (for factorability check)
        pos_to_fst = defaultdict(set)
        fst_to_pos = defaultdict(set)

        for (q, buf) in nfa_states:
            pos = len(buf)
            positions.add(pos)
            fst_states.add(q)
            pos_to_fst[pos].add(q)
            fst_to_pos[q].add(pos)

        n_positions = len(positions)
        n_fst_states = len(fst_states)
        position_uniform = (n_positions == 1)

        # Position-factored: each fst_state appears with the same set of positions
        # (equivalently, the Cartesian product positions x fst_states = S)
        product_size = n_positions * n_fst_states
        factoring_ratio = S / product_size if product_size > 0 else 0.0
        position_factored = (factoring_ratio == 1.0)

        metrics.append({
            'S': S,
            'n_positions': n_positions,
            'n_fst_states': n_fst_states,
            'position_uniform': position_uniform,
            'position_factored': position_factored,
            'factoring_ratio': factoring_ratio,
            'is_final': dfa.is_final(i),
        })

        for a, j in dfa.arcs(i):
            if j not in visited:
                worklist.append(j)
                visited.add(j)

    return metrics


def generate_targets(fst, max_len=4):
    """Generate all target strings up to max_len from the FST's output alphabet."""
    target_alpha = sorted(fst.B - {EPSILON})
    targets = [()]
    for length in range(1, max_len + 1):
        new_targets = []
        for t in targets:
            if len(t) == length - 1:
                for sym in target_alpha:
                    new_targets.append(t + (sym,))
        targets.extend(new_targets)
    return targets


def summarize_metrics(all_metrics, name):
    """Print summary statistics for a collection of DFA state metrics."""
    if not all_metrics:
        print(f"\n{'='*60}")
        print(f"  {name}: NO DATA (no reachable DFA states)")
        print(f"{'='*60}")
        return

    total = len(all_metrics)
    n_uniform = sum(1 for m in all_metrics if m['position_uniform'])
    n_factored = sum(1 for m in all_metrics if m['position_factored'])

    avg_S = sum(m['S'] for m in all_metrics) / total
    max_S = max(m['S'] for m in all_metrics)
    avg_positions = sum(m['n_positions'] for m in all_metrics) / total
    max_positions = max(m['n_positions'] for m in all_metrics)
    avg_fst_states = sum(m['n_fst_states'] for m in all_metrics) / total
    max_fst_states = max(m['n_fst_states'] for m in all_metrics)
    avg_ratio = sum(m['factoring_ratio'] for m in all_metrics) / total

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total DFA states:        {total}")
    print(f"  Position-uniform (SPM):  {n_uniform} ({100*n_uniform/total:.1f}%)")
    print(f"  Position-factored:       {n_factored} ({100*n_factored/total:.1f}%)")
    print(f"  Avg |S|:                 {avg_S:.2f}  (max {max_S})")
    print(f"  Avg |positions|:         {avg_positions:.2f}  (max {max_positions})")
    print(f"  Avg |fst_states|:        {avg_fst_states:.2f}  (max {max_fst_states})")
    print(f"  Avg factoring ratio:     {avg_ratio:.4f}")

    # Histogram of factoring ratios
    bins = [0, 0.25, 0.5, 0.75, 1.0, 1.01]
    bin_labels = ['[0, 0.25)', '[0.25, 0.5)', '[0.5, 0.75)', '[0.75, 1.0)', '1.0']
    counts = [0] * 5
    for m in all_metrics:
        r = m['factoring_ratio']
        if r < 0.25:
            counts[0] += 1
        elif r < 0.5:
            counts[1] += 1
        elif r < 0.75:
            counts[2] += 1
        elif r < 1.0:
            counts[3] += 1
        else:
            counts[4] += 1

    print(f"\n  Factoring ratio distribution:")
    for label, count in zip(bin_labels, counts):
        bar = '#' * min(count, 50)
        print(f"    {label:>15s}: {count:>5d}  {bar}")


def run_example_fst(name, fst, max_target_len=4):
    """Analyze an example FST over exhaustive short target strings."""
    all_metrics = []
    targets = generate_targets(fst, max_target_len)
    for target in targets:
        if not target:
            continue
        metrics = analyze_dfa_states(fst, target)
        all_metrics.extend(metrics)
    summarize_metrics(all_metrics, name)
    return all_metrics


def run_fst_on_specific_targets(name, fst, targets):
    """Analyze an FST on specific target strings."""
    all_metrics = []
    for target in targets:
        metrics = analyze_dfa_states(fst, target)
        all_metrics.extend(metrics)
    summarize_metrics(all_metrics, name)
    return all_metrics


def main():
    print("SPM Regime Analysis")
    print("=" * 60)

    results = {}

    # --- Example FSTs with exhaustive short strings ---

    # newspeak2: replacement FST (bad -> ungood)
    fst = examples.newspeak2()
    results['newspeak2'] = run_example_fst('newspeak2', fst, max_target_len=3)

    # samuel_example: non-trivial FST with epsilon arcs
    fst = examples.samuel_example()
    results['samuel_example'] = run_example_fst('samuel_example', fst, max_target_len=5)

    # number_comma_separator
    Domain = frozenset(set('0123456789, abcdefghijklmnopqrstuvwxyz'))
    fst = examples.number_comma_separator(Domain)
    # Use a subset of targets to keep it manageable
    target_alpha = sorted(fst.B - {EPSILON})
    # Pick a small subset of target symbols
    small_alpha = [s for s in target_alpha if s in set('0123,| ')]
    small_targets = [()]
    for length in range(1, 4):
        new_targets = []
        for t in small_targets:
            if len(t) == length - 1:
                for sym in small_alpha:
                    new_targets.append(t + (sym,))
        small_targets.extend(new_targets)
    small_targets = [t for t in small_targets if t]
    results['number_comma_separator'] = run_fst_on_specific_targets(
        'number_comma_separator', fst, small_targets
    )

    # doom(V={'a','b'}, K=3) - the triplets of doom
    fst = examples.doom({'a', 'b'}, 3)
    results['doom_3'] = run_example_fst('doom(K=3)', fst, max_target_len=5)

    # delete_b: infinite quotients
    fst = examples.delete_b()
    results['delete_b'] = run_example_fst('delete_b', fst, max_target_len=5)

    # mystery1
    fst = examples.mystery1()
    results['mystery1'] = run_example_fst('mystery1', fst, max_target_len=4)

    # mystery7
    fst = examples.mystery7()
    results['mystery7'] = run_example_fst('mystery7', fst, max_target_len=4)

    # replace (identity-like, should be fully SPM)
    fst = examples.replace([('a', 'x'), ('b', 'y'), ('c', 'z')])
    results['replace_xyz'] = run_example_fst('replace(a->x, b->y, c->z)', fst, max_target_len=5)

    # anbn
    fst = examples.anbn()
    results['anbn'] = run_example_fst('anbn', fst, max_target_len=5)

    # backticks_to_quote
    fst = examples.backticks_to_quote()
    results['backticks_to_quote'] = run_example_fst('backticks_to_quote', fst, max_target_len=4)

    # parity_copy
    fst = examples.parity_copy()
    results['parity_copy'] = run_example_fst('parity_copy', fst, max_target_len=4)

    # --- Summary table ---
    print("\n\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print(f"{'FST':<30s} {'States':>7s} {'Uniform%':>9s} {'Factored%':>10s} {'Avg Ratio':>10s} {'Max |S|':>8s}")
    print("-" * 80)
    for name, metrics in results.items():
        if not metrics:
            print(f"{name:<30s}  (no data)")
            continue
        total = len(metrics)
        n_uniform = sum(1 for m in metrics if m['position_uniform'])
        n_factored = sum(1 for m in metrics if m['position_factored'])
        avg_ratio = sum(m['factoring_ratio'] for m in metrics) / total
        max_S = max(m['S'] for m in metrics)
        print(f"{name:<30s} {total:>7d} {100*n_uniform/total:>8.1f}% {100*n_factored/total:>9.1f}% {avg_ratio:>10.4f} {max_S:>8d}")


    # --- Optional: BPE analysis ---
    try:
        from transformers import AutoTokenizer
        from transduction.applications.bpe import bpe_wfst

        print("\n\nBPE (GPT-2) Analysis")
        print("=" * 60)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        fst = bpe_wfst(tokenizer)

        # Use a few short byte-level target strings
        bpe_targets = [
            tuple(b'hello'),
            tuple(b'the'),
            tuple(b' the'),
            tuple(b'world'),
            tuple(b' a'),
        ]
        results['bpe_gpt2'] = run_fst_on_specific_targets('BPE (GPT-2)', fst, bpe_targets)
    except ImportError:
        print("\n(Skipping BPE analysis: transformers not available)")

    # --- Optional: PTB analysis ---
    try:
        from transduction.applications.ptb import build_ptb_fst_pynini

        print("\n\nPTB Analysis")
        print("=" * 60)
        fst = build_ptb_fst_pynini()

        # Use a few short byte-level target strings
        ptb_targets = [
            tuple(b'hello'),
            tuple(b'the'),
            tuple(b"it's"),
        ]
        results['ptb'] = run_fst_on_specific_targets('PTB', fst, ptb_targets)
    except ImportError:
        print("\n(Skipping PTB analysis: pynini not available)")

    print("\n\nDone.")


if __name__ == '__main__':
    main()
