"""
Saturation Analysis: measure when DFA states are "locally TokenDecomposable".

A DFA state S is *saturated* at position pos if:
    {q : (q, target[:pos]) in S} = Max(pos, target)

where Max(pos, target) = union over all reachable DFA states of FST states
at position pos. When S is saturated at every position it contains, it is
"locally TokenDecomposable" — the position set alone fully determines the state.

For all_input_universal FSTs (BPE), all DFA states should be 100% saturated.
For general FSTs, saturation may be partial.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transduction.util import set_memory_limit
set_memory_limit(8)

import signal
import time
from collections import defaultdict, deque
from transduction.fst import FST
from transduction.fsa import EPSILON
from transduction.precover_nfa import PrecoverNFA
from transduction import examples
from transduction.universality import check_all_input_universal


class Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise Timeout()

signal.signal(signal.SIGALRM, _alarm)


def analyze_saturation(fst, target, timeout=30):
    """Analyze saturation of all DFA states for a given target.

    Returns dict with metrics, or None on timeout.
    """
    target = tuple(target)
    oov = set(target) - (fst.B - {EPSILON})
    if oov:
        return None

    signal.alarm(timeout)
    try:
        dfa = PrecoverNFA(fst, target).det()

        # First pass: collect all DFA states and compute Max(pos, target)
        all_states = []
        max_sets = defaultdict(set)   # pos -> set of fst_states
        visited = set()
        worklist = deque()
        for i in dfa.start():
            worklist.append(i)
            visited.add(i)
        while worklist:
            i = worklist.popleft()
            all_states.append(i)
            for (q, buf) in i:
                max_sets[len(buf)].add(q)
            for a, j in dfa.arcs(i):
                if j not in visited:
                    worklist.append(j)
                    visited.add(j)

        max_sets_frozen = {pos: frozenset(qs) for pos, qs in max_sets.items()}

        # Second pass: check saturation of each DFA state
        n_total = len(all_states)
        n_fully_saturated = 0
        n_position_uniform = 0
        n_saturated_positions_total = 0
        n_positions_total = 0

        for state in all_states:
            pos_to_fst = defaultdict(set)
            for (q, buf) in state:
                pos_to_fst[len(buf)].add(q)

            n_pos = len(pos_to_fst)
            n_positions_total += n_pos
            n_sat = 0
            for pos, fst_states in pos_to_fst.items():
                if frozenset(fst_states) == max_sets_frozen.get(pos, frozenset()):
                    n_sat += 1
            n_saturated_positions_total += n_sat

            if n_sat == n_pos:
                n_fully_saturated += 1
            if n_pos == 1:
                n_position_uniform += 1

        signal.alarm(0)
        return {
            'n_states': n_total,
            'n_fully_saturated': n_fully_saturated,
            'n_position_uniform': n_position_uniform,
            'n_saturated_positions': n_saturated_positions_total,
            'n_positions': n_positions_total,
            'n_max_positions': len(max_sets_frozen),
            'max_set_sizes': {pos: len(qs) for pos, qs in sorted(max_sets_frozen.items())},
        }
    except Timeout:
        return None
    finally:
        signal.alarm(0)


def generate_targets(fst, max_len):
    target_alpha = sorted(fst.B - {EPSILON})
    targets = [()]
    for length in range(1, max_len + 1):
        new = []
        for t in targets:
            if len(t) == length - 1:
                for sym in target_alpha:
                    new.append(t + (sym,))
        targets.extend(new)
    return [t for t in targets if t]


def run_fst(name, fst, targets, timeout_per_target=30):
    """Run saturation analysis on an FST with given targets."""
    aiu = check_all_input_universal(fst)

    total_states = 0
    total_saturated = 0
    total_uniform = 0
    total_sat_positions = 0
    total_positions = 0
    n_targets_done = 0
    n_targets_timeout = 0

    for target in targets:
        result = analyze_saturation(fst, target, timeout=timeout_per_target)
        if result is None:
            n_targets_timeout += 1
            continue
        n_targets_done += 1
        total_states += result['n_states']
        total_saturated += result['n_fully_saturated']
        total_uniform += result['n_position_uniform']
        total_sat_positions += result['n_saturated_positions']
        total_positions += result['n_positions']

    if total_states == 0:
        print(f"  {name:<30s}  (no data)")
        return

    pct_sat = 100 * total_saturated / total_states
    pct_uni = 100 * total_uniform / total_states
    pct_sat_pos = 100 * total_sat_positions / total_positions if total_positions > 0 else 0

    timeout_note = f"  ({n_targets_timeout} timeouts)" if n_targets_timeout > 0 else ""

    print(f"  {name:<30s}  aiu={str(aiu):<6s}  "
          f"states={total_states:>8d}  "
          f"saturated={pct_sat:>5.1f}%  "
          f"uniform={pct_uni:>5.1f}%  "
          f"sat_pos={pct_sat_pos:>5.1f}%"
          f"{timeout_note}")


def build_subsampled_bpe(vocab_size, _cache={}):
    """Build a subsampled BPE FST (cached)."""
    if vocab_size in _cache:
        return _cache[vocab_size]
    from transformers import AutoTokenizer
    from transduction.lm.statelm import decode_hf_tokenizer
    if '_tok' not in _cache:
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
        _, _, _decode, _ = decode_hf_tokenizer(tokenizer)
        drop = {x.encode() for x in tokenizer.all_special_tokens}
        all_ids = sorted(i for i in range(len(_decode)) if _decode[i] not in drop)
        _cache['_tok'] = (_decode, drop, all_ids)
    _decode, drop, all_ids = _cache['_tok']
    used = all_ids[:vocab_size]
    m = FST()
    m.add_start(())
    for i in used:
        x = _decode[i]
        if x in drop:
            continue
        bx = tuple(x)
        for j in range(len(bx)):
            m.add_arc(bx[:j], EPSILON, bx[j], bx[:j+1])
        m.add_arc(bx, i, EPSILON, ())
    m.add_stop(())
    fst = m.renumber()
    _cache[vocab_size] = fst
    return fst


TARGET_TEXTS = [
    "The quick brown fox",
    "Hello world",
    "It was the best",
    "To be or not",
    "Call me Ishmael",
]


def main():
    print("Saturation Analysis: locally TokenDecomposable DFA states")
    print("=" * 100)
    print()
    print("  Definitions:")
    print("    saturated = FST state set at each position equals Max(pos, target)")
    print("    uniform   = all NFA elements share a single position")
    print("    sat_pos   = fraction of (state, position) pairs that are saturated")
    print()
    print(f"  {'FST':<30s}  {'aiu':<6s}  "
          f"{'states':>8s}  "
          f"{'saturated':>10s}  "
          f"{'uniform':>8s}  "
          f"{'sat_pos':>8s}")
    print(f"  {'-'*90}")

    # --- Example FSTs ---
    test_cases = [
        ('replace(xyz)', lambda: examples.replace([('a', 'x'), ('b', 'y'), ('c', 'z')]), 4),
        ('delete_b', examples.delete_b, 4),
        ('samuel_example', examples.samuel_example, 4),
        ('doom(K=3)', lambda: examples.doom({'a', 'b'}, 3), 4),
        ('mystery1', examples.mystery1, 4),
        ('mystery7', examples.mystery7, 4),
        ('newspeak2', examples.newspeak2, 3),
        ('anbn', examples.anbn, 4),
        ('backticks_to_quote', examples.backticks_to_quote, 3),
        ('parity_copy', examples.parity_copy, 3),
    ]

    for name, fst_fn, max_len in test_cases:
        fst = fst_fn()
        targets = generate_targets(fst, max_len)
        run_fst(name, fst, targets)

    # --- BPE ---
    print()
    bpe_byte_targets = [tuple(text.encode('utf-8'))[:length]
                        for text in TARGET_TEXTS
                        for length in [3, 5, 8]]

    for vocab_size in [100, 500]:
        fst = build_subsampled_bpe(vocab_size)
        output_alpha = fst.B - {EPSILON}
        valid = [t for t in bpe_byte_targets if set(t) <= output_alpha]
        run_fst(f"BPE (vocab={vocab_size})", fst, valid)

    # --- PTB ---
    try:
        print()
        from transduction.applications.ptb import build_ptb_fst_pynini
        ptb_fst = build_ptb_fst_pynini()
        output_alpha = ptb_fst.B - {EPSILON}
        ptb_targets = [tuple(text.encode('utf-8'))[:length]
                       for text in TARGET_TEXTS
                       for length in [3, 5, 8]]
        valid = [t for t in ptb_targets if set(t) <= output_alpha]
        run_fst("PTB", ptb_fst, valid)
    except Exception as e:
        print(f"  PTB: error — {e}")

    # --- Detailed breakdown for one example ---
    print()
    print()
    print("Detailed: Max set sizes by position")
    print("=" * 80)

    detail_cases = [
        ('replace(xyz)', examples.replace([('a', 'x'), ('b', 'y'), ('c', 'z')]), (120, 121, 122)),
        ('samuel_example', examples.samuel_example(), (97, 98)),
        ('newspeak2', examples.newspeak2(), tuple(b'bad')),
        ('anbn', examples.anbn(), (97, 98, 97)),
    ]

    for name, fst, target in detail_cases:
        if callable(fst):
            fst = fst()
        result = analyze_saturation(fst, target)
        if result:
            print(f"\n  {name}, target={target}")
            print(f"    states={result['n_states']}  "
                  f"saturated={result['n_fully_saturated']}  "
                  f"uniform={result['n_position_uniform']}")
            print(f"    Max set sizes by position:")
            for pos, size in sorted(result['max_set_sizes'].items()):
                print(f"      pos={pos}: |Max|={size}")

    # BPE detail
    fst = build_subsampled_bpe(100)
    output_alpha = fst.B - {EPSILON}
    target = tuple(b'The')
    if set(target) <= output_alpha:
        result = analyze_saturation(fst, target)
        if result:
            print(f"\n  BPE(100), target={target}")
            print(f"    states={result['n_states']}  "
                  f"saturated={result['n_fully_saturated']}  "
                  f"uniform={result['n_position_uniform']}")
            print(f"    Max set sizes by position:")
            for pos, size in sorted(result['max_set_sizes'].items()):
                print(f"      pos={pos}: |Max|={size}")

    print("\nDone.")


if __name__ == '__main__':
    main()
