"""
Token-decomposability analysis: measure when DFA transitions depend only on
position sets, not on FST states.

A decomposition is TokenDecomposable if:
  1. Position-deterministic transitions: for all DFA states S1, S2 with
     positions(S1) = positions(S2) and source symbol a,
     positions(succ(S1, a)) = positions(succ(S2, a))
  2. Position-deterministic finality: for all S1, S2 with
     positions(S1) = positions(S2),
     is_final(S1) = is_final(S2)

When both hold, positions alone determine DFA behavior and we can quotient
away the FST-state dimension (as TokenDecompose does for BPE).
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


def positions_of(state):
    """Extract position set from a DFA state (frozenset of (q, buf) pairs)."""
    return frozenset(len(buf) for (q, buf) in state)


def analyze_token_decomposability(fst, target, timeout=30):
    """Check whether DFA transitions are determined by position sets alone.

    Returns dict with:
      - n_states: total reachable DFA states
      - n_position_sets: number of distinct position sets
      - n_consistent_groups: position-set groups where all members agree on
        successor position sets AND finality
      - n_states_in_consistent: states in consistent groups
      - n_finality_consistent: groups where all members agree on finality
      - transition_violations: number of (pos_set, symbol) pairs where
        different DFA states disagree on successor positions
    """
    target = tuple(target)
    oov = set(target) - (fst.B - {EPSILON})
    if oov:
        return None

    signal.alarm(timeout)
    try:
        dfa = PrecoverNFA(fst, target).det()

        # BFS: collect all states and their transitions
        state_info = {}  # state -> {pos_set, is_final, succ: {symbol: succ_pos_set}}
        visited = set()
        worklist = deque()
        for i in dfa.start():
            worklist.append(i)
            visited.add(i)
        while worklist:
            i = worklist.popleft()
            pos_set = positions_of(i)
            is_final = dfa.is_final(i)
            succs = {}
            for a, j in dfa.arcs(i):
                succs[a] = positions_of(j)
                if j not in visited:
                    worklist.append(j)
                    visited.add(j)
            state_info[id(i)] = {
                'pos_set': pos_set,
                'is_final': is_final,
                'succs': succs,
                'state': i,
            }

        # Group by position set
        by_pos_set = defaultdict(list)
        for sid, info in state_info.items():
            by_pos_set[info['pos_set']].append(info)

        n_states = len(state_info)
        n_position_sets = len(by_pos_set)
        n_consistent_groups = 0
        n_states_in_consistent = 0
        n_finality_consistent = 0
        transition_violations = 0

        for pos_set, group in by_pos_set.items():
            # Check finality consistency
            finalities = {info['is_final'] for info in group}
            finality_ok = len(finalities) == 1
            if finality_ok:
                n_finality_consistent += 1

            # Check transition consistency
            # Collect all symbols seen across the group
            all_symbols = set()
            for info in group:
                all_symbols.update(info['succs'].keys())

            transitions_ok = True
            for sym in all_symbols:
                succ_pos_sets = set()
                for info in group:
                    if sym in info['succs']:
                        succ_pos_sets.add(info['succs'][sym])
                    else:
                        succ_pos_sets.add(None)  # no transition on this symbol
                if len(succ_pos_sets) > 1:
                    transitions_ok = False
                    transition_violations += 1

            if finality_ok and transitions_ok:
                n_consistent_groups += 1
                n_states_in_consistent += len(group)

        signal.alarm(0)
        return {
            'n_states': n_states,
            'n_position_sets': n_position_sets,
            'n_consistent_groups': n_consistent_groups,
            'n_states_in_consistent': n_states_in_consistent,
            'n_finality_consistent': n_finality_consistent,
            'transition_violations': transition_violations,
            'n_total_groups': len(by_pos_set),
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
    """Run token-decomposability analysis on an FST."""
    aiu = check_all_input_universal(fst)

    total_states = 0
    total_pos_sets = 0
    total_consistent = 0
    total_in_consistent = 0
    total_finality_consistent = 0
    total_violations = 0
    total_groups = 0
    n_done = 0
    n_timeout = 0

    for target in targets:
        result = analyze_token_decomposability(fst, target, timeout=timeout_per_target)
        if result is None:
            n_timeout += 1
            continue
        n_done += 1
        total_states += result['n_states']
        total_pos_sets += result['n_position_sets']
        total_consistent += result['n_consistent_groups']
        total_in_consistent += result['n_states_in_consistent']
        total_finality_consistent += result['n_finality_consistent']
        total_violations += result['transition_violations']
        total_groups += result['n_total_groups']

    if total_states == 0:
        print(f"  {name:<25s}  (no data)", flush=True)
        return

    pct_consistent_states = 100 * total_in_consistent / total_states
    pct_consistent_groups = 100 * total_consistent / total_groups if total_groups > 0 else 0
    compression = total_states / total_pos_sets if total_pos_sets > 0 else 1.0

    timeout_note = f"  ({n_timeout}T)" if n_timeout > 0 else ""

    print(f"  {name:<25s}  aiu={str(aiu):<6s}"
          f"  states={total_states:>7d}"
          f"  pos_sets={total_pos_sets:>7d}"
          f"  compress={compression:>5.1f}x"
          f"  td_states={pct_consistent_states:>5.1f}%"
          f"  td_groups={pct_consistent_groups:>5.1f}%"
          f"  violations={total_violations:>4d}"
          f"{timeout_note}", flush=True)


def build_subsampled_bpe(vocab_size, _cache={}):
    """Build a subsampled BPE FST (cached)."""
    if vocab_size in _cache:
        return _cache[vocab_size]
    from transformers import AutoTokenizer
    from transduction.lm.statelm import decode_hf_tokenizer
    if '_tok' not in _cache:
        print("  Loading GPT-2 tokenizer...", end=" ", flush=True)
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
        _, _, _decode, _ = decode_hf_tokenizer(tokenizer)
        drop = {x.encode() for x in tokenizer.all_special_tokens}
        all_ids = sorted(i for i in range(len(_decode)) if _decode[i] not in drop)
        _cache['_tok'] = (_decode, drop, all_ids)
        print("done", flush=True)
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
    print("Token-Decomposability Analysis", flush=True)
    print("=" * 120, flush=True)
    print(flush=True)
    print("  TokenDecomposable (td) = DFA transitions + finality determined by position set alone", flush=True)
    print("  compress = states / position_sets (how much TokenDecompose would compress the DFA)", flush=True)
    print("  td_states = % of states in fully consistent position-set groups", flush=True)
    print("  td_groups = % of position-set groups that are fully consistent", flush=True)
    print("  violations = (pos_set, symbol) pairs where states disagree on successor positions", flush=True)
    print(flush=True)

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
    print(flush=True)
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
        print(flush=True)
        print("  Building PTB FST...", end=" ", flush=True)
        from transduction.applications.ptb import build_ptb_fst_pynini
        ptb_fst = build_ptb_fst_pynini()
        print("done", flush=True)
        output_alpha = ptb_fst.B - {EPSILON}
        ptb_targets = [tuple(text.encode('utf-8'))[:length]
                       for text in TARGET_TEXTS
                       for length in [3, 5, 8]]
        valid = [t for t in ptb_targets if set(t) <= output_alpha]
        run_fst("PTB", ptb_fst, valid)
    except Exception as e:
        print(f"\n  PTB: error — {e}", flush=True)

    print(flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
