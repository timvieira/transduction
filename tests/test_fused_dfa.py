"""
Direct equivalence tests for the Rust lazy DFA used by FusedTransducedLM.

Compares the Rust RustFusedHelper's arcs() and classify() against the Python
reference (PeekabooPrecover + LazyDeterminize + FstUniversality) at every
reachable DFA state, not just through the beam-search output.
"""

import pytest
from collections import deque

from transduction import examples, FST
from transduction.fst import EPSILON
from transduction.precover_nfa import PeekabooLookaheadNFA as PeekabooPrecover
from transduction.peekaboo_incremental import FstUniversality, TruncatedDFA
from transduction.rust_bridge import to_rust_fst

import transduction_core


# ---------------------------------------------------------------------------
# Python reference: classify a frozenset DFA state
# ---------------------------------------------------------------------------

def _ref_meta(dfa_state, target, fst):
    """Extract metadata from a Python DFA state (frozenset of NFA triples)."""
    N = len(target)
    relevant = set()
    final_syms = set()
    is_preimage = False
    has_truncated = False
    trunc_output_syms = set()
    for fst_state, output, truncated in dfa_state:
        if len(output) == N and fst.is_final(fst_state):
            is_preimage = True
        if len(output) > N:
            sym = output[N]
            relevant.add(sym)
            if output[:N] == target and fst.is_final(fst_state):
                final_syms.add(sym)
        if truncated:
            has_truncated = True
            if len(output) > N:
                trunc_output_syms.add(output[N])
    return dict(
        relevant=relevant,
        final_syms=final_syms,
        is_preimage=is_preimage,
        has_truncated=has_truncated,
        trunc_output_syms=trunc_output_syms,
    )


def _ref_classify(dfa_state, target, fst, univ, raw_dfa, source_alphabet):
    """Classify a Python DFA state: find quotient/remainder symbols."""
    meta = _ref_meta(dfa_state, target, fst)
    all_input_universal = univ.all_input_universal

    quotient_sym = None
    remainder_syms = set()

    # Build universality filters lazily per symbol
    for y in sorted(meta['relevant']):
        if quotient_sym is not None:
            # Peekaboo: at most one quotient
            if y in meta['final_syms']:
                remainder_syms.add(y)
            continue

        if all_input_universal:
            is_univ = y in meta['final_syms']
        else:
            trunc_dfa = TruncatedDFA(dfa=raw_dfa, fst=fst, target=target + (y,))
            uf = univ.make_filter(fst, target + (y,), trunc_dfa, source_alphabet)
            is_univ = uf.is_universal(dfa_state)

        if is_univ:
            quotient_sym = y
        elif y in meta['final_syms']:
            remainder_syms.add(y)

    return dict(
        quotient_sym=quotient_sym,
        remainder_syms=remainder_syms,
        is_preimage=meta['is_preimage'],
        has_truncated=meta['has_truncated'],
        trunc_output_syms=meta['trunc_output_syms'],
    )


# ---------------------------------------------------------------------------
# Lockstep BFS comparison
# ---------------------------------------------------------------------------

def compare_dfa(fst, target, max_states=500):
    """BFS both DFAs in lockstep, comparing arcs and classification.

    Returns the number of states visited.
    """
    target = tuple(target)

    # --- Python reference DFA ---
    nfa = PeekabooPrecover(fst, target)
    raw_dfa = nfa.det()
    py_dfa = raw_dfa.cache()

    univ = FstUniversality(fst)
    source_alphabet = fst.A - {EPSILON}

    py_starts = list(py_dfa.start())
    assert len(py_starts) == 1, f"Expected 1 start state, got {len(py_starts)}"
    py_start = py_starts[0]

    # --- Rust lazy DFA ---
    rust_fst, sym_map, _ = to_rust_fst(fst)
    helper = transduction_core.RustFusedHelper(rust_fst)
    target_u32 = [sym_map(y) for y in target]
    helper.new_step(target_u32)
    rust_starts = helper.start_ids()
    assert len(rust_starts) == 1, f"Expected 1 Rust start state, got {len(rust_starts)}"
    rust_start = rust_starts[0]

    inv_sym = {v: k for k, v in sym_map.items()}

    # BFS: queue of (py_dfa_state, rust_sid) pairs
    visited_py = {py_start}
    visited_rust = {rust_start}
    queue = deque([(py_start, rust_start)])
    n_visited = 0

    while queue:
        if n_visited >= max_states:
            break
        py_state, rust_sid = queue.popleft()
        n_visited += 1

        # --- Compare classification ---
        py_cls = _ref_classify(py_state, target, fst, univ, raw_dfa, source_alphabet)
        rust_cls_raw = helper.classify(rust_sid)

        rust_q = inv_sym[rust_cls_raw.quotient_sym] if rust_cls_raw.quotient_sym is not None else None
        rust_r = set(inv_sym[s] for s in rust_cls_raw.remainder_syms)
        rust_trunc = set(inv_sym[s] for s in rust_cls_raw.trunc_output_syms)

        assert py_cls['quotient_sym'] == rust_q, (
            f"Quotient mismatch at state {rust_sid}: "
            f"python={py_cls['quotient_sym']!r}, rust={rust_q!r}"
        )
        assert py_cls['remainder_syms'] == rust_r, (
            f"Remainder mismatch at state {rust_sid}: "
            f"python={py_cls['remainder_syms']!r}, rust={rust_r!r}"
        )
        assert py_cls['is_preimage'] == rust_cls_raw.is_preimage, (
            f"is_preimage mismatch at state {rust_sid}: "
            f"python={py_cls['is_preimage']}, rust={rust_cls_raw.is_preimage}"
        )
        assert py_cls['has_truncated'] == rust_cls_raw.has_truncated, (
            f"has_truncated mismatch at state {rust_sid}: "
            f"python={py_cls['has_truncated']}, rust={rust_cls_raw.has_truncated}"
        )
        assert py_cls['trunc_output_syms'] == rust_trunc, (
            f"trunc_output_syms mismatch at state {rust_sid}: "
            f"python={py_cls['trunc_output_syms']!r}, rust={rust_trunc!r}"
        )

        # --- Compare arcs ---
        # If quotient, the Rust DFA shouldn't need arcs (search doesn't expand),
        # but for completeness, compare arcs for non-quotient states only.
        if py_cls['quotient_sym'] is not None:
            continue

        py_arcs_raw = list(py_dfa.arcs(py_state))
        rust_arcs_raw = helper.arcs(rust_sid)

        # Build arc dict: source_symbol -> dest_state
        py_arc_dict = {}
        for x, dest in py_arcs_raw:
            assert x not in py_arc_dict, f"Duplicate arc for symbol {x!r}"
            py_arc_dict[x] = dest

        rust_arc_dict = {}
        for x_u32, dest_sid in rust_arcs_raw:
            x = inv_sym[x_u32]
            assert x not in rust_arc_dict, f"Duplicate Rust arc for symbol {x!r}"
            rust_arc_dict[x] = dest_sid

        py_symbols = set(py_arc_dict.keys())
        rust_symbols = set(rust_arc_dict.keys())
        assert py_symbols == rust_symbols, (
            f"Arc symbol mismatch at state {rust_sid}: "
            f"python_only={py_symbols - rust_symbols}, "
            f"rust_only={rust_symbols - py_symbols}"
        )

        # Push matched successors for lockstep BFS
        for x in sorted(py_symbols):
            py_dest = py_arc_dict[x]
            rust_dest = rust_arc_dict[x]
            if py_dest not in visited_py:
                visited_py.add(py_dest)
                visited_rust.add(rust_dest)
                queue.append((py_dest, rust_dest))

    return n_visited


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestFusedDFAEquivalence:
    """Compare Rust lazy DFA against Python reference at the automaton level."""

    def test_copy_fst_empty_target(self):
        fst = FST()
        fst.add_start(0)
        fst.add_stop(0)
        for x in ['a', 'b']:
            fst.add_arc(0, x, x, 0)
        n = compare_dfa(fst, ())
        assert n >= 1

    def test_copy_fst_after_one_symbol(self):
        fst = FST()
        fst.add_start(0)
        fst.add_stop(0)
        for x in ['a', 'b']:
            fst.add_arc(0, x, x, 0)
        n = compare_dfa(fst, ('a',))
        assert n >= 1

    def test_copy_fst_after_two_symbols(self):
        fst = FST()
        fst.add_start(0)
        fst.add_stop(0)
        for x in ['a', 'b']:
            fst.add_arc(0, x, x, 0)
        n = compare_dfa(fst, ('a', 'b'))
        assert n >= 1

    def test_small(self):
        n = compare_dfa(examples.small(), ())
        assert n >= 1

    def test_small_after_advance(self):
        fst = examples.small()
        target_alpha = fst.B - {EPSILON}
        for y in sorted(target_alpha):
            n = compare_dfa(fst, (y,))
            assert n >= 1

    def test_lowercase(self):
        n = compare_dfa(examples.lowercase(), ())
        assert n >= 1

    def test_lowercase_after_advance(self):
        fst = examples.lowercase()
        for y in ['h', 'e']:
            n = compare_dfa(fst, (y,))
            assert n >= 1

    def test_delete_b(self):
        n = compare_dfa(examples.delete_b(), ())
        assert n >= 1

    def test_delete_b_after_advance(self):
        n = compare_dfa(examples.delete_b(), ('a',))
        assert n >= 1

    def test_samuel_example(self):
        n = compare_dfa(examples.samuel_example(), ())
        assert n >= 1

    def test_samuel_after_advance(self):
        fst = examples.samuel_example()
        for y in ['b', 'a']:
            n = compare_dfa(fst, (y,))
            assert n >= 1

    def test_duplicate(self):
        fst = examples.duplicate(['a', 'b'], K=2)
        n = compare_dfa(fst, ())
        assert n >= 1

    def test_duplicate_after_advance(self):
        fst = examples.duplicate(['a', 'b'], K=2)
        n = compare_dfa(fst, ('a', 'a'))
        assert n >= 1

    def test_togglecase(self):
        n = compare_dfa(examples.togglecase(), ())
        assert n >= 1

    def test_togglecase_after_advance(self):
        fst = examples.togglecase()
        n = compare_dfa(fst, ('A',))
        assert n >= 1

    def test_infinite_quotient(self):
        """FST with infinite quotient language — tests truncation handling."""
        fst = examples.infinite_quotient(alphabet=('a',), separators=('#',))
        n = compare_dfa(fst, ())
        assert n >= 1

    def test_infinite_quotient_after_advance(self):
        fst = examples.infinite_quotient(alphabet=('a',), separators=('#',))
        n = compare_dfa(fst, ('#',))
        assert n >= 1

    def test_backticks_to_quote(self):
        fst = examples.backticks_to_quote()
        n = compare_dfa(fst, ())
        assert n >= 1

    def test_backticks_to_quote_after_advance(self):
        fst = examples.backticks_to_quote()
        n = compare_dfa(fst, ('\u2018',))
        assert n >= 1

    def test_parity_copy(self):
        fst = examples.parity_copy()
        n = compare_dfa(fst, ())
        assert n >= 1

    def test_parity_copy_after_advance(self):
        fst = examples.parity_copy()
        n = compare_dfa(fst, ('a',))
        assert n >= 1

    def test_multi_step_sequence(self):
        """Test multiple target symbols in sequence using the same helper."""
        fst = FST()
        fst.add_start(0)
        fst.add_stop(0)
        for x in ['a', 'b']:
            fst.add_arc(0, x, x, 0)

        for depth in range(4):
            target = ('a',) * depth
            n = compare_dfa(fst, target)
            assert n >= 1

    def test_carry_forward_arena_persistence(self):
        """Verify that arena IDs from one step survive into the next.

        This is the key property for carry-forward: a state ID obtained
        during step N must be queryable (arcs/classify) at step N+1.
        """
        fst = FST()
        fst.add_start(0)
        fst.add_stop(0)
        for x in ['a', 'b']:
            fst.add_arc(0, x, x, 0)

        rust_fst, sym_map, _ = to_rust_fst(fst)
        helper = transduction_core.RustFusedHelper(rust_fst)
        inv_sym = {v: k for k, v in sym_map.items()}

        # Step 0: empty target
        helper.new_step([])
        starts_0 = helper.start_ids()
        arcs_0 = helper.arcs(starts_0[0])
        # Collect some state IDs from step 0
        dest_ids_0 = [dst for _, dst in arcs_0]
        assert len(dest_ids_0) > 0

        # Step 1: target = ('a',) — old IDs should still be queryable
        a_u32 = sym_map('a')
        helper.new_step([a_u32])
        starts_1 = helper.start_ids()

        # The old dest_ids_0 should be queryable (arcs + classify)
        for sid in dest_ids_0:
            cls = helper.classify(sid)
            arcs = helper.arcs(sid)
            # No crash = success; classification and arcs may differ
            # (different target), but the IDs must be valid
            assert isinstance(cls.is_preimage, bool)
            assert isinstance(arcs, list)
