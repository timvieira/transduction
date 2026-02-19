"""
Tests for rho-arc precover DFA: verify that SymbolicLazyDeterminize (with
DFA-level rho factoring) + ExpandRho produces a DFA equal to the standard
LazyDeterminize(PrecoverNFA).

For each (FST, target), we materialize both DFAs and check:
  L(standard) = L(rho-factored)

Uses the same FSTs and targets as test_general.py.
"""

from itertools import product

import pytest

from transduction import examples, EPSILON
from transduction.fst import FST
from transduction.lazy import LazyDeterminize
from transduction.precover_nfa import PrecoverNFA
from transduction.symbolic_precover import (
    RHO, SymbolicLazyDeterminize, ExpandRho,
)

try:
    from transduction.rust_bridge import RustRhoDeterminize
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def standard_precover_dfa(fst, target):
    """Materialize the standard precover DFA."""
    return LazyDeterminize(PrecoverNFA(fst, target)).materialize()


def rho_precover_dfa(fst, target):
    """Materialize the rho-factored precover DFA with RHO arcs expanded."""
    alphabet = fst.A - {EPSILON}
    sym_dfa = SymbolicLazyDeterminize(PrecoverNFA(fst, target), alphabet)
    return ExpandRho(sym_dfa, alphabet).materialize()


def target_prefixes(fst, max_len=3):
    """Generate target prefixes up to max_len from the FST's output alphabet."""
    target_alpha = sorted(fst.B - {EPSILON})
    targets = [()]
    for length in range(1, max_len + 1):
        for t in product(target_alpha, repeat=length):
            targets.append(t)
    return targets


def check_equality(fst, target):
    """Verify L(standard) = L(rho-factored) via materialized FSAs."""
    target = tuple(target)
    std = standard_precover_dfa(fst, target)
    rho = rho_precover_dfa(fst, target)
    assert std.min().equal(rho.min()), (
        f"Language mismatch for target={target}"
    )


# ============================================================
# Equality tests: same FSTs as test_general.py
# ============================================================

class TestEquality:
    """L(PrecoverNFA.det()) = L(SymbolicLazyDeterminize + ExpandRho)."""

    def test_abc(self):
        fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])
        for t in target_prefixes(fst, max_len=3):
            check_equality(fst, t)

    def test_delete_b(self):
        fst = examples.delete_b()
        for t in target_prefixes(fst, max_len=3):
            check_equality(fst, t)

    def test_samuel(self):
        fst = examples.samuel_example()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_small(self):
        fst = examples.small()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_sdd1(self):
        fst = examples.sdd1_fst()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_duplicate(self):
        fst = examples.duplicate(set('123'))
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_number_comma_separator(self):
        fst = examples.number_comma_separator({'a', ',', ' ', '0'}, Digit={'0'})
        check_equality(fst, ())
        check_equality(fst, ('0', ',', '|', ' ', '0', ','))

    def test_newspeak2(self):
        fst = examples.newspeak2()
        check_equality(fst, ())
        check_equality(fst, ('b', 'a'))
        check_equality(fst, ('b', 'a', 'd'))

    def test_lookahead(self):
        fst = examples.lookahead()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_weird_copy(self):
        fst = examples.weird_copy()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_triplets_of_doom(self):
        fst = examples.triplets_of_doom()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_infinite_quotient(self):
        fst = examples.infinite_quotient()
        for t in target_prefixes(fst, max_len=1):
            check_equality(fst, t)

    def test_parity(self):
        fst = examples.parity({'a', 'b'})
        for t in target_prefixes(fst, max_len=1):
            check_equality(fst, t)

    def test_gated_universal(self):
        fst = examples.gated_universal()
        for t in target_prefixes(fst, max_len=1):
            check_equality(fst, t)

    def test_complementary_halves(self):
        fst = examples.complementary_halves()
        for t in target_prefixes(fst, max_len=1):
            check_equality(fst, t)

    def test_shrinking_nonuniversal(self):
        fst = examples.shrinking_nonuniversal()
        for t in target_prefixes(fst, max_len=1):
            check_equality(fst, t)

    def test_anbn(self):
        fst = examples.anbn()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_backticks_to_quote(self):
        fst = examples.backticks_to_quote()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_parity_copy(self):
        fst = examples.parity_copy()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_mystery1(self):
        fst = examples.mystery1()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_mystery2(self):
        fst = examples.mystery2()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_mystery3(self):
        fst = examples.mystery3()
        for t in target_prefixes(fst, max_len=1):
            check_equality(fst, t)

    def test_mystery4(self):
        fst = examples.mystery4()
        for t in target_prefixes(fst, max_len=1):
            check_equality(fst, t)

    def test_mystery5(self):
        fst = examples.mystery5()
        for t in target_prefixes(fst, max_len=1):
            check_equality(fst, t)

    def test_mystery7(self):
        fst = examples.mystery7()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_mystery8(self):
        fst = examples.mystery8()
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_infinite_quotient2(self):
        fst = examples.infinite_quotient2()
        for t in target_prefixes(fst, max_len=1):
            check_equality(fst, t)

    def test_multiple_start_states(self):
        fst = FST()
        fst.add_start(0); fst.add_start(1)
        fst.add_stop(0); fst.add_stop(1)
        fst.add_arc(0, 'a', 'x', 0)
        fst.add_arc(1, 'b', 'x', 1)
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)

    def test_multiple_start_states_shared_sink(self):
        fst = FST()
        fst.add_start(0); fst.add_start(1)
        fst.add_stop(2)
        fst.add_arc(0, 'a', 'x', 2)
        fst.add_arc(1, 'b', 'x', 2)
        fst.add_arc(2, 'a', 'y', 2)
        fst.add_arc(2, 'b', 'y', 2)
        for t in target_prefixes(fst, max_len=2):
            check_equality(fst, t)


# ============================================================
# Structural tests
# ============================================================

class TestStructural:

    def test_dfa_has_rho_arcs_at_boundary(self):
        """Complete DFA states should be factored into RHO arcs."""
        fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c')])
        alphabet = fst.A - {EPSILON}
        sym_dfa = SymbolicLazyDeterminize(PrecoverNFA(fst, ('a',)), alphabet)
        mat = sym_dfa.materialize()
        rho_arcs = [a for s in mat.states for a, _ in mat.arcs(s) if a is RHO]
        assert len(rho_arcs) > 0

    def test_fewer_arcs_than_standard(self):
        fst = examples.replace([(str(i), chr(ord('a') + i)) for i in range(10)])
        target = ('a',)
        alphabet = fst.A - {EPSILON}
        std_mat = standard_precover_dfa(fst, target)
        sym_mat = SymbolicLazyDeterminize(PrecoverNFA(fst, target), alphabet).materialize()
        std_arcs = sum(1 for s in std_mat.states for _ in std_mat.arcs(s))
        sym_arcs = sum(1 for s in sym_mat.states for _ in sym_mat.arcs(s))
        assert sym_arcs < std_arcs

    def test_bpe_like_fst_gets_rho(self):
        """BPE-like FST (epsilon-input trie) should get RHO at boundary."""
        m = FST()
        m.add_start(())
        for i in range(20):
            m.add_arc((), EPSILON, i, (i,))
            m.add_arc((i,), i, EPSILON, ())
        m.add_stop(())
        alphabet = m.A - {EPSILON}
        sym_dfa = SymbolicLazyDeterminize(PrecoverNFA(m, (0,)), alphabet)
        mat = sym_dfa.materialize()
        rho_arcs = sum(1 for s in mat.states for a, _ in mat.arcs(s) if a is RHO)
        total_arcs = sum(1 for s in mat.states for _ in mat.arcs(s))
        assert rho_arcs > 0, "BPE boundary should have RHO arcs"
        # Without factoring, the boundary state would have 20 arcs
        # With factoring, it should have 1 RHO (all go to same dest)
        assert total_arcs < 20

    def test_incomplete_state_not_factored(self):
        """DFA states that aren't complete should NOT get RHO arcs."""
        fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c')])
        alphabet = fst.A - {EPSILON}
        sym_dfa = SymbolicLazyDeterminize(PrecoverNFA(fst, ('a',)), alphabet)
        mat = sym_dfa.materialize()
        # Start state only has arc for '1' (the one that matches target)
        # It should NOT have a RHO arc
        for s in sym_dfa.start():
            arcs = list(sym_dfa.arcs(s))
            rho = [a for a, _ in arcs if a is RHO]
            assert len(rho) == 0, "Incomplete start state should not have RHO"


# ============================================================
# BPE-specific tests
# ============================================================

def _bpe_fst(tokens):
    """Build a BPE-like FST from a list of byte-sequence tokens.

    Each token is a bytes object. The FST maps token-ID sequences to byte
    sequences via an epsilon-input byte trie.
    """
    m = FST()
    m.add_start(())
    for i, tok in enumerate(tokens):
        bx = tuple(tok) if isinstance(tok, bytes) else tok
        for j in range(len(bx)):
            m.add_arc(bx[:j], EPSILON, bx[j], bx[:j+1])
        m.add_arc(bx, i, EPSILON, ())
    m.add_stop(())
    return m


def _check_bpe_equality(tokens, target_bytes):
    """Build a BPE FST and verify rho-factored DFA equals standard DFA."""
    fst = _bpe_fst(tokens)
    target = tuple(target_bytes)
    check_equality(fst, target)


class TestBPE:
    """Equality and structural tests for BPE-like FSTs with rho factoring."""

    # --- Single-byte tokens ---

    def test_single_byte_small(self):
        """Single-byte BPE with 10 tokens."""
        tokens = [bytes([b]) for b in range(10)]
        for target in [(), (0,), (5,), (0, 1)]:
            _check_bpe_equality(tokens, target)

    def test_single_byte_printable_ascii(self):
        """Single-byte BPE with all printable ASCII."""
        tokens = [bytes([b]) for b in range(32, 127)]
        _check_bpe_equality(tokens, (72,))         # 'H'
        _check_bpe_equality(tokens, (72, 105))      # 'Hi'
        _check_bpe_equality(tokens, ())

    # --- Multi-byte tokens ---

    def test_multi_byte_small(self):
        """Mixed single/multi-byte tokens, small vocab."""
        tokens = [b'a', b'b', b'c', b'ab', b'bc', b'abc']
        for target in [(), tuple(b'a'), tuple(b'ab'), tuple(b'abc')]:
            _check_bpe_equality(tokens, target)

    def test_multi_byte_medium(self):
        """Multi-byte tokens with shared prefixes."""
        tokens = [
            b'H', b'He', b'Hel', b'Hell', b'Hello',
            b'Hi', b'Ho', b'Ha',
            b'W', b'Wo', b'Wor', b'World',
            b'a', b'b', b'c',
        ]
        for target in [(), tuple(b'H'), tuple(b'He'), tuple(b'Hel'),
                        tuple(b'Hi'), tuple(b'W'), tuple(b'Wo')]:
            _check_bpe_equality(tokens, target)

    def test_multi_byte_all_same_prefix(self):
        """All tokens share the same first byte."""
        tokens = [bytes([65, i]) for i in range(20)] + [bytes([65])]
        _check_bpe_equality(tokens, (65,))
        _check_bpe_equality(tokens, (65, 0))
        _check_bpe_equality(tokens, (65, 10))

    # --- Scaled BPE (equality-checked at moderate size) ---

    def test_bpe_100_tokens(self):
        """100-token BPE with mixed lengths, equality-checked."""
        import random
        random.seed(123)
        tokens = set()
        for b in range(32, 80):
            tokens.add(bytes([b]))
        while len(tokens) < 100:
            n = random.choice([2, 3])
            tokens.add(bytes([random.choice(range(32, 80)) for _ in range(n)]))
        tokens = sorted(tokens)
        fst = _bpe_fst(tokens)
        for target in [(), (72,), (72, 105), (50, 60, 70)]:
            check_equality(fst, tuple(target))

    def test_bpe_400_tokens(self):
        """~400-token BPE, equality-checked."""
        import random
        random.seed(42)
        tokens = set()
        for b in range(32, 127):
            tokens.add(bytes([b]))
        while len(tokens) < 400:
            n = random.choice([2, 3])
            tokens.add(bytes([random.choice(range(32, 127)) for _ in range(n)]))
        tokens = sorted(tokens)
        fst = _bpe_fst(tokens)
        for target in [(), (72,), (72, 105)]:
            check_equality(fst, tuple(target))

    # --- Structural properties of BPE rho DFAs ---

    def test_bpe_boundary_has_rho(self):
        """BPE boundary state (m==N) should be complete and get RHO."""
        tokens = [bytes([b]) for b in range(50)]
        fst = _bpe_fst(tokens)
        alphabet = fst.A - {EPSILON}
        sym_dfa = SymbolicLazyDeterminize(PrecoverNFA(fst, (0,)), alphabet)
        mat = sym_dfa.materialize()
        rho_count = sum(1 for s in mat.states for a, _ in mat.arcs(s) if a is RHO)
        assert rho_count > 0, "Boundary should have RHO arcs"

    def test_bpe_dramatic_arc_reduction(self):
        """BPE rho DFA should have far fewer arcs than standard."""
        tokens = [bytes([b]) for b in range(50)]
        fst = _bpe_fst(tokens)
        target = (0,)
        alphabet = fst.A - {EPSILON}
        sym_mat = SymbolicLazyDeterminize(
            PrecoverNFA(fst, target), alphabet
        ).materialize()
        std_mat = standard_precover_dfa(fst, target)
        sym_arcs = sum(1 for s in sym_mat.states for _ in sym_mat.arcs(s))
        std_arcs = sum(1 for s in std_mat.states for _ in std_mat.arcs(s))
        # With 50 single-byte tokens, standard has ~51 arcs, rho has ~2
        assert sym_arcs < std_arcs / 5, (
            f"Expected dramatic reduction: {sym_arcs} vs {std_arcs}"
        )

    def test_bpe_dfa_state_count_linear_in_target(self):
        """Number of DFA states should be O(len(target)), not O(vocab)."""
        tokens = [bytes([b]) for b in range(100)]
        fst = _bpe_fst(tokens)
        alphabet = fst.A - {EPSILON}
        for target_len in [1, 2, 3, 5]:
            target = tuple(range(target_len))
            sym_dfa = SymbolicLazyDeterminize(
                PrecoverNFA(fst, target), alphabet
            )
            mat = sym_dfa.materialize()
            # DFA states = target_len + 1 (growing states + boundary)
            assert len(mat.states) == target_len + 1, (
                f"target_len={target_len}: expected {target_len+1} states, "
                f"got {len(mat.states)}"
            )

    def test_bpe_all_same_dest_single_rho(self):
        """When all arcs at boundary go to same dest, should get exactly 1 RHO arc."""
        tokens = [bytes([b]) for b in range(30)]
        fst = _bpe_fst(tokens)
        alphabet = fst.A - {EPSILON}
        sym_dfa = SymbolicLazyDeterminize(PrecoverNFA(fst, (0,)), alphabet)
        # At boundary, all tokens return to hub → same DFA state → 1 RHO
        mat = sym_dfa.materialize()
        for s in mat.states:
            arcs = list(mat.arcs(s))
            rho = [a for a, _ in arcs if a is RHO]
            non_rho = [a for a, _ in arcs if a is not RHO]
            if rho:
                # At most 1 RHO arc per state
                assert len(rho) == 1
                # With all single-byte tokens, boundary should have
                # only 1 explicit + 1 RHO (the token matching target[0]
                # goes to a different state than the rest)
                assert len(non_rho) <= 1, (
                    f"Expected at most 1 explicit arc with RHO, "
                    f"got {len(non_rho)}"
                )


# ============================================================
# Rust cross-validation tests
# ============================================================

def rust_check_equality(fst, target):
    """Verify L(Rust rho expanded) = L(standard precover DFA)."""
    target = tuple(target)
    std = standard_precover_dfa(fst, target)
    rust_rho = RustRhoDeterminize(fst, target)
    rust_expanded = rust_rho.expand()
    assert std.min().equal(rust_expanded.min()), (
        f"Rust rho language mismatch for target={target}"
    )


def rust_python_cross_validate(fst, target):
    """Verify L(Rust rho expanded) = L(Python rho expanded)."""
    target = tuple(target)
    py_rho = rho_precover_dfa(fst, target)
    rust_rho = RustRhoDeterminize(fst, target)
    rust_expanded = rust_rho.expand()
    assert py_rho.min().equal(rust_expanded.min()), (
        f"Rust vs Python rho language mismatch for target={target}"
    )


@pytest.mark.skipif(not HAS_RUST, reason="transduction_core not built")
class TestRustEquality:
    """L(Rust rho expanded) = L(standard precover DFA) = L(Python rho expanded)."""

    def test_abc(self):
        fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])
        for t in target_prefixes(fst, max_len=3):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_delete_b(self):
        fst = examples.delete_b()
        for t in target_prefixes(fst, max_len=3):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_samuel(self):
        fst = examples.samuel_example()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_small(self):
        fst = examples.small()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_sdd1(self):
        fst = examples.sdd1_fst()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_duplicate(self):
        fst = examples.duplicate(set('123'))
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_number_comma_separator(self):
        fst = examples.number_comma_separator({'a', ',', ' ', '0'}, Digit={'0'})
        for t in [(), ('0', ',', '|', ' ', '0', ',')]:
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_newspeak2(self):
        fst = examples.newspeak2()
        for t in [(), ('b', 'a'), ('b', 'a', 'd')]:
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_lookahead(self):
        fst = examples.lookahead()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_weird_copy(self):
        fst = examples.weird_copy()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_triplets_of_doom(self):
        fst = examples.triplets_of_doom()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_infinite_quotient(self):
        fst = examples.infinite_quotient()
        for t in target_prefixes(fst, max_len=1):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_parity(self):
        fst = examples.parity({'a', 'b'})
        for t in target_prefixes(fst, max_len=1):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_gated_universal(self):
        fst = examples.gated_universal()
        for t in target_prefixes(fst, max_len=1):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_complementary_halves(self):
        fst = examples.complementary_halves()
        for t in target_prefixes(fst, max_len=1):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_shrinking_nonuniversal(self):
        fst = examples.shrinking_nonuniversal()
        for t in target_prefixes(fst, max_len=1):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_anbn(self):
        fst = examples.anbn()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_backticks_to_quote(self):
        fst = examples.backticks_to_quote()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_parity_copy(self):
        fst = examples.parity_copy()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_mystery1(self):
        fst = examples.mystery1()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_mystery2(self):
        fst = examples.mystery2()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_mystery3(self):
        fst = examples.mystery3()
        for t in target_prefixes(fst, max_len=1):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_mystery4(self):
        fst = examples.mystery4()
        for t in target_prefixes(fst, max_len=1):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_mystery5(self):
        fst = examples.mystery5()
        for t in target_prefixes(fst, max_len=1):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_mystery7(self):
        fst = examples.mystery7()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_mystery8(self):
        fst = examples.mystery8()
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_infinite_quotient2(self):
        fst = examples.infinite_quotient2()
        for t in target_prefixes(fst, max_len=1):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_multiple_start_states(self):
        fst = FST()
        fst.add_start(0); fst.add_start(1)
        fst.add_stop(0); fst.add_stop(1)
        fst.add_arc(0, 'a', 'x', 0)
        fst.add_arc(1, 'b', 'x', 1)
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)

    def test_multiple_start_states_shared_sink(self):
        fst = FST()
        fst.add_start(0); fst.add_start(1)
        fst.add_stop(2)
        fst.add_arc(0, 'a', 'x', 2)
        fst.add_arc(1, 'b', 'x', 2)
        fst.add_arc(2, 'a', 'y', 2)
        fst.add_arc(2, 'b', 'y', 2)
        for t in target_prefixes(fst, max_len=2):
            rust_check_equality(fst, t)
            rust_python_cross_validate(fst, t)


@pytest.mark.skipif(not HAS_RUST, reason="transduction_core not built")
class TestRustStructural:
    """Verify structural properties of the Rust rho DFA."""

    def test_rho_arcs_at_boundary(self):
        """Complete states should produce RHO arcs."""
        fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c')])
        result = RustRhoDeterminize(fst, ('a',))
        assert result.num_rho_arcs > 0

    def test_fewer_arcs_than_standard(self):
        fst = examples.replace([(str(i), chr(ord('a') + i)) for i in range(10)])
        target = ('a',)
        std_mat = standard_precover_dfa(fst, target)
        std_arcs = sum(1 for s in std_mat.states for _ in std_mat.arcs(s))
        result = RustRhoDeterminize(fst, target)
        assert result.total_arcs < std_arcs

    def test_stats_consistency(self):
        """num_rho_arcs + num_explicit_arcs == total arcs in the DFA."""
        fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c')])
        result = RustRhoDeterminize(fst, ('a',))
        assert result.num_rho_arcs + result.num_explicit_arcs == result.total_arcs


@pytest.mark.skipif(not HAS_RUST, reason="transduction_core not built")
class TestRustBPE:
    """BPE-specific tests for the Rust rho implementation."""

    def test_single_byte_small(self):
        tokens = [bytes([b]) for b in range(10)]
        for target in [(), (0,), (5,), (0, 1)]:
            _check_bpe_equality(tokens, target)
            fst = _bpe_fst(tokens)
            rust_check_equality(fst, target)

    def test_single_byte_printable_ascii(self):
        tokens = [bytes([b]) for b in range(32, 127)]
        for target in [(), (72,), (72, 105)]:
            fst = _bpe_fst(tokens)
            rust_check_equality(fst, target)

    def test_multi_byte_small(self):
        tokens = [b'a', b'b', b'c', b'ab', b'bc', b'abc']
        for target in [(), tuple(b'a'), tuple(b'ab'), tuple(b'abc')]:
            fst = _bpe_fst(tokens)
            rust_check_equality(fst, target)

    def test_multi_byte_medium(self):
        tokens = [
            b'H', b'He', b'Hel', b'Hell', b'Hello',
            b'Hi', b'Ho', b'Ha',
            b'W', b'Wo', b'Wor', b'World',
            b'a', b'b', b'c',
        ]
        for target in [(), tuple(b'H'), tuple(b'He'), tuple(b'Hel'),
                        tuple(b'Hi'), tuple(b'W'), tuple(b'Wo')]:
            fst = _bpe_fst(tokens)
            rust_check_equality(fst, target)

    def test_bpe_100_tokens(self):
        import random
        random.seed(123)
        tokens = set()
        for b in range(32, 80):
            tokens.add(bytes([b]))
        while len(tokens) < 100:
            n = random.choice([2, 3])
            tokens.add(bytes([random.choice(range(32, 80)) for _ in range(n)]))
        tokens = sorted(tokens)
        fst = _bpe_fst(tokens)
        for target in [(), (72,), (72, 105), (50, 60, 70)]:
            rust_check_equality(fst, tuple(target))

    def test_bpe_400_tokens(self):
        import random
        random.seed(42)
        tokens = set()
        for b in range(32, 127):
            tokens.add(bytes([b]))
        while len(tokens) < 400:
            n = random.choice([2, 3])
            tokens.add(bytes([random.choice(range(32, 127)) for _ in range(n)]))
        tokens = sorted(tokens)
        fst = _bpe_fst(tokens)
        for target in [(), (72,), (72, 105)]:
            rust_check_equality(fst, tuple(target))

    def test_bpe_boundary_has_rho(self):
        tokens = [bytes([b]) for b in range(50)]
        fst = _bpe_fst(tokens)
        result = RustRhoDeterminize(fst, (0,))
        assert result.num_rho_arcs > 0, "BPE boundary should have RHO arcs"

    def test_bpe_dramatic_arc_reduction(self):
        tokens = [bytes([b]) for b in range(50)]
        fst = _bpe_fst(tokens)
        target = (0,)
        std_mat = standard_precover_dfa(fst, target)
        std_arcs = sum(1 for s in std_mat.states for _ in std_mat.arcs(s))
        result = RustRhoDeterminize(fst, target)
        assert result.total_arcs < std_arcs / 5, (
            f"Expected dramatic reduction: {result.total_arcs} vs {std_arcs}"
        )
