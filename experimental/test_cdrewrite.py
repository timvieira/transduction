"""
Tests for context-dependent rewrite rule compilation (cdrewrite).
"""
import pytest
from itertools import islice

from transduction.fst import FST, EPSILON
from transduction.fsa import FSA
from transduction.experimental.cdrewrite import cdrewrite, cross, union_fst, BOS, EOS


def _relation_set(fst, max_length=10):
    """Extract the relation of an FST as a set of (input, output) pairs."""
    return set(fst.relation(max_length))


def _apply(rule_fst, input_str, max_length=20):
    """Apply a cdrewrite rule FST to an input string, return set of outputs."""
    input_fst = FST.from_string(input_str)
    composed = input_fst @ rule_fst
    outputs = set()
    for x, y in composed.relation(max_length):
        if x == input_str:
            outputs.add(y)
    return outputs


def _apply_one(rule_fst, input_str, max_length=20):
    """Apply a cdrewrite rule FST, expecting exactly one output."""
    outputs = _apply(rule_fst, input_str, max_length)
    assert len(outputs) == 1, f'Expected 1 output, got {len(outputs)}: {outputs}'
    return outputs.pop()


class TestCross:
    """Tests for the cross() helper."""

    def test_cross_strings_same_length(self):
        fst = cross('ab', 'cd')
        rel = _relation_set(fst)
        assert ('ab', 'cd') in rel

    def test_cross_strings_different_length(self):
        fst = cross('a', 'bc')
        rel = _relation_set(fst)
        assert ('a', 'bc') in rel

    def test_cross_strings_empty(self):
        fst = cross('a', '')
        rel = _relation_set(fst)
        assert ('a', '') in rel

    def test_cross_single_char(self):
        fst = cross('a', 'b')
        rel = _relation_set(fst)
        assert ('a', 'b') in rel


class TestCdrewriteNoContext:
    """Tests for cdrewrite with no left/right context (unconditional rewrite)."""

    def setup_method(self):
        self.sigma = {'a', 'b', 'c'}
        self.sigma_star = FSA.universal(self.sigma)

    def test_simple_rewrite(self):
        """a -> b (unconditional)"""
        tau = cross('a', 'b')
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'a') == 'b'
        assert _apply_one(rule, 'b') == 'b'
        assert _apply_one(rule, 'c') == 'c'

    def test_rewrite_in_context(self):
        """a -> b applied to 'abc'"""
        tau = cross('a', 'b')
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'abc') == 'bbc'

    def test_multiple_occurrences(self):
        """a -> b applied to 'aba'"""
        tau = cross('a', 'b')
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'aba') == 'bbb'

    def test_no_match(self):
        """a -> b applied to 'bc' (no a to rewrite)"""
        tau = cross('a', 'b')
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'bc') == 'bc'

    def test_empty_string(self):
        """a -> b applied to '' (empty input)"""
        tau = cross('a', 'b')
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, '') == ''

    def test_multi_symbol_rewrite(self):
        """ab -> c (unconditional)"""
        tau = cross('ab', 'c')
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'ab') == 'c'
        assert _apply_one(rule, 'aab') == 'ac'

    def test_expansion_rewrite(self):
        """a -> bc (unconditional, one-to-many)"""
        tau = cross('a', 'bc')
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'a') == 'bc'
        assert _apply_one(rule, 'aa') == 'bcbc'

    def test_deletion_rewrite(self):
        """a -> '' (deletion)"""
        tau = cross('a', '')
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'a') == ''
        assert _apply_one(rule, 'abc') == 'bc'
        assert _apply_one(rule, 'bac') == 'bc'


class TestCdrewriteWithContext:
    """Tests for cdrewrite with left and/or right context."""

    def setup_method(self):
        self.sigma = {'a', 'b', 'c', 'd'}
        self.sigma_star = FSA.universal(self.sigma)

    def test_left_context_only(self):
        """a -> b / c _ (rewrite a to b only after c)"""
        tau = cross('a', 'b')
        lc = FSA.from_string('c')
        rule = cdrewrite(tau, lambda_=lc, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'ca') == 'cb'
        assert _apply_one(rule, 'da') == 'da'  # no match: d is not c
        assert _apply_one(rule, 'a') == 'a'    # no left context

    def test_right_context_only(self):
        """a -> b / _ d (rewrite a to b only before d)"""
        tau = cross('a', 'b')
        rc = FSA.from_string('d')
        rule = cdrewrite(tau, rho=rc, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'ad') == 'bd'
        assert _apply_one(rule, 'ac') == 'ac'  # no match: c is not d
        assert _apply_one(rule, 'a') == 'a'    # no right context

    def test_both_contexts(self):
        """a -> b / c _ d (rewrite a to b only between c and d)"""
        tau = cross('a', 'b')
        lc = FSA.from_string('c')
        rc = FSA.from_string('d')
        rule = cdrewrite(tau, lambda_=lc, rho=rc, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'cad') == 'cbd'
        assert _apply_one(rule, 'cac') == 'cac'  # right context doesn't match
        assert _apply_one(rule, 'dad') == 'dad'  # left context doesn't match
        assert _apply_one(rule, 'a') == 'a'

    def test_context_with_multiple_matches(self):
        """a -> b / c _ d applied to 'cadcad'"""
        tau = cross('a', 'b')
        lc = FSA.from_string('c')
        rc = FSA.from_string('d')
        rule = cdrewrite(tau, lambda_=lc, rho=rc, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'cadcad') == 'cbdcbd'


class TestCdrewriteBOS:
    """Tests for cdrewrite with BOS (beginning of string) context."""

    def setup_method(self):
        self.sigma = {'a', 'b', 'c'}
        self.sigma_star = FSA.universal(self.sigma)

    def test_bos_left_context(self):
        """a -> b / [BOS] _ (rewrite a only at start of string)"""
        tau = cross('a', 'b')
        lc = FSA()
        lc.add_start(0)
        lc.add_arc(0, BOS, 1)
        lc.add_stop(1)
        rule = cdrewrite(tau, lambda_=lc, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'a') == 'b'
        assert _apply_one(rule, 'aa') == 'ba'    # only first a is at BOS
        assert _apply_one(rule, 'ba') == 'ba'    # a is not at BOS


class TestCdrewriteEOS:
    """Tests for cdrewrite with EOS (end of string) context."""

    def setup_method(self):
        self.sigma = {'a', 'b', 'c'}
        self.sigma_star = FSA.universal(self.sigma)

    def test_eos_right_context(self):
        """a -> b / _ [EOS] (rewrite a only at end of string)"""
        tau = cross('a', 'b')
        rc = FSA()
        rc.add_start(0)
        rc.add_arc(0, EOS, 1)
        rc.add_stop(1)
        rule = cdrewrite(tau, rho=rc, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'a') == 'b'
        assert _apply_one(rule, 'aa') == 'ab'     # only last a is at EOS
        assert _apply_one(rule, 'ab') == 'ab'     # a is not at EOS


class TestCdrewriteComposition:
    """Tests for composing multiple cdrewrite rules."""

    def setup_method(self):
        self.sigma = {'a', 'b', 'c', 'd'}
        self.sigma_star = FSA.universal(self.sigma)

    def test_two_rule_composition(self):
        """a -> b then b -> c"""
        rule1 = cdrewrite(cross('a', 'b'), sigma_star=self.sigma_star)
        rule2 = cdrewrite(cross('b', 'c'), sigma_star=self.sigma_star)
        composed = rule1 @ rule2
        # 'a' -> 'b' (rule1) -> 'c' (rule2)
        assert _apply_one(composed, 'a') == 'c'
        # 'b' -> 'b' (rule1, no change) -> 'c' (rule2)
        assert _apply_one(composed, 'b') == 'c'

    def test_non_overlapping_composition(self):
        """a -> b then c -> d"""
        rule1 = cdrewrite(cross('a', 'b'), sigma_star=self.sigma_star)
        rule2 = cdrewrite(cross('c', 'd'), sigma_star=self.sigma_star)
        composed = rule1 @ rule2
        assert _apply_one(composed, 'ac') == 'bd'


class TestCdrewriteUnion:
    """Tests for cdrewrite with union of rewrite rules (multiple tau)."""

    def setup_method(self):
        self.sigma = {'a', 'b', 'c', 'd'}
        self.sigma_star = FSA.universal(self.sigma)

    def test_union_rewrite(self):
        """(a -> c | b -> d) unconditional via union of cross products"""
        tau = union_fst(cross('a', 'c'), cross('b', 'd'))
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'a') == 'c'
        assert _apply_one(rule, 'b') == 'd'
        assert _apply_one(rule, 'ab') == 'cd'
        assert _apply_one(rule, 'c') == 'c'


class TestCdrewriteMultiCharContext:
    """Tests with multi-character contexts."""

    def setup_method(self):
        self.sigma = {'a', 'b', 'c', 'd'}
        self.sigma_star = FSA.universal(self.sigma)

    def test_multi_char_left_context(self):
        """a -> b / cd _ (rewrite a after 'cd')"""
        tau = cross('a', 'b')
        lc = FSA.from_string('cd')
        rule = cdrewrite(tau, lambda_=lc, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'cda') == 'cdb'
        assert _apply_one(rule, 'ca') == 'ca'   # only 'c', not 'cd'
        assert _apply_one(rule, 'da') == 'da'

    def test_multi_char_right_context(self):
        """a -> b / _ cd (rewrite a before 'cd')"""
        tau = cross('a', 'b')
        rc = FSA.from_string('cd')
        rule = cdrewrite(tau, rho=rc, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'acd') == 'bcd'
        assert _apply_one(rule, 'ac') == 'ac'   # only 'c', not 'cd'
        assert _apply_one(rule, 'ad') == 'ad'


class TestCdrewriteOverlapping:
    """Tests with overlapping matches."""

    def setup_method(self):
        self.sigma = {'a', 'b', 'c'}
        self.sigma_star = FSA.universal(self.sigma)

    def test_overlapping_phi_ltr(self):
        """aa -> b, LTR obligatory: 'aaa' -> 'ba' (leftmost match)"""
        tau = cross('aa', 'b')
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'aa') == 'b'
        assert _apply_one(rule, 'aaa') == 'ba'   # leftmost 'aa' replaced
        assert _apply_one(rule, 'aaaa') == 'bb'   # two non-overlapping 'aa'

    def test_identity_passthrough(self):
        """Symbols not in phi pass through unchanged."""
        tau = cross('a', 'b')
        rule = cdrewrite(tau, sigma_star=self.sigma_star)
        assert _apply_one(rule, 'ccc') == 'ccc'
        assert _apply_one(rule, 'bbb') == 'bbb'


class TestCdrewriteEdgeCases:
    """Edge cases."""

    def test_single_symbol_alphabet(self):
        """Works with single-symbol alphabet."""
        sigma_star = FSA.universal({'a'})
        tau = cross('a', 'a')  # identity rule
        rule = cdrewrite(tau, sigma_star=sigma_star)
        assert _apply_one(rule, 'a') == 'a'
        assert _apply_one(rule, 'aaa') == 'aaa'

    def test_larger_alphabet(self):
        """Works with larger alphabet."""
        sigma = set('abcdefgh')
        sigma_star = FSA.universal(sigma)
        tau = cross('a', 'x')
        rule = cdrewrite(tau, sigma_star=sigma_star)
        assert _apply_one(rule, 'abcdefgh') == 'xbcdefgh'

    def test_context_plus(self):
        """a -> b / (c+) _ where c+ means one or more c's."""
        sigma = {'a', 'b', 'c'}
        sigma_star = FSA.universal(sigma)
        tau = cross('a', 'b')
        # Build c+ = c . c*
        lc = FSA.from_string('c').p()  # c+
        rule = cdrewrite(tau, lambda_=lc, sigma_star=sigma_star)
        assert _apply_one(rule, 'ca') == 'cb'
        assert _apply_one(rule, 'cca') == 'ccb'
        assert _apply_one(rule, 'a') == 'a'     # no c before a
