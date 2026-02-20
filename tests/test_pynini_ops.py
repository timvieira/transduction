"""Tests for pynini-based decomposition operations.

Compares PyniniDecomposition against the reference Precover implementation
across example FSTs.
"""

import pytest
from itertools import islice

from transduction import examples, EPSILON, Precover
from transduction.fst import FST
from transduction.fsa import FSA

try:
    from transduction.pynini_ops import (
        PyniniDecomposition,
        native_fst_to_pynini,
        pynini_acceptor_to_native_fsa,
    )
    HAS_PYNINI = True
except ImportError:
    HAS_PYNINI = False

pytestmark = pytest.mark.skipif(not HAS_PYNINI, reason="pynini not installed")


# ─── Helper functions ────────────────────────────────────────────────

def finite_language(fsa, max_length=10):
    """Enumerate language of an FSA up to max_length as a set of tuples."""
    return set(islice(fsa.language(max_length=max_length), 500))


def assert_languages_equal(fsa1, fsa2, max_length=10):
    """Assert that two FSAs accept the same language (up to max_length for finite check).

    Uses FSA.equal() which does minimal-DFA isomorphism.
    """
    assert fsa1.min().equal(fsa2.min()), (
        f"Languages differ.\n"
        f"  fsa1 sample: {sorted(finite_language(fsa1, max_length))[:10]}\n"
        f"  fsa2 sample: {sorted(finite_language(fsa2, max_length))[:10]}"
    )


# ─── Example FSTs to test ───────────────────────────────────────────

EXAMPLE_FSTS = [
    pytest.param(examples.small, id="small"),
    pytest.param(examples.weird_copy, id="weird_copy"),
    pytest.param(examples.samuel_example, id="samuel_example"),
    pytest.param(examples.lookahead, id="lookahead"),
    pytest.param(examples.triplets_of_doom, id="triplets_of_doom"),
    pytest.param(lambda: examples.replace([('a', 'x'), ('b', 'y')]), id="replace_ab"),
    pytest.param(examples.mystery1, id="mystery1"),
    pytest.param(examples.mystery7, id="mystery7"),
    pytest.param(examples.mystery8, id="mystery8"),
]


@pytest.fixture(params=EXAMPLE_FSTS)
def fst_factory(request):
    return request.param


# ─── Conversion tests ───────────────────────────────────────────────

class TestConversion:
    """Test native FST <-> pynini conversion."""

    def test_roundtrip_relation(self, fst_factory):
        """Verify that native->pynini conversion preserves the FST relation."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)

        # Check that the pynini FST has the right structure
        import pynini
        pfst = pd.pfst
        assert pfst.start() != pynini.NO_STATE_ID

        # Verify the relation matches by checking some string pairs
        for xs, ys in islice(fst.relation(max_length=6), 50):
            # Each (xs, ys) pair from the native FST should produce a valid
            # composition path in the pynini FST
            in_labels = [pd.in_label_map[x] for x in xs]
            out_labels = [pd.out_label_map[y] for y in ys]

            # Build string acceptors and compose
            in_acc = _pynini_string_acceptor(in_labels, pd.in_sym_table)
            out_acc = _pynini_string_acceptor(out_labels, pd.out_sym_table)

            composed = pynini.compose(in_acc, pynini.compose(pfst, out_acc))
            # The composition should be non-empty (have at least one accepting path)
            try:
                paths = list(composed.paths())
                assert len(list(composed.paths())) > 0 or _has_accepting_path(composed), (
                    f"Pair ({xs}, {ys}) not found in pynini FST"
                )
            except Exception:
                # paths() might fail on empty FST; check num_states
                assert _has_accepting_path(composed), (
                    f"Pair ({xs}, {ys}) not found in pynini FST"
                )


def _pynini_string_acceptor(labels, sym_table):
    """Build a pynini acceptor for a single string (sequence of labels)."""
    import pynini
    fst = pynini.Fst()
    n = len(labels)
    for _ in range(n + 1):
        fst.add_state()
    fst.set_start(0)
    fst.set_final(n)
    for i, label in enumerate(labels):
        fst.add_arc(i, pynini.Arc(label, label, 0, i + 1))
    fst.set_input_symbols(sym_table)
    fst.set_output_symbols(sym_table)
    return fst


def _has_accepting_path(pfst):
    """Check if a pynini FST has at least one accepting path."""
    import pynini
    zero = pynini.Weight.zero(pfst.weight_type())
    start = pfst.start()
    if start == pynini.NO_STATE_ID:
        return False
    # BFS
    visited = {start}
    queue = [start]
    while queue:
        state = queue.pop(0)
        if pfst.final(state) != zero:
            return True
        for arc in pfst.arcs(state):
            if arc.nextstate not in visited:
                visited.add(arc.nextstate)
                queue.append(arc.nextstate)
    return False


# ─── Precover (P) tests ─────────────────────────────────────────────

class TestPrecover:
    """Test P(y) equivalence against reference Precover."""

    def test_empty_target(self, fst_factory):
        """P('') should equal the domain of the FST."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)
        ref = Precover(fst, '')

        pynini_P = pd.precover_as_native_fsa('')
        ref_P = ref.det.materialize()

        assert_languages_equal(pynini_P, ref_P)

    def test_single_symbol_targets(self, fst_factory):
        """P(y) for each single output symbol."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)

        for y in fst.B - {EPSILON}:
            target = (y,)
            pynini_P = pd.precover_as_native_fsa(target)
            ref = Precover(fst, target)
            ref_P = ref.det.materialize()
            assert_languages_equal(pynini_P, ref_P)

    def test_multi_symbol_targets(self, fst_factory):
        """P(y) for longer target strings (up to length 3)."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)

        # Generate target strings from output pairs in the FST relation
        targets_seen = set()
        for _xs, ys in islice(fst.relation(max_length=5), 50):
            for length in range(1, min(len(ys) + 1, 4)):
                target = ys[:length]
                if target in targets_seen:
                    continue
                targets_seen.add(target)

                pynini_P = pd.precover_as_native_fsa(target)
                ref = Precover(fst, target)
                ref_P = ref.det.materialize()
                assert_languages_equal(pynini_P, ref_P)


# ─── Quotient/Remainder (Q/R) tests ─────────────────────────────────

class TestQuotientRemainder:
    """Test Q(y) and R(y) equivalence against reference Precover."""

    def test_empty_target(self, fst_factory):
        """Q('') and R('') should match reference."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)
        ref = Precover(fst, '')

        pynini_Q = pd.quotient_as_native_fsa('')
        pynini_R = pd.remainder_as_native_fsa('')

        ref_Q = ref.quotient.trim()
        ref_R = ref.remainder.trim()

        assert_languages_equal(pynini_Q, ref_Q)
        assert_languages_equal(pynini_R, ref_R)

    def test_single_symbol_targets(self, fst_factory):
        """Q(y) and R(y) for each single output symbol."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)

        for y in fst.B - {EPSILON}:
            target = (y,)
            ref = Precover(fst, target)

            pynini_Q = pd.quotient_as_native_fsa(target)
            pynini_R = pd.remainder_as_native_fsa(target)

            ref_Q = ref.quotient.trim()
            ref_R = ref.remainder.trim()

            assert_languages_equal(pynini_Q, ref_Q), f"Q mismatch for target={target}"
            assert_languages_equal(pynini_R, ref_R), f"R mismatch for target={target}"

    def test_multi_symbol_targets(self, fst_factory):
        """Q(y) and R(y) for longer targets."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)

        targets_seen = set()
        for _xs, ys in islice(fst.relation(max_length=5), 30):
            for length in range(1, min(len(ys) + 1, 4)):
                target = ys[:length]
                if target in targets_seen:
                    continue
                targets_seen.add(target)

                ref = Precover(fst, target)
                pynini_Q = pd.quotient_as_native_fsa(target)
                pynini_R = pd.remainder_as_native_fsa(target)

                ref_Q = ref.quotient.trim()
                ref_R = ref.remainder.trim()

                assert_languages_equal(pynini_Q, ref_Q)
                assert_languages_equal(pynini_R, ref_R)


# ─── Remainder identity: R = P - Q·Σ* ───────────────────────────────

class TestRemainderIdentity:
    """Verify R(y) = P(y) - Q(y)·Sigma_in* using native FSA operations."""

    def test_identity(self, fst_factory):
        """R(y) should equal P(y) - Q(y)·Σ*."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)
        source_alphabet = fst.A - {EPSILON}
        U = FSA.universal(source_alphabet)

        targets_seen = set()
        for _xs, ys in islice(fst.relation(max_length=5), 30):
            for length in range(0, min(len(ys) + 1, 3)):
                target = ys[:length]
                if target in targets_seen:
                    continue
                targets_seen.add(target)

                P = pd.precover_as_native_fsa(target)
                Q = pd.quotient_as_native_fsa(target)
                R = pd.remainder_as_native_fsa(target)

                # R should equal P - Q·U
                expected_R = (P - Q * U).min()
                assert_languages_equal(R, expected_R)


# ─── Membership tests ───────────────────────────────────────────────

class TestMembership:
    """Test is_in_precover, is_in_quotient, is_in_remainder."""

    def test_membership_consistency(self, fst_factory):
        """For strings in the precover, check Q/R classification matches reference."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)

        targets_seen = set()
        for _xs, ys in islice(fst.relation(max_length=5), 30):
            for length in range(0, min(len(ys) + 1, 3)):
                target = ys[:length]
                if target in targets_seen:
                    continue
                targets_seen.add(target)

                ref = Precover(fst, target)
                ref_Q = ref.quotient.trim()
                ref_R = ref.remainder.trim()
                ref_P = ref.det.materialize()

                # Test some source strings from the precover
                for src in islice(ref_P.language(max_length=6), 20):
                    assert pd.is_in_precover(target, src), (
                        f"{src} should be in P({target})"
                    )

                    in_Q_ref = src in ref_Q.min()
                    in_R_ref = src in ref_R.min()
                    in_Q_pynini = pd.is_in_quotient(target, src)
                    in_R_pynini = pd.is_in_remainder(target, src)

                    assert in_Q_pynini == in_Q_ref, (
                        f"Q membership mismatch for src={src}, target={target}: "
                        f"pynini={in_Q_pynini}, ref={in_Q_ref}"
                    )
                    assert in_R_pynini == in_R_ref, (
                        f"R membership mismatch for src={src}, target={target}: "
                        f"pynini={in_R_pynini}, ref={in_R_ref}"
                    )

    def test_not_in_precover(self, fst_factory):
        """Strings not in the precover should return False for all membership tests."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)
        source_syms = sorted(fst.A - {EPSILON}, key=repr)
        if not source_syms:
            return

        for y in fst.B - {EPSILON}:
            target = (y,)
            ref = Precover(fst, target)
            ref_P = ref.det.materialize()

            # Try some strings and check ones NOT in precover
            for length in range(0, 4):
                for src in _generate_strings(source_syms, length):
                    if src not in ref_P.min():
                        assert not pd.is_in_precover(target, src)
                        assert not pd.is_in_quotient(target, src)
                        assert not pd.is_in_remainder(target, src)
            break  # test one target symbol to keep fast


# ─── Prefix tests ───────────────────────────────────────────────────

class TestPrefix:
    """Test is_prefix_of_precover."""

    def test_valid_prefixes(self, fst_factory):
        """Prefixes of precover strings should be detected."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)

        for y in fst.B - {EPSILON}:
            target = (y,)
            ref = Precover(fst, target)
            ref_P = ref.det.materialize()

            for src in islice(ref_P.language(max_length=5), 10):
                # All prefixes of src should be valid prefixes of the precover
                for k in range(len(src) + 1):
                    prefix = src[:k]
                    assert pd.is_prefix_of_precover(target, prefix), (
                        f"Prefix {prefix} of {src} should be prefix of P({target})"
                    )
            break  # test one target symbol

    def test_invalid_prefixes(self, fst_factory):
        """Strings that are definitely not prefixes should return False."""
        fst = fst_factory()
        pd = PyniniDecomposition(fst)
        source_syms = sorted(fst.A - {EPSILON}, key=repr)
        if not source_syms:
            return

        for y in fst.B - {EPSILON}:
            target = (y,)
            ref = Precover(fst, target)
            ref_P = ref.det.materialize()

            for length in range(0, 5):
                for src in _generate_strings(source_syms, length):
                    # Check if src is a prefix of anything in P
                    is_prefix_ref = _is_prefix_in_fsa(ref_P, src)
                    is_prefix_pynini = pd.is_prefix_of_precover(target, src)
                    assert is_prefix_pynini == is_prefix_ref, (
                        f"Prefix check mismatch for src={src}, target={target}"
                    )
            break


# ─── Specific example tests ─────────────────────────────────────────

class TestSpecificExamples:
    """Hand-verified test cases for specific FSTs."""

    def test_small_precover_empty_target(self):
        """small() FST: P('') is the domain = {ε, a, b} · {a,b}*."""
        fst = examples.small()
        pd = PyniniDecomposition(fst)
        ref = Precover(fst, '')

        P = pd.precover_as_native_fsa('')
        ref_P = ref.det.materialize()
        assert_languages_equal(P, ref_P)

    def test_small_known_strings(self):
        """small() FST: verify specific known memberships."""
        fst = examples.small()
        pd = PyniniDecomposition(fst)

        # small() maps: a->x (accept), ba->xa, bb->xb, baa->xaa, etc.
        # P('x') should include 'a', 'ba', 'bb', 'baa', 'bab', etc.
        assert pd.is_in_precover(('x',), ('a',))
        assert pd.is_in_precover(('x',), ('b', 'a'))
        assert pd.is_in_precover(('x',), ('b', 'b'))

    def test_weird_copy_decomposition(self):
        """weird_copy(): identity transducer on {a,b}. Q/R should match."""
        fst = examples.weird_copy()
        pd = PyniniDecomposition(fst)

        for y in [(), ('a',), ('b',), ('a', 'b')]:
            ref = Precover(fst, y)
            pynini_Q = pd.quotient_as_native_fsa(y)
            pynini_R = pd.remainder_as_native_fsa(y)
            assert_languages_equal(pynini_Q, ref.quotient.trim())
            assert_languages_equal(pynini_R, ref.remainder.trim())

    def test_samuel_example_decomposition(self):
        """samuel_example(): a tricky FST with epsilon outputs."""
        fst = examples.samuel_example()
        pd = PyniniDecomposition(fst)

        for y in [(), ('c',), ('y',), ('c', 'x')]:
            ref = Precover(fst, y)
            pynini_Q = pd.quotient_as_native_fsa(y)
            pynini_R = pd.remainder_as_native_fsa(y)
            assert_languages_equal(pynini_Q, ref.quotient.trim())
            assert_languages_equal(pynini_R, ref.remainder.trim())


# ─── Infinite quotient tests ────────────────────────────────────────

class TestInfiniteQuotient:
    """Test FSTs with infinite quotients (Q(y) is an infinite language)."""

    def test_delete_b(self):
        """delete_b(): has infinite quotients due to b-deletion."""
        fst = examples.delete_b()
        pd = PyniniDecomposition(fst)

        # P('') is the domain: all strings over {a, b}
        # Q('') should accept all strings (everything is a valid quotient prefix
        # since any input can produce empty output)
        ref = Precover(fst, '')
        pynini_Q = pd.quotient_as_native_fsa('')
        pynini_R = pd.remainder_as_native_fsa('')
        assert_languages_equal(pynini_Q, ref.quotient.trim())
        assert_languages_equal(pynini_R, ref.remainder.trim())

        # P('A') - strings that transduce to something starting with 'A'
        ref_A = Precover(fst, ('A',))
        pynini_Q_A = pd.quotient_as_native_fsa(('A',))
        pynini_R_A = pd.remainder_as_native_fsa(('A',))
        assert_languages_equal(pynini_Q_A, ref_A.quotient.trim())
        assert_languages_equal(pynini_R_A, ref_A.remainder.trim())

    def test_triplets_of_doom(self):
        """triplets_of_doom(): copy transducer for (aaa|bbb)*."""
        fst = examples.triplets_of_doom()
        pd = PyniniDecomposition(fst)

        for y in [(), ('a',), ('a', 'a'), ('a', 'a', 'a'), ('b',)]:
            ref = Precover(fst, y)
            pynini_Q = pd.quotient_as_native_fsa(y)
            pynini_R = pd.remainder_as_native_fsa(y)
            assert_languages_equal(pynini_Q, ref.quotient.trim())
            assert_languages_equal(pynini_R, ref.remainder.trim())


# ─── Benchmark section ───────────────────────────────────────────────

class TestBenchmark:
    """Performance comparison (not strict timing, but correctness + smoke)."""

    def test_benchmark_small(self):
        """Benchmark on small() with various target lengths."""
        import time
        fst = examples.small()
        pd = PyniniDecomposition(fst)

        targets = [(), ('x',), ('x', 'a'), ('x', 'b'), ('x', 'a', 'b')]
        for target in targets:
            t0 = time.time()
            P = pd.precover(target)
            Q = pd.quotient(target)
            R = pd.remainder(target)
            t_pynini = time.time() - t0

            t0 = time.time()
            ref = Precover(fst, target)
            _ = ref.quotient
            _ = ref.remainder
            t_ref = time.time() - t0

            # Just verify correctness (timing is informational)
            pynini_Q = pd.quotient_as_native_fsa(target)
            pynini_R = pd.remainder_as_native_fsa(target)
            assert_languages_equal(pynini_Q, ref.quotient.trim())
            assert_languages_equal(pynini_R, ref.remainder.trim())


# ─── Utility functions ───────────────────────────────────────────────

def _generate_strings(alphabet, length):
    """Generate all strings of a given length over alphabet."""
    if length == 0:
        yield ()
        return
    for s in _generate_strings(alphabet, length - 1):
        for a in alphabet:
            yield s + (a,)


def _is_prefix_in_fsa(fsa, prefix):
    """Check if prefix is a prefix of some string in the FSA.

    Uses the trimmed FSA: if we can reach any state after reading prefix,
    then it's a valid prefix (since all states in a trimmed FSA are
    coaccessible).
    """
    trimmed = fsa.trim()
    if not trimmed.states:
        return len(prefix) == 0 and bool(trimmed.start & trimmed.stop)

    # BFS/NFA simulation on the trimmed FSA
    current = set(trimmed.start)
    # Handle epsilon closure at start
    changed = True
    while changed:
        changed = False
        for s in list(current):
            for a, t in trimmed.arcs(s):
                if a == EPSILON and t not in current:
                    current.add(t)
                    changed = True

    for sym in prefix:
        next_states = set()
        for s in current:
            for a, t in trimmed.arcs(s):
                if a == sym:
                    next_states.add(t)
        # Epsilon closure
        changed = True
        while changed:
            changed = False
            for s in list(next_states):
                for a, t in trimmed.arcs(s):
                    if a == EPSILON and t not in next_states:
                        next_states.add(t)
                        changed = True
        current = next_states
        if not current:
            return False

    return len(current) > 0
