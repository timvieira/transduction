"""Tests for LazyPrecoverDFA: verify equivalence with PrecoverNFA(...).det().

For each (FST, target) pair, we fully materialize both the reference DFA
(via PrecoverNFA.det()) and the optimized LazyPrecoverDFA, then check that
they accept exactly the same language (same set of source strings).
"""

import pytest
from collections import deque

from transduction.fst import FST, EPSILON
from transduction.precover_nfa import PrecoverNFA
from transduction.lazy_precover_dfa import LazyPrecoverDFA, PowersetArena, PackedPrecoverNFA
from transduction import examples


# ── Helpers ────────────────────────────────────────────────────────

def accepted_strings(dfa, alphabet, max_len=6):
    """Enumerate all strings up to max_len accepted by a Lazy DFA."""
    accepted = set()
    [start] = list(dfa.start())

    # BFS over (state, string) pairs
    worklist = deque([(start, ())])
    visited = {(start, ())}

    while worklist:
        state, path = worklist.popleft()

        if dfa.is_final(state):
            accepted.add(path)

        if len(path) >= max_len:
            continue

        for x in alphabet:
            successors = list(dfa.arcs_x(state, x))
            for dest in successors:
                key = (dest, path + (x,))
                if key not in visited:
                    visited.add(key)
                    worklist.append((dest, path + (x,)))

    return accepted


def check_equivalence(fst, target, max_len=6):
    """Assert that LazyPrecoverDFA and PrecoverNFA.det() accept the same strings."""
    alphabet = fst.A - {EPSILON}

    # Reference: existing PrecoverNFA + LazyDeterminize
    ref_dfa = PrecoverNFA(fst, target).det()
    ref_strings = accepted_strings(ref_dfa, alphabet, max_len)

    # Optimized: LazyPrecoverDFA
    opt_dfa = LazyPrecoverDFA(fst, target)
    opt_strings = accepted_strings(opt_dfa, alphabet, max_len)

    assert ref_strings == opt_strings, (
        f"Mismatch for target={target!r}:\n"
        f"  ref_only={ref_strings - opt_strings}\n"
        f"  opt_only={opt_strings - ref_strings}"
    )

    return opt_dfa  # return for further inspection


# ── PowersetArena unit tests ──────────────────────────────────────

class TestPowersetArena:

    def test_intern_returns_stable_ids(self):
        arena = PowersetArena()
        id0 = arena.intern((1, 2, 3), False)
        id1 = arena.intern((4, 5), True)
        id2 = arena.intern((1, 2, 3), False)
        assert id0 == 0
        assert id1 == 1
        assert id2 == 0  # same set → same ID

    def test_singleton_fast_path(self):
        arena = PowersetArena()
        id0 = arena.intern((42,), True)
        id1 = arena.intern((42,), False)
        assert id0 == id1  # same singleton → same ID
        assert not arena.is_final[id0]  # updated to False on re-intern

    def test_finality_updated_on_hit(self):
        arena = PowersetArena()
        # Multi-element: non-final → final
        id0 = arena.intern((1, 2, 3), False)
        assert not arena.is_final[id0]
        arena.intern((1, 2, 3), True)
        assert arena.is_final[id0]

        # Singleton: final → non-final
        id1 = arena.intern((99,), True)
        assert arena.is_final[id1]
        arena.intern((99,), False)
        assert not arena.is_final[id1]

    def test_lookup(self):
        arena = PowersetArena()
        arena.intern((10, 20), False)
        arena.intern((30,), True)
        assert arena.lookup((10, 20)) == 0
        assert arena.lookup((30,)) == 1
        assert arena.lookup((999,)) is None
        assert arena.lookup((10, 20, 30)) is None

    def test_len(self):
        arena = PowersetArena()
        assert len(arena) == 0
        arena.intern((1,), False)
        assert len(arena) == 1
        arena.intern((2, 3), True)
        assert len(arena) == 2
        arena.intern((1,), True)  # re-intern, no new state
        assert len(arena) == 2


# ── PackedPrecoverNFA unit tests ──────────────────────────────────

class TestPackedPrecoverNFA:

    def test_pack_unpack_roundtrip(self):
        fst = examples.small()
        nfa = PackedPrecoverNFA(fst, ('x',))
        for s in range(10):
            for p in range(nfa.target_len + 1):
                packed = nfa.pack(s, p)
                assert nfa.unpack(packed) == (s, p)

    def test_start_states(self):
        fst = examples.small()
        nfa = PackedPrecoverNFA(fst, ('x',))
        starts = nfa.start_states()
        # Each start state should have buf_pos=0
        for s in starts:
            fst_state, buf_pos = nfa.unpack(s)
            assert buf_pos == 0
            assert fst_state in fst.start

    def test_eps_closure_caching(self):
        fst = examples.samuel_example()  # has epsilon arcs
        nfa = PackedPrecoverNFA(fst, ('c',))
        starts = nfa.start_states()

        # First call: miss
        for s in starts:
            nfa.eps_closure_single(s)
        h1, m1 = nfa.eps_cache_stats()
        assert m1 > 0

        # Second call: hit
        for s in starts:
            nfa.eps_closure_single(s)
        h2, m2 = nfa.eps_cache_stats()
        assert h2 > h1  # got cache hits
        assert m2 == m1  # no new misses


# ── LazyPrecoverDFA equivalence tests ─────────────────────────────

class TestLazyPrecoverDFA:

    def test_empty_target(self):
        """Empty target: precover is the set of all strings (FST accepts ε output)."""
        fst = examples.small()
        dfa = check_equivalence(fst, ())
        # Start state should be final (empty target is always a preimage of final FST states)
        assert dfa.is_final(dfa._start_id)

    def test_small_fst(self):
        fst = examples.small()
        check_equivalence(fst, ('x',))
        check_equivalence(fst, ('x', 'a'))
        check_equivalence(fst, ('x', 'b', 'a'))

    def test_weird_copy(self):
        fst = examples.weird_copy()
        check_equivalence(fst, ())
        check_equivalence(fst, ('a',))
        check_equivalence(fst, ('b',))
        check_equivalence(fst, ('a', 'b'))
        check_equivalence(fst, ('b', 'a', 'b'))

    def test_lookahead(self):
        fst = examples.lookahead()
        check_equivalence(fst, ('x',))
        check_equivalence(fst, ('x', 'x'))
        check_equivalence(fst, ('x', 'a'))
        check_equivalence(fst, ('x', 'a', 'b'))

    def test_triplets_of_doom(self):
        fst = examples.triplets_of_doom()
        check_equivalence(fst, ('a',))
        check_equivalence(fst, ('a', 'a', 'a'))
        check_equivalence(fst, ('b', 'b', 'b'))
        check_equivalence(fst, ('a', 'a', 'a', 'b', 'b', 'b'))

    def test_samuel_example(self):
        """Has epsilon arcs — exercises epsilon closure."""
        fst = examples.samuel_example()
        check_equivalence(fst, ('c',))
        check_equivalence(fst, ('c', 'x'))
        check_equivalence(fst, ('c', 'x', 'x'))
        check_equivalence(fst, ('y',))

    def test_delete_b(self):
        """Infinite quotient FST — tests boundary phase behavior."""
        fst = examples.delete_b()
        check_equivalence(fst, ())
        check_equivalence(fst, ('A',))
        check_equivalence(fst, ('A', 'A'))

    def test_newspeak(self):
        """Multi-state replacement FST with epsilon chains."""
        fst = examples.newspeak2()
        # newspeak has a 26-letter alphabet — keep max_len small to avoid
        # reference DFA timeout.
        check_equivalence(fst, ('a',), max_len=3)
        check_equivalence(fst, ('b',), max_len=3)
        check_equivalence(fst, ('u', 'n'), max_len=3)

    def test_mystery1(self):
        fst = examples.mystery1()
        check_equivalence(fst, ('c',))
        check_equivalence(fst, ('c', 'x'))

    def test_mystery2(self):
        fst = examples.mystery2()
        check_equivalence(fst, ('c',))
        check_equivalence(fst, ('c', 'x'))

    def test_sdd1(self):
        """Tricky FST with ε:ε self-loops."""
        fst = examples.sdd1_fst()
        check_equivalence(fst, ('a',))
        check_equivalence(fst, ('a', 'a'))
        check_equivalence(fst, ('a', 'b'))

    def test_replace(self):
        fst = examples.replace([('a', 'x'), ('b', 'y'), ('c', 'z')])
        check_equivalence(fst, ('x',))
        check_equivalence(fst, ('x', 'y'))
        check_equivalence(fst, ('x', 'y', 'z'))

    def test_bpe_like(self):
        """BPE-style trie FST — the main use case for these optimizations."""
        fst = examples.bpe_like(vocab_size=10, alphabet=tuple("ab"), max_len=2)
        # BPE FSTs use integer token IDs as input symbols — large alphabet.
        # Keep max_len small to avoid reference DFA timeout.
        check_equivalence(fst, ('a',), max_len=4)
        check_equivalence(fst, ('a', 'b'), max_len=4)
        check_equivalence(fst, ('a', 'b', 'a'), max_len=4)

    def test_parity(self):
        """Epsilon-output-only FST."""
        fst = examples.parity(('a', 'b'))
        check_equivalence(fst, ('0',))
        check_equivalence(fst, ('1',))

    def test_anbn(self):
        fst = examples.anbn()
        check_equivalence(fst, ('b',))
        check_equivalence(fst, ('c',))
        check_equivalence(fst, ('b', 'b', 'b'))

    def test_run_method(self):
        """Test the run() traversal method."""
        fst = examples.small()
        target = ('x',)
        dfa = LazyPrecoverDFA(fst, target)

        # 'a' maps to output 'x', so path ('a',) should reach a final state
        result = dfa.run(('a',))
        assert result is not None
        assert dfa.is_final(result)

        # 'b' also maps to 'x'
        result = dfa.run(('b',))
        assert result is not None

        # Nonexistent symbol should return None
        result = dfa.run(('z',))
        assert result is None

    def test_stats(self):
        """Stats method should return sensible values."""
        fst = examples.bpe_like(vocab_size=20, alphabet=tuple("ab"), max_len=3)
        target = ('a', 'b')
        dfa = LazyPrecoverDFA(fst, target)

        # Force full expansion by materializing
        dfa.materialize()

        stats = dfa.stats()
        assert stats['num_dfa_states'] > 0
        assert stats['num_expanded'] > 0
        assert 0 <= stats['singleton_fraction'] <= 1
        assert stats['eps_cache_size'] >= 0

    def test_nfa_states_and_unpack(self):
        """Introspection methods should work correctly."""
        fst = examples.small()
        dfa = LazyPrecoverDFA(fst, ('x',))

        nfa_set = dfa.nfa_states(dfa._start_id)
        assert isinstance(nfa_set, tuple)
        assert len(nfa_set) > 0

        for packed in nfa_set:
            fst_state, buf_pos = dfa.unpack_nfa_state(packed)
            assert isinstance(fst_state, int)
            assert 0 <= buf_pos <= len(dfa.target)
