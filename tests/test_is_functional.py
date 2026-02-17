"""Tests for FST.is_functional()."""

import pytest
from transduction import FST, EPSILON
from transduction import examples


class TestIsFunctional:

    def test_simple_functional(self):
        f = FST()
        f.add_start(0); f.add_stop(1)
        f.add_arc(0, 'a', 'x', 1)
        f.add_arc(0, 'b', 'y', 1)
        assert f.is_functional() == (True, None)

    def test_simple_relation(self):
        r = FST()
        r.add_start(0); r.add_stop(1)
        r.add_arc(0, 'a', 'x', 1)
        r.add_arc(0, 'a', 'y', 1)
        ok, witness = r.is_functional()
        assert not ok
        x, y1, y2 = witness
        assert x == ('a',)
        assert {y1, y2} == {('x',), ('y',)}

    def test_identity_cyclic(self):
        """Identity on {a,b}* — functional with infinite input language."""
        d = FST()
        d.add_start(0); d.add_stop(0)
        d.add_arc(0, 'a', 'a', 0)
        d.add_arc(0, 'b', 'b', 0)
        assert d.is_functional() == (True, None)

    def test_epsilon_relation(self):
        """Non-functional via epsilon-output arcs."""
        r = FST()
        r.add_start(0); r.add_stop(2); r.add_stop(3)
        r.add_arc(0, 'a', 'x', 1)
        r.add_arc(1, EPSILON, 'y', 2)
        r.add_arc(1, EPSILON, 'z', 3)
        ok, witness = r.is_functional()
        assert not ok
        x, y1, y2 = witness
        assert x == ('a',)
        assert {y1, y2} == {('x', 'y'), ('x', 'z')}

    def test_nondeterministic_but_functional(self):
        """Multiple paths, same output — still functional."""
        nd = FST()
        nd.add_start(0); nd.add_stop(3)
        nd.add_arc(0, 'a', 'x', 1)
        nd.add_arc(0, 'a', 'x', 2)
        nd.add_arc(1, 'b', 'y', 3)
        nd.add_arc(2, 'b', 'y', 3)
        assert nd.is_functional() == (True, None)

    def test_cyclic_relation(self):
        """Cycle that pumps the output delay — terminates without max_length."""
        c = FST()
        c.add_start(0); c.add_stop(0)
        c.add_arc(0, 'a', 'x', 0)
        c.add_arc(0, 'a', 'xx', 0)
        ok, witness = c.is_functional()
        assert not ok

    def test_empty_fst(self):
        assert FST().is_functional() == (True, None)

    def test_no_accepting_paths(self):
        """Start state with arcs but no final states."""
        f = FST()
        f.add_start(0)
        f.add_arc(0, 'a', 'x', 1)
        assert f.is_functional() == (True, None)

    def test_epsilon_only_relation(self):
        """Empty input string maps to two different outputs."""
        r = FST()
        r.add_start(0); r.add_stop(1); r.add_stop(2)
        r.add_arc(0, EPSILON, 'a', 1)
        r.add_arc(0, EPSILON, 'b', 2)
        ok, witness = r.is_functional()
        assert not ok
        x, y1, y2 = witness
        assert x == ()
        assert {y1, y2} == {('a',), ('b',)}

    def test_longer_divergence(self):
        """Outputs agree on a prefix then diverge."""
        r = FST()
        r.add_start(0); r.add_stop(2); r.add_stop(3)
        r.add_arc(0, 'a', 'x', 1)
        r.add_arc(1, 'b', 'y', 2)
        r.add_arc(1, 'b', 'z', 3)
        ok, witness = r.is_functional()
        assert not ok
        x, y1, y2 = witness
        assert x == ('a', 'b')
        assert {y1, y2} == {('x', 'y'), ('x', 'z')}

    def test_multiple_start_states(self):
        """Two start states, same behavior — functional."""
        f = FST()
        f.add_start(0); f.add_start(1); f.add_stop(2)
        f.add_arc(0, 'a', 'x', 2)
        f.add_arc(1, 'a', 'x', 2)
        assert f.is_functional() == (True, None)

    def test_multiple_start_states_relation(self):
        """Two start states with different outputs on same input."""
        r = FST()
        r.add_start(0); r.add_start(1); r.add_stop(2)
        r.add_arc(0, 'a', 'x', 2)
        r.add_arc(1, 'a', 'y', 2)
        ok, witness = r.is_functional()
        assert not ok
        x, y1, y2 = witness
        assert x == ('a',)
        assert {y1, y2} == {('x',), ('y',)}


class TestIsFunctionalExamples:
    """Test against example FSTs from the codebase."""

    @pytest.mark.parametrize('name', [
        'small', 'delete_b', 'togglecase', 'triplets_of_doom',
        'weird_copy', 'lookahead', 'lowercase',
        'gated_universal', 'complementary_halves',
        'shrinking_nonuniversal', 'scaled_newspeak',
        'layered_witnesses', 'samuel_example',
    ])
    def test_known_functional(self, name):
        fst = getattr(examples, name)()
        assert fst.is_functional() == (True, None)

    def test_mystery6_is_relation(self):
        fst = examples.mystery6()
        ok, witness = fst.is_functional()
        assert not ok
