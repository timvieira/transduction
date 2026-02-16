"""Tests for FST.make_total()."""

from transduction import FST, EPSILON


def domain(fst, max_length):
    """Return the set of input strings accepted by the FST, up to max_length."""
    return {x for x, _y in fst.relation(max_length)}


def test_partial_becomes_total():
    """A partial FST should accept all of Sigma* after make_total()."""
    # This FST reads 'a' or 'b' but only produces output on 'a';
    # 'b' leads to a dead state.
    f = FST()
    f.add_start(0); f.add_stop(0)
    f.add_arc(0, 'a', 'x', 0)
    f.add_arc(0, 'b', 'y', 1)  # state 1 is a dead end (no arcs out, not final)

    # Before: 'b' is in the alphabet but not in the domain
    assert '' in domain(f, 3)    # empty string accepted (start is final)
    assert 'a' in domain(f, 3)
    assert 'b' not in domain(f, 3)

    g = f.make_total('FAIL')

    # After: 'b' should be accepted
    assert 'b' in domain(g, 3)
    assert 'bb' in domain(g, 3)
    assert 'ab' in domain(g, 3)


def test_total_fst_unchanged():
    """An already-total FST should preserve its relation after make_total()."""
    f = FST()
    f.add_start(0); f.add_stop(0)
    f.add_arc(0, 'a', 'x', 0)
    f.add_arc(0, 'b', 'y', 0)
    # f is already total over {a,b}

    g = f.make_total('FAIL')

    # The original relation should be preserved
    orig = set(f.relation(3))
    new = set(g.relation(3))
    assert orig <= new


def test_marker_appears_in_output():
    """Inputs not in the original domain should produce the FAIL marker."""
    f = FST()
    f.add_start(0); f.add_stop(1)
    f.add_arc(0, 'a', 'x', 1)
    f.add_arc(0, 'b', 'y', 2)  # dead end — 'b' not in domain

    g = f.make_total('FAIL')

    # 'b' should now be accepted, and its output should contain FAIL
    outputs_for_b = {y for x, y in g.relation(3) if x == 'b'}
    assert len(outputs_for_b) > 0
    assert all('FAIL' in y for y in outputs_for_b)

    # 'a' should still map to 'x'
    outputs_for_a = {y for x, y in g.relation(3) if x == 'a'}
    assert 'x' in outputs_for_a
