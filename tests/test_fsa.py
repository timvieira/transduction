"""Tests for FSA with non-str alphabet types (int, bytes, tuple).

Exercises the full FSA algebra — construction, language enumeration,
determinization, minimization, Boolean operations, quotients, and
map_labels — on alphabets beyond str, confirming the Generic[A]
parameterization works end-to-end.
"""

from transduction.fsa import FSA, EPSILON


# --- Helpers ---

def int_fsa_ab() -> FSA[int]:
    """FSA over {1, 2} accepting sequences (1, 2) and (1, 1, 2)."""
    m: FSA[int] = FSA()
    m.add_start('s')
    m.add('s', 1, 'a')
    m.add('a', 2, 'f1')
    m.add('a', 1, 'b')
    m.add('b', 2, 'f2')
    m.add_stop('f1')
    m.add_stop('f2')
    return m


def int_fsa_c() -> FSA[int]:
    """FSA over {1, 2, 3} accepting (1, 3) and (2, 3)."""
    m: FSA[int] = FSA()
    m.add_start('s')
    m.add('s', 1, 'a')
    m.add('s', 2, 'a')
    m.add('a', 3, 'f')
    m.add_stop('f')
    return m


def bytes_fsa() -> FSA[bytes]:
    """FSA over {b'a', b'b'} accepting (b'a', b'b')."""
    m: FSA[bytes] = FSA()
    m.add_start(0)
    m.add(0, b'a', 1)
    m.add(1, b'b', 2)
    m.add_stop(2)
    return m


def bytes_fsa_2() -> FSA[bytes]:
    """FSA over {b'a', b'b'} accepting (b'a',) and (b'b',)."""
    m: FSA[bytes] = FSA()
    m.add_start(0)
    m.add(0, b'a', 1)
    m.add(0, b'b', 2)
    m.add_stop(1)
    m.add_stop(2)
    return m


# --- Basic construction & language enumeration ---

def test_int_alphabet_construction() -> None:
    m = int_fsa_ab()
    assert m.syms == {1, 2}
    assert len(m.start) == 1
    assert len(m.stop) == 2


def test_int_alphabet_language() -> None:
    m = int_fsa_ab()
    lang = set(m.language())
    assert lang == {(1, 2), (1, 1, 2)}


def test_bytes_alphabet_construction() -> None:
    m = bytes_fsa()
    assert m.syms == {b'a', b'b'}


def test_bytes_alphabet_language() -> None:
    m = bytes_fsa()
    lang = set(m.language())
    assert lang == {(b'a', b'b')}


def test_tuple_alphabet_construction() -> None:
    """FSA over tuple-of-str-int labels."""
    m: FSA[tuple[str, int]] = FSA()
    m.add_start(0)
    m.add(0, ('x', 1), 1)
    m.add(1, ('y', 2), 2)
    m.add_stop(2)
    assert m.syms == {('x', 1), ('y', 2)}
    lang = set(m.language())
    assert lang == {(('x', 1), ('y', 2))}


# --- Core operations on int-alphabet FSAs ---

def test_int_determinize() -> None:
    m = int_fsa_ab()
    d = m.det().trim()
    assert set(d.language()) == set(m.language())
    # DFA: each state has at most one transition per symbol
    for s in d.states:
        for sym in d.syms:
            assert len(d.edges[s][sym]) <= 1


def test_int_minimize() -> None:
    m = int_fsa_ab()
    mn = m.min()
    assert mn.equal(m)
    # minimized should have <= det states
    assert len(mn.states) <= len(m.det().trim().states)


def test_int_reverse() -> None:
    m = int_fsa_ab()
    r = m.reverse()
    lang = set(r.language())
    assert lang == {(2, 1), (2, 1, 1)}


def test_int_trim() -> None:
    m: FSA[int] = FSA()
    m.add_start(0)
    m.add(0, 1, 1)
    m.add(1, 2, 2)
    m.add_stop(2)
    # add unreachable state
    m.add(10, 3, 11)
    t = m.trim()
    assert 10 not in t.states
    assert 11 not in t.states
    assert set(t.language()) == {(1, 2)}


def test_int_union() -> None:
    a = int_fsa_ab()
    c = int_fsa_c()
    u = a + c
    lang = set(u.language())
    assert lang == {(1, 2), (1, 1, 2), (1, 3), (2, 3)}


def test_int_concatenation() -> None:
    # (1, 2) * (3,) = (1, 2, 3)
    m1: FSA[int] = FSA()
    m1.add_start(0); m1.add(0, 1, 1); m1.add(1, 2, 2); m1.add_stop(2)
    m2: FSA[int] = FSA()
    m2.add_start(0); m2.add(0, 3, 1); m2.add_stop(1)
    cat = m1 * m2
    lang = set(cat.epsremove().language())
    assert lang == {(1, 2, 3)}


def test_int_intersection() -> None:
    a = int_fsa_ab()
    # FSA accepting anything starting with (1,)
    b: FSA[int] = FSA()
    b.add_start(0)
    b.add(0, 1, 1)
    b.add(1, 1, 1)
    b.add(1, 2, 1)
    b.add_stop(1)
    inter = a & b
    lang = set(inter.language())
    assert lang == {(1, 2), (1, 1, 2)}


def test_int_difference() -> None:
    a = int_fsa_ab()
    # FSA accepting just (1, 2)
    b: FSA[int] = FSA()
    b.add_start(0); b.add(0, 1, 1); b.add(1, 2, 2); b.add_stop(2)
    diff = a - b
    lang = set(diff.trim().language())
    assert lang == {(1, 1, 2)}


def test_int_star() -> None:
    m: FSA[int] = FSA()
    m.add_start(0)
    m.add(0, 1, 1)
    m.add_stop(1)
    s = m.star()
    lang = set(s.language(max_length=3))
    assert () in lang           # epsilon from star
    assert (1,) in lang
    assert (1, 1) in lang
    assert (1, 1, 1) in lang


def test_int_complement() -> None:
    m: FSA[int] = FSA()
    m.add_start(0); m.add(0, 1, 1); m.add_stop(1)
    # complement over {1, 2}
    c = m.invert({1, 2})
    # () should be accepted (not in original)
    assert () in c
    # (1,) should not be accepted
    assert (1,) not in c
    # (2,) should be accepted
    assert (2,) in c
    # (1, 1) accepted
    assert (1, 1) in c


def test_int_left_quotient() -> None:
    # L = {(1, 2, 3)}, R = {(1,)} => L // R = {(2, 3)}
    L: FSA[int] = FSA()
    L.add_start(0); L.add(0, 1, 1); L.add(1, 2, 2); L.add(2, 3, 3); L.add_stop(3)
    R: FSA[int] = FSA()
    R.add_start(0); R.add(0, 1, 1); R.add_stop(1)
    q = L // R
    lang = set(q.epsremove().det().trim().language())
    assert lang == {(2, 3)}


def test_int_right_quotient() -> None:
    # L = {(1, 2, 3)}, R = {(3,)} => L / R = {(1, 2)}
    L: FSA[int] = FSA()
    L.add_start(0); L.add(0, 1, 1); L.add(1, 2, 2); L.add(2, 3, 3); L.add_stop(3)
    R: FSA[int] = FSA()
    R.add_start(0); R.add(0, 3, 1); R.add_stop(1)
    q = L / R
    lang = set(q.epsremove().det().trim().language())
    assert lang == {(1, 2)}


def test_int_equal() -> None:
    a = int_fsa_ab()
    b = int_fsa_ab()
    assert a.equal(b)

    c = int_fsa_c()
    assert not a.equal(c)


def test_int_run_and_contains() -> None:
    m = int_fsa_ab()
    assert (1, 2) in m
    assert (1, 1, 2) in m
    assert (1,) not in m
    assert (2,) not in m
    assert (1, 1, 1) not in m

    reached = m.run([1, 2])
    assert reached & m.stop


# --- Cross-type: map_labels ---

def test_map_labels_int_to_str() -> None:
    m = int_fsa_ab()
    mapped = m.map_labels(str)
    lang = set(mapped.language())
    assert lang == {('1', '2'), ('1', '1', '2')}
    assert mapped.syms == {'1', '2'}


def test_map_labels_str_to_bytes() -> None:
    m: FSA[str] = FSA()
    m.add_start(0)
    m.add(0, 'a', 1)
    m.add(1, 'b', 2)
    m.add_stop(2)
    mapped = m.map_labels(lambda s: s.encode())
    lang = set(mapped.language())
    assert lang == {(b'a', b'b')}


# --- bytes alphabet operations ---

def test_bytes_det_min_equal() -> None:
    m = bytes_fsa()
    d = m.det().trim()
    mn = d.min()
    assert mn.equal(m)


def test_bytes_union_intersection() -> None:
    a = bytes_fsa()
    b = bytes_fsa_2()
    u = a + b
    lang = set(u.language())
    assert lang == {(b'a', b'b'), (b'a',), (b'b',)}

    inter = a & b
    lang_i = set(inter.language())
    assert lang_i == set()  # no overlap


# --- Regression: EPSILON handling ---

def test_epsilon_not_in_syms_after_eps_arcs() -> None:
    """EPSILON arcs should not pollute syms."""
    m: FSA[int] = FSA()
    m.add_start(0)
    m.add(0, 1, 1)
    m._add_eps(1, 2)
    m.add(2, 2, 3)
    m.add_stop(3)
    assert EPSILON not in m.syms  # type: ignore[comparison-overlap]
    assert m.syms == {1, 2}
    # language should still work
    lang = set(m.epsremove().language())
    assert lang == {(1, 2)}


# --- from_string / from_strings with non-str sequences ---

def test_from_string_tuple_of_ints() -> None:
    """FSA.from_string works with tuple[int, ...] sequences."""
    m = FSA.from_string((10, 20, 30))
    lang = set(m.language())
    assert lang == {(10, 20, 30)}


def test_from_strings_list_of_tuples() -> None:
    """FSA.from_strings works with list[tuple[int, ...]]."""
    m = FSA.from_strings([(1, 2), (3, 4)])
    lang = set(m.language())
    assert lang == {(1, 2), (3, 4)}


# --- universal ---

def test_universal_int() -> None:
    u = FSA.universal([1, 2, 3])
    assert () in u
    assert (1,) in u
    assert (1, 2, 3, 1) in u
    assert u.syms == {1, 2, 3}


# --- lift ---

def test_lift_int() -> None:
    m = FSA.lift(42)
    lang = set(m.language())
    assert lang == {(42,)}
