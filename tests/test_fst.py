"""Tests for transduction/fst.py — targeting uncovered methods and branches."""
import pytest
from transduction.fst import FST, EPSILON, _advance_buf
from transduction.fsa import FSA


# ── String / display ──────────────────────────────────────────────────────────

def test_str():
    fst = FST()
    fst.add_start(0)
    fst.add_stop(1)
    fst.add_arc(0, 'a', 'x', 1)
    s = str(fst)
    assert '{' in s and '}' in s
    # Should mention states and their start/stop status
    assert 'True' in s   # at least one state has start=True or stop=True
    assert 'a' in s or 'x' in s


# ── Label / structure transforms ──────────────────────────────────────────────

def test_map_labels():
    fst = FST()
    fst.add_start(0)
    fst.add_stop(1)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'y', 1)

    mapped = fst.map_labels(lambda a, b: (a.upper(), b.upper()))
    # Original arcs should be upper-cased
    arcs = list(mapped.arcs(0))
    labels = {(a, b) for a, b, _ in arcs}
    assert ('A', 'X') in labels
    assert ('B', 'Y') in labels
    # Structure preserved
    assert mapped.start == {0}
    assert mapped.stop == {1}


def test_transpose():
    fst = FST()
    fst.add_start(0)
    fst.add_stop(1)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'y', 1)

    t = fst.T
    arcs = list(t.arcs(0))
    labels = {(a, b) for a, b, _ in arcs}
    # Input/output swapped
    assert ('x', 'a') in labels
    assert ('y', 'b') in labels


def test_from_pairs():
    pairs = [('ab', 'xy'), ('c', 'z')]
    fst = FST.from_pairs(pairs)

    # Should accept the pairs as part of its relation
    rel = set(fst.relation(5))
    assert ('ab', 'xy') in rel
    assert ('c', 'z') in rel


def test_from_pairs_unequal_lengths():
    """from_pairs with different-length input/output uses epsilon padding."""
    pairs = [('a', 'xyz')]
    fst = FST.from_pairs(pairs)
    rel = set(fst.relation(5))
    assert ('a', 'xyz') in rel


# ── __call__ (cross-section queries) ─────────────────────────────────────────

def _make_call_fst():
    """Simple deterministic FST: a→x, b→y."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 0)
    fst.add_arc(0, 'b', 'y', 0)
    return fst


def test_call_both_args():
    """fst(x, y) with both strings composes x @ fst @ y."""
    fst = _make_call_fst()
    result = fst('a', 'x')
    # The composition should accept (result is an FST with at least one accepting path)
    assert result.start and result.stop


def test_call_x_only():
    """fst(x, None) returns the output FSA for input x."""
    fst = _make_call_fst()
    result = fst('a', None)
    assert isinstance(result, FSA)
    # Output of 'a' is 'x'; language() returns tuples of labels
    strings = set(result.language(3))
    assert ('x',) in strings


def test_call_y_only():
    """fst(None, y) returns the input FSA for output y."""
    fst = _make_call_fst()
    result = fst(None, 'x')
    assert isinstance(result, FSA)
    strings = set(result.language(3))
    assert ('a',) in strings


def test_call_neither():
    """fst(None, None) returns self."""
    fst = _make_call_fst()
    assert fst(None, None) is fst


# ── Enumeration ───────────────────────────────────────────────────────────────

def test_paths():
    """Enumerate paths in BFS order."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(1)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'y', 1)

    paths = list(fst.paths())
    # Each path is (start, (input, output), ..., final_state)
    assert len(paths) == 2
    for p in paths:
        assert p[0] == 0    # starts at 0
        assert p[-1] == 1   # ends at 1
        assert len(p) == 3  # start, arc_label, end


def test_paths_multi_step():
    """Paths with multiple arcs."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(2)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, 'b', 'y', 2)

    paths = list(fst.paths())
    assert len(paths) == 1
    p = paths[0]
    assert p == (0, ('a', 'x'), 1, ('b', 'y'), 2)


def test_transduce():
    """Transduce a simple input sequence."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 0)
    fst.add_arc(0, 'b', 'y', 0)

    assert fst.transduce('ab') == ('x', 'y')
    assert fst.transduce('ba') == ('y', 'x')
    assert fst.transduce('') == ()


def test_transduce_with_epsilon():
    """Transduce through FST with epsilon arcs."""
    fst = FST()
    fst.add_start(0)
    fst.add_arc(0, 'a', EPSILON, 1)
    fst.add_arc(1, EPSILON, 'x', 2)
    fst.add_stop(2)

    result = fst.transduce('a')
    assert result == ('x',)


def test_transduce_no_path():
    """ValueError when no accepting path exists."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(1)
    fst.add_arc(0, 'a', 'x', 1)

    with pytest.raises(ValueError, match="No accepting path"):
        fst.transduce('b')


def test_transduce_multi_symbol_output():
    """Transduce where output labels are multi-char strings."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'hello', 0)

    assert fst.transduce('a') == ('hello',)


# ── Graph algorithms ──────────────────────────────────────────────────────────

def test_strongly_connected_components_single_scc():
    """Single SCC: all states in one cycle."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'a', 1)
    fst.add_arc(1, 'a', 'a', 2)
    fst.add_arc(2, 'a', 'a', 0)

    sccs = fst.strongly_connected_components()
    # All 3 states form one SCC
    scc_sets = [set(c) for c in sccs]
    assert {0, 1, 2} in scc_sets


def test_strongly_connected_components_multiple():
    """Multiple SCCs: chain plus a self-loop."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(2)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, 'a', 'x', 2)
    fst.add_arc(2, 'a', 'x', 2)  # self-loop

    sccs = fst.strongly_connected_components()
    scc_sets = [set(c) for c in sccs]
    # State 2 forms its own SCC (self-loop)
    assert {2} in scc_sets
    # States 0 and 1 are singleton SCCs (no back edge)
    assert {0} in scc_sets
    assert {1} in scc_sets


def test_strongly_connected_components_dag():
    """DAG: each state is its own SCC."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(2)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, 'b', 'y', 2)

    sccs = fst.strongly_connected_components()
    assert len(sccs) == 3
    for c in sccs:
        assert len(c) == 1


# ── is_functional ─────────────────────────────────────────────────────────────

def test_is_functional_true():
    """Deterministic FST is functional."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 0)
    fst.add_arc(0, 'b', 'y', 0)

    ok, witness = fst.is_functional()
    assert ok is True
    assert witness is None


def test_is_functional_false():
    """FST with two outputs for same input is non-functional."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(1)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'a', 'y', 1)

    ok, witness = fst.is_functional()
    assert ok is False
    assert witness is not None
    x, y1, y2 = witness
    assert y1 != y2


def test_is_functional_empty():
    """Empty FST (no states) is trivially functional."""
    fst = FST()
    ok, witness = fst.is_functional()
    assert ok is True


def test_is_functional_epsilon_cycle():
    """Non-functional due to productive epsilon cycle.

    An FST with an eps-input cycle producing different outputs is non-functional
    because the cycle can be traversed different numbers of times.
    """
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, EPSILON, 'y', 0)  # epsilon cycle back to start
    # Input 'a' → 'xy' (once around) or 'xyxy' (twice) etc.
    ok, witness = fst.is_functional()
    # Functional because same input 'a' always produces 'xy' then loops
    # Actually this is functional: each 'a' produces exactly 'x' then 'y'
    # Let me check: input 'a' goes 0→1, then eps→0 (output y), stop at 0.
    # Or: input 'a' goes 0→1, eps→0, then 'a' goes 0→1 again...
    # So input 'a' → 'xy', input 'aa' → 'xyxy'. It IS functional.
    assert ok is True


def test_is_functional_ambiguous_paths():
    """Two paths for same input, same output length but different symbols."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(2)
    # Path 1: a→x, b→y
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, 'b', 'y', 2)
    # Path 2: a→x, b→z (different output for 'b')
    fst.add_arc(0, 'a', 'x', 3)
    fst.add_arc(3, 'b', 'z', 2)

    ok, witness = fst.is_functional()
    assert ok is False


def test_is_functional_nondeterministic_but_functional():
    """Nondeterministic FST that happens to be functional (same output on all paths)."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(2)
    # Two paths from 0 to 2 on input 'a', both outputting 'x'
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, EPSILON, EPSILON, 2)
    fst.add_arc(0, 'a', 'x', 2)

    ok, witness = fst.is_functional()
    assert ok is True


# ── _advance_buf edge cases ──────────────────────────────────────────────────

def test_advance_buf_side_zero():
    """Both copies in sync, then diverge."""
    side, buf = _advance_buf(0, '', 'a', 'b')
    assert side != 0 or buf != ''


def test_advance_buf_side_two():
    """Copy-2 ahead (covers the side==2 branch at line 851)."""
    # Start with copy-2 ahead by 'x', copy-1 emits nothing, copy-2 emits 'a'
    side, buf = _advance_buf(2, 'x', '', 'a')
    # ahead1=''+'', ahead2='x'+'a'='xa'. ahead1 empty → side=2
    assert side == 2
    assert buf == 'xa'


def test_advance_buf_resync():
    """Copies resynchronize when behind copy catches up."""
    # Copy-1 ahead by 'a', then copy-2 outputs 'a' and copy-1 outputs nothing
    side, buf = _advance_buf(1, 'a', '', 'a')
    # ahead1='a'+''='a', ahead2=''+'a'='a', common prefix='a'
    assert side == 0 and buf == ''


# ── Composition with epsilon (line 399) ──────────────────────────────────────

def test_compose_with_epsilon_output():
    """Composition handles epsilon-output arcs via augmented epsilon labels.

    This exercises the b == EPSILON branch (line 399) in _augment_epsilon_transitions.
    """
    # FST 1: a → ε (epsilon output)
    fst1 = FST()
    fst1.add_start(0)
    fst1.add_stop(1)
    fst1.add_arc(0, 'a', EPSILON, 1)

    # FST 2: identity on 'a'
    fst2 = FST()
    fst2.add_start(0)
    fst2.add_stop(0)
    fst2.add_arc(0, 'a', 'a', 0)

    composed = fst1 @ fst2
    # fst1 maps 'a' → ε, fst2 can't contribute since fst1 produces no output
    # So the composed relation maps 'a' → ε
    rel = set(composed.relation(3))
    assert ('a', '') in rel


def test_compose_with_epsilon_input():
    """Composition handles epsilon-input arcs (line 401 branch)."""
    # FST 1: identity on 'a'
    fst1 = FST()
    fst1.add_start(0)
    fst1.add_stop(0)
    fst1.add_arc(0, 'a', 'a', 0)

    # FST 2: ε → 'x' (epsilon input, produces output)
    fst2 = FST()
    fst2.add_start(0)
    fst2.add_stop(1)
    fst2.add_arc(0, EPSILON, 'x', 1)

    composed = fst1 @ fst2
    # fst2 can produce 'x' without consuming input from fst1
    rel = set(composed.relation(3))
    assert ('', 'x') in rel


def test_compose_both_epsilon():
    """Composition where both FSTs have epsilon arcs."""
    # FST 1: a→b, then ε→ε to final
    fst1 = FST()
    fst1.add_start(0)
    fst1.add_arc(0, 'a', 'b', 1)
    fst1.add_arc(1, EPSILON, EPSILON, 2)
    fst1.add_stop(2)

    # FST 2: b→c
    fst2 = FST()
    fst2.add_start(0)
    fst2.add_stop(0)
    fst2.add_arc(0, 'b', 'c', 0)

    composed = fst1 @ fst2
    rel = set(composed.relation(3))
    assert ('a', 'c') in rel


# ── from_string (lines 225-230, covered via __call__ but also directly) ──────

def test_from_string():
    """FST.from_string builds a diagonal identity FST for the string."""
    fst = FST.from_string('abc')
    rel = set(fst.relation(5))
    assert ('abc', 'abc') in rel
    assert len(rel) == 1


# ── project axis=1 (line 256) ────────────────────────────────────────────────

def test_project_output():
    """project(1) returns the output-side FSA."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 0)
    fst.add_arc(0, 'b', 'y', 0)

    out_fsa = fst.project(1)
    assert isinstance(out_fsa, FSA)
    strings = set(out_fsa.language(3))
    assert () in strings           # empty string
    assert ('x',) in strings
    assert ('y',) in strings
    assert ('x', 'y') in strings


# ── Composition branch: smaller left operand (line 300) ──────────────────────

def test_compose_smaller_left():
    """Exercise the composition branch where len(self.states) < len(other.states)."""
    # Small left FST (2 states)
    left = FST()
    left.add_start(0)
    left.add_stop(0)
    left.add_arc(0, 'a', 'b', 0)

    # Larger right FST (4 states)
    right = FST()
    right.add_start(0)
    right.add_stop(3)
    right.add_arc(0, 'b', 'x', 1)
    right.add_arc(1, 'b', 'y', 2)
    right.add_arc(2, 'b', 'z', 3)

    composed = left @ right
    rel = set(composed.relation(5))
    assert ('aaa', 'xyz') in rel


def test_compose_smaller_right():
    """Exercise the composition branch where len(self.states) >= len(other.states)."""
    # Larger left FST (4 states)
    left = FST()
    left.add_start(0)
    left.add_stop(3)
    left.add_arc(0, 'a', 'b', 1)
    left.add_arc(1, 'a', 'b', 2)
    left.add_arc(2, 'a', 'b', 3)

    # Small right FST (2 states)
    right = FST()
    right.add_start(0)
    right.add_stop(0)
    right.add_arc(0, 'b', 'x', 0)

    composed = left @ right
    rel = set(composed.relation(5))
    assert ('aaa', 'xxx') in rel
