"""Tests for algs.Incremental against the reference Precover."""

import pytest
import numpy as np
from functools import cached_property
from transduction import examples, EPSILON, Precover, DecompositionResult, FST
from transduction.util import set_memory_limit, LogDistr, logsumexp
from transduction.lm.base import LM, LMState
from transduction.lm.reference_transduced import ReferenceTransducedLM, BruteForceTransducedLM
from algs import Incremental

set_memory_limit(4)


def _factory(fst):
    """Wrap Incremental in the factory-style API used by test_finite.py."""
    inc = Incremental(fst)
    def call(target):
        target = tuple(target)
        R, Q = inc.decompose(target)
        return DecompositionResult(Q, R)
    return call


def _precover_factory(fst):
    return Precover.factory(fst)


def _to_fsa(x):
    """Coerce sets/frozensets to FSA; pass FSA instances through."""
    from transduction.fsa import FSA
    if isinstance(x, (set, frozenset)):
        return FSA.from_strings(x)
    return x


def _assert_equal(have, want):
    hq, hr = _to_fsa(have.quotient), _to_fsa(have.remainder)
    wq, wr = _to_fsa(want.quotient), _to_fsa(want.remainder)
    assert hq.equal(wq), f"quotient mismatch: {hq.min()} != {wq.min()}"
    assert hr.equal(wr), f"remainder mismatch: {hr.min()} != {wr.min()}"


def _run_test(fst, depth):
    """Compare Incremental against Precover for all target prefixes up to depth."""
    factory = _factory(fst)
    reference = _precover_factory(fst)
    target_alphabet = fst.B - {EPSILON}

    def recurse(target, depth):
        if depth == 0:
            return
        _assert_equal(factory(target), reference(target))
        for y in target_alphabet:
            ref_child = reference(target + y)
            q, r = _to_fsa(ref_child.quotient), _to_fsa(ref_child.remainder)
            if q.trim().states or r.trim().states:
                recurse(target + y, depth - 1)

    recurse('', depth)


# ── Tests adapted from test_finite.py ──────────────────────────────────────

def test_sdd1():
    fst = examples.sdd1_fst()
    tmp = _factory(fst)
    _assert_equal(tmp(''), DecompositionResult({'a'}, set()))
    _assert_equal(tmp('a'), DecompositionResult({'a'}, set()))
    _assert_equal(tmp('aa'), DecompositionResult({'aa'}, set()))


def test_simple():
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])
    tmp = _factory(fst)
    _assert_equal(tmp(''), DecompositionResult({''}, set()))
    _assert_equal(tmp('a'), DecompositionResult({'1'}, set()))
    _assert_equal(tmp('ab'), DecompositionResult({'12'}, set()))
    _assert_equal(tmp('abc'), DecompositionResult({'123'}, set()))


def test_duplicate():
    fst = examples.duplicate(set('12345'))
    tmp = _factory(fst)
    _assert_equal(tmp(''), DecompositionResult({''}, set()))
    _assert_equal(tmp('1'), DecompositionResult({'1'}, set()))
    _assert_equal(tmp('11'), DecompositionResult({'1'}, set()))
    _assert_equal(tmp('1155'), DecompositionResult({'15'}, set()))
    _assert_equal(tmp('115'), DecompositionResult({'15'}, set()))


def test_samuel_example():
    fst = examples.samuel_example()
    tmp = _factory(fst)
    _assert_equal(tmp('c'), DecompositionResult({'a'}, set()))


def test_anbn():
    fst = examples.anbn()
    tmp = _factory(fst)
    _assert_equal(tmp('b'), DecompositionResult({'aaa'}, {'a'}))
    _assert_equal(tmp('c'), DecompositionResult(set(), {'aa'}))
    _assert_equal(tmp('bb'), DecompositionResult({'aaa'}, set()))


def test_backticks_to_quote():
    fst = examples.backticks_to_quote()
    tmp = _factory(fst)
    _assert_equal(tmp('b'), DecompositionResult({'a'}, set()))
    _assert_equal(tmp('`'), DecompositionResult({'`a'}, {'`'}))
    _assert_equal(tmp('"'), DecompositionResult({'``'}, set()))
    _assert_equal(tmp('`b'), DecompositionResult({'`a'}, set()))


def test_togglecase():
    _run_test(examples.togglecase(), depth=1)


def test_lowercase():
    _run_test(examples.lowercase(), depth=1)


def test_mystery1():
    _run_test(examples.mystery1(), depth=5)


def test_small():
    _run_test(examples.small(), depth=3)


def test_lookahead():
    _run_test(examples.lookahead(), depth=3)


def test_diamond_with_selfloops():
    """Diamond FST with epsilon output and self-loops on the accept state.
    q0(start,stop) --a:ε--> q1 --b:c--> q3(stop, a:c/b:c self-loops)
    q0 --a:c--> q2(stop) --a:c--> q3
    Source 'a' and 'ab' both map to 'c', creating ambiguity.
    """
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_stop(2)
    fst.add_stop(3)
    fst.add_arc(0, 'a', EPSILON, 1)
    fst.add_arc(0, 'a', 'c', 2)
    fst.add_arc(1, 'b', 'c', 3)
    fst.add_arc(2, 'a', 'c', 3)
    fst.add_arc(3, 'a', 'c', 3)
    fst.add_arc(3, 'b', 'c', 3)
    _run_test(fst, depth=4)


# ── Low-level method tests ────────────────────────────────────────────────

def test_run_empty_input():
    fst = examples.small()
    inc = Incremental(fst)
    result = inc.run((), ('x',))
    assert all(len(ys) == 0 for (_, ys) in result)


def test_is_member():
    fst = examples.small()
    inc = Incremental(fst)
    assert inc.is_member(('a',), ('x',))
    assert not inc.is_member(('b',), ('x',))


def test_oov_target_symbol():
    fst = examples.replace([('1', 'a'), ('2', 'b')])
    inc = Incremental(fst)
    with pytest.raises(ValueError, match="Out of vocabulary"):
        inc.decompose(('z',))


# ── logp_prefix tests ────────────────────────────────────────────────────

def _compare_logprefix(fst, inner_lm, target, atol=1e-8):
    """Compare Incremental.logprefix against ReferenceTransducedLM._score."""
    inc = Incremental(fst, lm=inner_lm)
    ref_tlm = ReferenceTransducedLM(inner_lm, fst)
    ref_state = ref_tlm(tuple(target))
    got = inc.logprefix(tuple(target))
    want = ref_state._score(tuple(target))
    assert got == pytest.approx(want, abs=atol), \
        f"target={target}: got={got} want={want}"


class TestLogpPrefix:

    def test_mapping_fst_empty(self):
        fst = _mapping_fst()
        lm = _finite_inner_lm()
        _compare_logprefix(fst, lm, '')

    def test_mapping_fst_after_x(self):
        fst = _mapping_fst()
        lm = _finite_inner_lm()
        _compare_logprefix(fst, lm, 'x')

    def test_bounded_copy(self):
        fst = _bounded_copy_fst(['a', 'b'], 2)
        lm = _finite_inner_lm()
        _compare_logprefix(fst, lm, '')
        _compare_logprefix(fst, lm, 'a')
        _compare_logprefix(fst, lm, 'ab')

    def test_anbn(self):
        fst = examples.anbn()
        lm = FiniteLM({('a', 'a', 'a'): 0.4, ('a',): 0.3, ('a', 'a'): 0.3})
        _compare_logprefix(fst, lm, '')
        _compare_logprefix(fst, lm, 'b')

    def test_duplicate(self):
        fst = examples.duplicate(set('ab'))
        lm = FiniteLM({
            (): 0.1, ('a',): 0.35, ('b',): 0.25,
            ('a', 'b'): 0.15, ('b', 'a'): 0.15,
        })
        _compare_logprefix(fst, lm, '')
        _compare_logprefix(fst, lm, 'a')
        _compare_logprefix(fst, lm, 'aa')

    def test_dead_target(self):
        """OOV target symbol raises ValueError."""
        fst = _mapping_fst()
        lm = _finite_inner_lm()
        inc = Incremental(fst, lm=lm)
        with pytest.raises(ValueError, match="Out of vocabulary"):
            inc.logprefix(('z',))


# ── logprob tests ────────────────────────────────────────────────────────

def _compare_logprob(fst, inner_lm, target, atol=1e-8):
    """Compare Incremental.logprob against ReferenceTransducedLM."""
    inc = Incremental(fst, lm=inner_lm)
    ref_tlm = ReferenceTransducedLM(inner_lm, fst)
    # Reference: P(output = target) = P(prefix target) * P(EOS | target)
    ref_state = ref_tlm(tuple(target))
    ref_prefix = ref_state._score(tuple(target))
    ref_eos_cond = ref_state.logp_next[ref_tlm.eos]
    want = ref_prefix + ref_eos_cond
    got = inc.logprob(tuple(target))
    assert got == pytest.approx(want, abs=atol), \
        f"target={target}: got={got} want={want}"


class TestLogprob:

    def test_mapping_fst_empty(self):
        _compare_logprob(_mapping_fst(), _finite_inner_lm(), '')

    def test_mapping_fst_after_x(self):
        _compare_logprob(_mapping_fst(), _finite_inner_lm(), 'x')

    def test_bounded_copy(self):
        fst = _bounded_copy_fst(['a', 'b'], 2)
        lm = _finite_inner_lm()
        _compare_logprob(fst, lm, '')
        _compare_logprob(fst, lm, 'a')
        _compare_logprob(fst, lm, 'ab')

    def test_anbn(self):
        fst = examples.anbn()
        lm = FiniteLM({('a', 'a', 'a'): 0.4, ('a',): 0.3, ('a', 'a'): 0.3})
        _compare_logprob(fst, lm, '')
        _compare_logprob(fst, lm, 'b')

    def test_duplicate(self):
        fst = examples.duplicate(set('ab'))
        lm = FiniteLM({
            (): 0.1, ('a',): 0.35, ('b',): 0.25,
            ('a', 'b'): 0.15, ('b', 'a'): 0.15,
        })
        _compare_logprob(fst, lm, '')
        _compare_logprob(fst, lm, 'a')
        _compare_logprob(fst, lm, 'aa')

    def test_dead_target(self):
        fst = _mapping_fst()
        lm = _finite_inner_lm()
        inc = Incremental(fst, lm=lm)
        with pytest.raises(ValueError, match="Out of vocabulary"):
            inc.logprob(('z',))

    def test_consistency_with_prefix(self):
        """logprob(target) <= logprefix(target) always."""
        fst = _bounded_copy_fst(['a', 'b'], 2)
        lm = _finite_inner_lm()
        inc = Incremental(fst, lm=lm)
        for target in [(), ('a',), ('b',), ('a', 'b')]:
            lp = inc.logprob(target)
            lp_prefix = inc.logprefix(target)
            assert lp <= lp_prefix + 1e-10, \
                f"logprob should be <= logprefix for target={target}"


# ── logp_next tests ──────────────────────────────────────────────────────

class FiniteLMState(LMState):
    """State for a finite-support LM. Computes exact conditionals from the trie."""

    def __init__(self, lm, prefix, logprefix):
        self._lm = lm
        self._prefix = prefix
        self.logprefix = logprefix
        self.eos = lm.eos

    def _prefix_mass(self, prefix):
        n = len(prefix)
        return sum(p for s, p in self._lm._string_probs.items()
                   if len(s) >= n and s[:n] == prefix)

    @cached_property
    def logp_next(self):
        Z = self._prefix_mass(self._prefix)
        if Z <= 0:
            return LogDistr({self.eos: 0.0})
        scores = {}
        n = len(self._prefix)
        next_tokens = {s[n] for s in self._lm._string_probs
                       if len(s) > n and s[:n] == self._prefix}
        for tok in next_tokens:
            mass = self._prefix_mass(self._prefix + (tok,))
            if mass > 0:
                scores[tok] = np.log(mass / Z)
        eos_mass = self._lm._string_probs.get(self._prefix, 0)
        if eos_mass > 0:
            scores[self.eos] = np.log(eos_mass / Z)
        elif not scores:
            scores[self.eos] = 0.0
        return LogDistr(scores)

    def __rshift__(self, token):
        if token == self.eos:
            raise ValueError("Cannot advance past EOS")
        lp = self.logp_next[token]
        return FiniteLMState(self._lm, self._prefix + (token,), self.logprefix + lp)


class FiniteLM(LM):
    """LM with exact support on a finite set of strings."""

    def __init__(self, string_probs, eos='<EOS>'):
        self.eos = eos
        self._string_probs = string_probs

    def initial(self):
        return FiniteLMState(self, (), 0.0)


def _random_finite_lm(alphabet, max_len, seed=0):
    """Random FiniteLM over all strings up to max_len from alphabet."""
    rng = np.random.RandomState(seed)
    from itertools import product
    strings = [()]
    for length in range(1, max_len + 1):
        strings.extend(product(alphabet, repeat=length))
    weights = rng.dirichlet(np.ones(len(strings)))
    return FiniteLM({s: w for s, w in zip(strings, weights)})



def _run_logp_test(fst, max_source_len=None, target_depth=None, seed=42, atol=1e-8):
    """Test logp_next against BruteForceTransducedLM for target prefixes up to depth."""
    source_alphabet = fst.A - {EPSILON}
    target_alphabet = fst.B - {EPSILON}

    if max_source_len is None:
        k = len(source_alphabet)
        if k <= 3: max_source_len = 4
        elif k <= 6: max_source_len = 3
        elif k <= 15: max_source_len = 2
        else: max_source_len = 1

    if target_depth is None:
        target_depth = 2 if len(target_alphabet) <= 3 else 1

    lm = _random_finite_lm(source_alphabet, max_len=max_source_len, seed=seed)
    inc = Incremental(fst, lm=lm)
    ref_tlm = BruteForceTransducedLM(lm._string_probs, fst, eos=lm.eos)

    def check(target):
        got = inc.logp_next(target)
        got_v3 = inc.logp_next_v3(target)
        got_v4 = inc.logp_next_v4(target)
        ref_logp_next = ref_tlm(target).logp_next
        for y in target_alphabet:
            got_val = got[y] if y in got else float('-inf')
            got_v3_val = got_v3[y] if y in got_v3 else float('-inf')
            got_v4_val = got_v4[y] if y in got_v4 else float('-inf')
            ref_val = ref_logp_next[y]
            assert got_val == pytest.approx(ref_val, abs=atol), \
                f"target={target} symbol={y}: got={got_val} ref={ref_val}"
            assert got_v3_val == pytest.approx(ref_val, abs=atol), \
                f"target={target} symbol={y}: got_v3={got_v3_val} ref={ref_val}"
            assert got_v4_val == pytest.approx(ref_val, abs=atol), \
                f"target={target} symbol={y}: got_v4={got_v4_val} ref={ref_val}"
        got_eos = got[inc.EOS] if inc.EOS in got else float('-inf')
        got_v3_eos = got_v3[inc.EOS] if inc.EOS in got_v3 else float('-inf')
        got_v4_eos = got_v4[inc.EOS] if inc.EOS in got_v4 else float('-inf')
        ref_eos = ref_logp_next[lm.eos]
        assert got_eos == pytest.approx(ref_eos, abs=atol), \
            f"target={target} EOS: got={got_eos} ref={ref_eos}"
        assert got_v3_eos == pytest.approx(ref_eos, abs=atol), \
            f"target={target} EOS: got_v3={got_v3_eos} ref={ref_eos}"
        assert got_v4_eos == pytest.approx(ref_eos, abs=atol), \
            f"target={target} EOS: got_v4={got_v4_eos} ref={ref_eos}"

    def recurse(target, depth):
        if depth < 0:
            return
        check(target)
        if depth > 0:
            ref_state = ref_tlm(target)
            for y in target_alphabet:
                if ref_state.logp_next[y] > float('-inf'):
                    recurse(target + (y,), depth - 1)

    recurse((), target_depth)


def _mapping_fst():
    """Acyclic FST: a→x, b→y. Start state is also stop (empty→empty)."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'y', 2)
    fst.add_stop(1)
    fst.add_stop(2)
    return fst


def _bounded_copy_fst(alphabet, max_len):
    """Acyclic copy FST that copies strings up to max_len symbols."""
    fst = FST()
    for i in range(max_len + 1):
        if i == 0:
            fst.add_start(i)
        fst.add_stop(i)
        if i < max_len:
            for a in alphabet:
                fst.add_arc(i, a, a, i + 1)
    return fst


def _finite_inner_lm():
    return FiniteLM({
        (): 0.2,
        ('a',): 0.3,
        ('b',): 0.2,
        ('a', 'b'): 0.15,
        ('b', 'a'): 0.15,
    })


def _compare_logp_next(fst, inner_lm, target, atol=1e-8):
    """Compare Incremental.logp_next against ReferenceTransducedLM."""
    inc = Incremental(fst, lm=inner_lm)
    ref_tlm = ReferenceTransducedLM(inner_lm, fst)

    got = inc.logp_next(tuple(target))
    ref_logp_next = ref_tlm(tuple(target)).logp_next

    # Compare each target symbol
    target_alphabet = fst.B - {EPSILON}
    for y in target_alphabet:
        got_val = got[y] if y in got else float('-inf')
        ref_val = ref_logp_next[y]
        assert got_val == pytest.approx(ref_val, abs=atol), \
            f"target={target} symbol={y}: got={got_val} ref={ref_val}"

    # Compare EOS
    eos_key = inc.EOS
    got_eos = got[eos_key] if eos_key in got else float('-inf')
    ref_eos = ref_logp_next[inner_lm.eos]
    assert got_eos == pytest.approx(ref_eos, abs=atol), \
        f"target={target} EOS: got={got_eos} ref={ref_eos}"


class TestLogpNext:

    def test_mapping_fst_initial(self):
        _compare_logp_next(_mapping_fst(), _finite_inner_lm(), '')

    def test_mapping_fst_after_x(self):
        _compare_logp_next(_mapping_fst(), _finite_inner_lm(), 'x')

    def test_bounded_copy_initial(self):
        _compare_logp_next(
            _bounded_copy_fst(['a', 'b'], 2), _finite_inner_lm(), '')

    def test_bounded_copy_after_a(self):
        _compare_logp_next(
            _bounded_copy_fst(['a', 'b'], 2), _finite_inner_lm(), 'a')

    def test_bounded_copy_after_ab(self):
        _compare_logp_next(
            _bounded_copy_fst(['a', 'b'], 2), _finite_inner_lm(), 'ab')

    def test_simple_replace(self):
        fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c')])
        lm = FiniteLM({
            ('1',): 0.3, ('2',): 0.3, ('3',): 0.2,
            ('1', '2'): 0.1, ('2', '1'): 0.1,
        })
        _compare_logp_next(fst, lm, '')
        _compare_logp_next(fst, lm, 'a')

    def test_samuel_example(self):
        fst = examples.samuel_example()
        lm = FiniteLM({('a',): 0.5, ('b',): 0.3, (): 0.2})
        _compare_logp_next(fst, lm, '')
        _compare_logp_next(fst, lm, 'c')

    def test_simple_replace_deeper(self):
        """Deeper targets exercise the expansion loop more."""
        fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c')])
        lm = FiniteLM({
            ('1',): 0.3, ('2',): 0.3, ('3',): 0.2,
            ('1', '2'): 0.1, ('2', '1'): 0.1,
        })
        _compare_logp_next(fst, lm, 'ab')
        _compare_logp_next(fst, lm, 'ba')

    def test_bounded_copy_3(self):
        """Larger copy FST with more source strings."""
        fst = _bounded_copy_fst(['a', 'b'], 3)
        lm = FiniteLM({
            (): 0.1,
            ('a',): 0.2, ('b',): 0.15,
            ('a', 'b'): 0.15, ('b', 'a'): 0.1,
            ('a', 'a', 'b'): 0.1, ('b', 'b', 'a'): 0.1,
            ('a', 'b', 'a'): 0.1,
        })
        _compare_logp_next(fst, lm, '')
        _compare_logp_next(fst, lm, 'a')
        _compare_logp_next(fst, lm, 'ab')

    def test_ambiguous_fst(self):
        """Two source symbols map to the same output."""
        fst = FST()
        fst.add_start(0); fst.add_stop(0)
        fst.add_arc(0, 'a', 'x', 1)
        fst.add_arc(0, 'b', 'x', 2)
        fst.add_stop(1); fst.add_stop(2)
        lm = FiniteLM({(): 0.2, ('a',): 0.5, ('b',): 0.3})
        _compare_logp_next(fst, lm, '')
        _compare_logp_next(fst, lm, 'x')

    def test_duplicate_fst(self):
        """Duplicate FST: each source symbol produces two copies."""
        fst = examples.duplicate(set('ab'))
        lm = FiniteLM({
            (): 0.1,
            ('a',): 0.35, ('b',): 0.25,
            ('a', 'b'): 0.15, ('b', 'a'): 0.15,
        })
        _compare_logp_next(fst, lm, '')
        _compare_logp_next(fst, lm, 'a')
        _compare_logp_next(fst, lm, 'aa')

    def test_anbn(self):
        fst = examples.anbn()
        lm = FiniteLM({
            ('a', 'a', 'a'): 0.4,
            ('a',): 0.3,
            ('a', 'a'): 0.3,
        })
        _compare_logp_next(fst, lm, '')
        _compare_logp_next(fst, lm, 'b')

    def test_diamond_with_selfloops(self):
        """Diamond FST with epsilon output, ambiguity, and self-loops."""
        fst = FST()
        fst.add_start(0)
        fst.add_stop(0)
        fst.add_stop(2)
        fst.add_stop(3)
        fst.add_arc(0, 'a', EPSILON, 1)
        fst.add_arc(0, 'a', 'c', 2)
        fst.add_arc(1, 'b', 'c', 3)
        fst.add_arc(2, 'a', 'c', 3)
        fst.add_arc(3, 'a', 'c', 3)
        fst.add_arc(3, 'b', 'c', 3)
        lm = _random_finite_lm(sorted(fst.A - {EPSILON}), max_len=5, seed=42)
        _compare_logp_next(fst, lm, '')
        _compare_logp_next(fst, lm, 'c')
        _compare_logp_next(fst, lm, 'cc')

    def test_step4_eos(self):
        """Exercises Step 4 EOS: cylinder extension produces exactly target."""
        # FST: 0(start,stop) --a:x--> 1(stop), 0 --b:x--> 1
        #       1 --a:eps--> 0, 1 --b:y--> 0
        # T(a)=x, T(b)=x, T(aa)=x, T(ba)=x, T(ab)=xy, T(bb)=xy
        # (a,) and (b,) are cylinders for (x,) but NOT for (x,y).
        # Expanding the queue discovers (aa) and (ba) with T=x=target exactly.
        fst = FST()
        fst.add_start(0)
        fst.add_stop(0)
        fst.add_stop(1)
        fst.add_arc(0, 'a', 'x', 1)
        fst.add_arc(0, 'b', 'x', 1)
        fst.add_arc(1, 'a', EPSILON, 0)
        fst.add_arc(1, 'b', 'y', 0)
        lm = FiniteLM({
            (): 0.05,
            ('a',): 0.15, ('b',): 0.15,
            ('a', 'a'): 0.15, ('b', 'a'): 0.1,
            ('a', 'b'): 0.2, ('b', 'b'): 0.2,
        })
        _compare_logp_next(fst, lm, '')
        _compare_logp_next(fst, lm, 'x')

    def test_normalization(self):
        """Verify logp_next produces a valid distribution (sums to ~1)."""
        fst = _bounded_copy_fst(['a', 'b'], 2)
        lm = _finite_inner_lm()
        inc = Incremental(fst, lm=lm)
        dist = inc.logp_next(())
        total = logsumexp(list(dist.values()))
        assert abs(total) < 1e-8, f"Should sum to 1 in log-space, got {total}"


# ── logp_next tests for all test_general.py examples ─────────────────────

_GENERAL_EXAMPLES = [
    ('replace', lambda: examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])),
    ('delete_b', examples.delete_b),
    ('samuel_example', examples.samuel_example),
    ('small', examples.small),
    ('sdd1_fst', examples.sdd1_fst),
    ('duplicate', lambda: examples.duplicate(set('12345'))),
    ('number_comma_separator', lambda: examples.number_comma_separator({'a', ',', ' ', '0'}, Digit={'0'})),
    ('newspeak2', examples.newspeak2),
    ('lookahead', examples.lookahead),
    ('weird_copy', examples.weird_copy),
    ('triplets_of_doom', examples.triplets_of_doom),
    ('infinite_quotient', examples.infinite_quotient),
    ('parity', lambda: examples.parity({'a', 'b'})),
    ('gated_universal', examples.gated_universal),
    ('complementary_halves', examples.complementary_halves),
    ('shrinking_nonuniversal', examples.shrinking_nonuniversal),
    ('scaled_newspeak', lambda: examples.scaled_newspeak(n_patterns=3, alpha_size=6)),
    ('scaled_newspeak_partial', lambda: examples.scaled_newspeak(n_patterns=3, alpha_size=6, n_partial=2)),
    ('layered_witnesses', examples.layered_witnesses),
    ('doom', lambda: examples.doom({'a', 'b'}, K=5)),
    ('mystery1', examples.mystery1),
    ('mystery2', examples.mystery2),
    ('infinite_quotient2', examples.infinite_quotient2),
    ('mystery3', examples.mystery3),
    ('mystery4', examples.mystery4),
    ('mystery5', examples.mystery5),
    ('mystery7', examples.mystery7),
    ('mystery8', examples.mystery8),
    ('anbn', examples.anbn),
    ('backticks_to_quote', examples.backticks_to_quote),
    ('parity_copy', examples.parity_copy),
    ('togglecase', examples.togglecase),
    ('lowercase', examples.lowercase),
    ('bpe_like', lambda: examples.bpe_like(vocab_size=30, alphabet=tuple("ab"), max_len=3)),
    ('bpe_embedded', lambda: examples.bpe_embedded(vocab_size=30, alphabet=tuple("ab"), max_len=3, wrapper_alpha=tuple("xy"))),
    ('diamond_with_selfloops', lambda: _diamond_fst()),
]


def _diamond_fst():
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_stop(2)
    fst.add_stop(3)
    fst.add_arc(0, 'a', EPSILON, 1)
    fst.add_arc(0, 'a', 'c', 2)
    fst.add_arc(1, 'b', 'c', 3)
    fst.add_arc(2, 'a', 'c', 3)
    fst.add_arc(3, 'a', 'c', 3)
    fst.add_arc(3, 'b', 'c', 3)
    return fst


_XFAIL_EXAMPLES = {
    'samuel_example',  # is_functional() misses epsilon ambiguity: T(ab)={c,cx}
}


@pytest.mark.parametrize('name,fst_fn', _GENERAL_EXAMPLES,
                         ids=[x[0] for x in _GENERAL_EXAMPLES])
def test_logp_next_general(name, fst_fn):
    fst = fst_fn()
    if not fst.is_functional()[0]:
        pytest.skip("non-functional FST")
    if name in _XFAIL_EXAMPLES:
        pytest.xfail(f"{name}: known issue")
    _run_logp_test(fst)
