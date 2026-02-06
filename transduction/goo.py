from transduction.lazy import Lazy, EPSILON
from transduction.fsa import FSA
from transduction.fst import FST, UniversalityFilter
from collections import deque

from transduction.lm.statelm import decode_hf_tokenizer



def bpe_wfst(tokenizer, readable=False):
    m = FST()
    m.add_start(())
    drop = {x.encode() for x in tokenizer.all_special_tokens}
    _, _, _decode, _ = decode_hf_tokenizer(tokenizer)
    for i, x in enumerate(_decode):
        if x in drop:
            continue
        _x = x
        x = tuple(x)
        for j in range(len(x)):
            m.add_arc(x[:j], EPSILON, bytes([x[j]]), x[:j+1])
        if readable:
            m.add_arc(x, Token(i, _x), EPSILON, ())
        else:
            m.add_arc(x, i, EPSILON, ())
    m.add_stop(())
    return m


class Token:
    def __init__(self, i, bytes):
        self.i = i
        self.bytes = bytes
    def __repr__(self):
        return f'{str(self.bytes)[2:-1]}/{self.i}'
    def __hash__(self):
        return hash(self.i)
    def __eq__(self, other):
        if isinstance(other, Token):
            return self.i == other.i
        else:
            return self.i == other
    def __radd__(self, other):
        if isinstance(other, int):
            return (other, self)
        else:
            return (*other, self)
    def __add__(self, other):
        if isinstance(other, (int, Token)):
            return (self, other)
        else:
            return (self, *other)



class LazyPrecoverNFA_slower(Lazy):

    def __init__(self, fst, target):
        self.fst = fst
        self.target = target
        self.N = len(target)

    def arcs(self, state):
        (i, n) = state
        if n == self.N:                    # i.e, target <= ys
            for x, _, j in self.fst.arcs(i):
                yield (x, (j, self.N))
        else:                              # i.e, ys < target)
            for x, y, j in self.fst.arcs(i):
                if y == EPSILON:
                    yield (x, (j, n))
                elif y == self.target[n]:
                    yield (x, (j, n+1))

    def start(self):
        for i in self.fst.I:
            yield (i, 0)

    def is_final(self, state):
        (i, ys) = state
        return self.fst.is_final(i) and ys == self.N



class LazyPrecoverNFA(Lazy):

    def __init__(self, fst, target):
        self.fst = fst
        self.target = target
        self.N = len(target)
        fst.ensure_arc_indexes()
        self._has_eps = EPSILON in fst.A

    def epsremove(self):
        if self._has_eps:
            return super().epsremove()
        return self

    def arcs(self, state):
        (i, n) = state
        if n == self.N:
            for x, j in self.fst.index_i_xj.get(i, ()):
                yield (x, (j, self.N))
        else:
            for x, j in self.fst.index_iy_xj.get((i, EPSILON), ()):
                yield (x, (j, n))
            for x, j in self.fst.index_iy_xj.get((i, self.target[n]), ()):
                yield (x, (j, n+1))

    def arcs_x(self, state, x):
        (i, n) = state
        if n == self.N:
            for j in self.fst.index_ix_j.get((i, x), ()):
                yield (j, self.N)
        else:
            for j in self.fst.index_ixy_j.get((i, x, EPSILON), ()):
                yield (j, n)
            for j in self.fst.index_ixy_j.get((i, x, self.target[n]), ()):
                yield (j, n+1)

    def start(self):
        for i in self.fst.I:
            yield (i, 0)

    def is_final(self, state):
        (i, ys) = state
        return self.fst.is_final(i) and ys == self.N


class NonrecursiveDFADecomp:

    def __init__(self, fst, target):
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        assert set(target) <= self.target_alphabet

        # Implementation note: this is a truncated representation of the
        # Precover(fst, target).  The recursive algorithm attempts to do
        # something differently where the automaton allows the target buffer to
        # grow without bound.  This works in a surprising number of cases, but
        # it can fail to terminate (e.g., on the `triplets_of_doom`).
        fsa = LazyPrecoverNFA(fst, target).materialize().renumber().trim()
        dfa = fsa.lazy().det()
        filt = UniversalityFilter(fst, target, dfa, self.source_alphabet)

        Q = FSA()
        R = FSA()

        worklist = deque()
        visited = set()

        for i in dfa.start():
            worklist.append(i)
            visited.add(i)
            Q.add_start(i)
            R.add_start(i)

        while worklist:
            i = worklist.popleft()

            if dfa.is_final(i):
                if filt.is_universal(i):
                    Q.add_stop(i)
                    continue       # will not expand further
                else:
                    R.add_stop(i)  # will expand further

            for a, j in dfa.arcs(i):
                if j not in visited:
                    worklist.append(j)
                    visited.add(j)

                Q.add_arc(i, a, j)
                R.add_arc(i, a, j)

        self.fst = fst
        self.fsa = fsa
        self.dfa = dfa
        self.target = target
        self.quotient = Q
        self.remainder = R
