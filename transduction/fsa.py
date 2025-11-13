from arsenal import Integerizer, colors
from collections import defaultdict
from functools import lru_cache
from graphviz import Digraph


def dfs(Ps, arcs):
    stack = list(Ps)
    m = FSA()
    for P in Ps: m.add_start(P)
    while stack:
        P = stack.pop()
        for a, Q in arcs(P):
            if Q not in m.states:
                stack.append(Q)
                m.states.add(Q)
            m.add(P, a, Q)
    return m


_frozenset = frozenset
class frozenset(_frozenset):
    def __repr__(self):
        return '{%s}' % (','.join(str(x) for x in self))


class FSA:

    def __init__(self, start=(), stop=()):
        self.states = set()
        self.start = set()
        self.stop = set()
        # use the official methods for the constructor's initialization
        for i in start: self.add_start(i)
        for i in stop: self.add_stop(i)
        self.edges = defaultdict(lambda: defaultdict(set))
        self.syms = set()

    def as_tuple(self):
        return (frozenset(self.states),
                frozenset(self.start),
                frozenset(self.stop),
#                frozenset(self.syms),
                frozenset(self.arcs()))

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other):
        return self.as_tuple() == other.as_tuple()

#    def __repr__(self):
#        return f'<{self.__class__.__name__} id={id(self)}>'
#        return repr(self.to_regex())

    def __repr__(self):
        x = ['{']   # todo: better print; include start/stop
        for s in self.states:
            ss = f'{s}'
            if s in self.start:
                ss = f'^{ss}'
            if s in self.stop:
                ss = f'{ss}$'
            x.append(f'  {ss}:')
            for a, t in self.arcs(s):
                x.append(f'    {a} -> {t}')
        x.append('}')
        return '\n'.join(x)

    def _repr_mimebundle_(self, *args, **kwargs):
        return self.graphviz()._repr_mimebundle_(*args, **kwargs)

    def graphviz(self, fmt_node=lambda x: x, sty_node=lambda x: {}, fmt_edge=lambda i,a,j: 'ε' if a == EPSILON else a):
        import html
        g = Digraph(
            graph_attr=dict(rankdir='LR'),
            node_attr=dict(
                fontname='Monospace',
                fontsize='8',
                height='.05',
                width='.05',
                margin="0.055,0.042",
                shape='box',
                style='rounded',
            ),
            edge_attr=dict(
                arrowsize='0.3',
                fontname='Monospace',
                fontsize='8'
            ),
        )
        f = Integerizer()

        # FIXME: make sure this name is actually unique
        start = '<start>'
        assert start not in self.states
        g.node(start, label='', shape='point', height='0', width='0')
        for i in self.start:
            g.edge(start, str(f(i)), label='')

        for i in self.states:
            label = html.escape(str(fmt_node(i)))
            #if i in self.start: label = '*'
            sty = dict(peripheries='2' if i in self.stop else '1')
            sty.update(sty_node(i))
            g.node(str(f(i)), label=label, **sty)

        # Collect parallel-edge labels by (i, j)
        by_pair = defaultdict(list)
        for i in self.states:
            for a, j in self.arcs(i):
                lbl = html.escape(str(fmt_edge(i,a,j)))
                by_pair[(str(f(i)), str(f(j)))].append(lbl)

        # Emit one edge per (i, j) with stacked labels
        for (u, v), labels in by_pair.items():
            # Stack with literal newlines. Graphviz renders '\n' as a line break.
            g.edge(u, v, label='\n'.join(sorted(labels)))

        return g

    def D(self, x):
        "left derivative"
        m = FSA()

        e = self.epsremoval()
        for i,a,j in e.arcs():

            if i in e.start and a == x:
                m.add(i,eps,j)
            else:
                m.add(i,a,j)

        m.start = set(e.start)
        m.stop = set(e.stop)
        return m

    def add(self, i, a, j):
        self.edges[i][a].add(j)
        self.states.add(i); self.syms.add(a); self.states.add(j)
        return self

    add_arc = add

    def add_start(self, i):
        self.start.add(i)
        self.states.add(i)
        return self

    def add_stop(self, i):
        self.stop.add(i)
        self.states.add(i)
        return self

    def is_final(self, i):
        return i in self.stop

    def arcs(self, i=None, a=None):
        if i is None and a is None:

            for i in self.edges:
                for a in self.edges[i]:
                    for j in self.edges[i][a]:
                        yield (i,a,j)

        elif i is not None and a is None:

            for a in self.edges[i]:
                for j in self.edges[i][a]:
                    yield (a,j)

        elif i is not None and a is not None:

            for j in self.edges[i][a]:
                yield j

        else:
            raise NotImplementedError()

    def reverse(self):
        m = FSA()
        for i in self.start:
            m.add_stop(i)
        for i in self.stop:
            m.add_start(i)
        for i, a, j in self.arcs():
            m.add(j, a, i)     # pylint: disable=W1114
        return m

    def _accessible(self, start):
        return dfs(start, self.arcs).states

    def accessible(self):
        return self._accessible(self.start)

    @lru_cache(None)
    def trim(self):
        c = self.accessible() & self.reverse().accessible()
        m = FSA()
        for i in self.start & c:
            m.add_start(i)
        for i in self.stop & c:
            m.add_stop(i)
        for i,a,j in self.arcs():
            if i in c and j in c:
                m.add(i,a,j)
        return m

    def renumber(self):
        return self.rename(Integerizer())

    def rename(self, f):
        "Note: f is not bijective, states may split/merge."
        m = FSA()
        for i in self.start:
            m.add_start(f(i))
        for i in self.stop:
            m.add_stop(f(i))
        for i, a, j in self.arcs():
            m.add(f(i), a, f(j))
        return m

    def rename_apart(self, other):
        f = Integerizer()
        self = self.rename(lambda i: f((0, i)))
        other = other.rename(lambda i: f((1, i)))
        assert self.states.isdisjoint(other.states)
        return (self, other)

    def __mul__(self, other):
        m = FSA()
        self, other = self.rename_apart(other)
        m.start = self.start
        m.stop = other.stop
        for i in self.stop:
            for j in other.start:
                m.add(i,eps,j)
        for i,a,j in self.arcs():
            m.add(i,a,j)
        for i,a,j in other.arcs():
            m.add(i,a,j)
        return m

    def __add__(self, other):
        m = FSA()
        [self, other] = self.rename_apart(other)
        m.start = self.start | other.start
        m.stop = self.stop | other.stop
        for i,a,j in self.arcs():
            m.add(i,a,j)
        for i,a,j in other.arcs():
            m.add(i,a,j)
        return m

    def p(self):
        "self^+"
        m = FSA()
        m.start = set(self.start)
        m.stop = set(self.stop)
        for i,a,j in self.arcs():
            m.add(i,a,j)
        for i in self.stop:
            m.add_stop(i)
            for j in self.start:
                m.add(i, eps, j)
        return m

    def star(self):
        "self^*"
        return one + self.p()

#    def L(self, s):
#        assert s in self.states
#        return dfs({s}, self.arcs)

    @lru_cache(None)
    def epsremoval(self):

        eps_m = FSA()
        for i,a,j in self.arcs():
            if a == eps:
                eps_m.add(i,a,j)

        @lru_cache
        def eps_accessible(i):
            return eps_m._accessible({i})

        m = FSA()

        for i,a,j in self.arcs():
            if a == eps: continue
            m.add(i, a, j)
            for k in eps_accessible(j):
                m.add(i, a, k)

        for i in self.start:
            m.add_start(i)
            for k in eps_accessible(i):
                m.add_start(k)

        for i in self.stop:
            m.add_stop(i)

        return m

    @lru_cache(None)
    def det(self):

        self = self.epsremoval()

        def powerarcs(Q):
            for a in self.syms:
                yield a, frozenset({j for i in Q for j in self.edges[i][a]})

        m = dfs([frozenset(self.start)], powerarcs)

        for powerstate in m.states:
            if powerstate & self.stop:
                m.add_stop(powerstate)

        return m

    def min_brzozowski(self):
        "Brzozowski's minimization algorithm"
        # https://en.wikipedia.org/wiki/DFA_minimization#Brzozowski's_algorithm

        # Proof of correctness:
        #
        # Let M' = M.r.d.r
        # Clearly,  [[M']] = [[M]]
        #
        # In M', there are no two states that can accept the same suffix
        # language because the reverse of M' is deterministic.
        #
        # The determinization of M' then creates powerstates, where every pair
        # of distinct powerstates R and S, there exists by construction at least
        # one state q of M' where q \in R and q \notin S. Such a q contributes
        # at least one word w \in [[q]] to the suffix language of q in [[R]]
        # that is not present in [[S]], since this word is unique to q (i.e., no
        # other state accepts it).  Thus, all pairs of states in M'.d are
        # distinguishable.
        #
        # Thus, after trimming of M'.d, we have a DFA with no indistinguishable
        # or unreachable states, which is must minimal.

        return self.reverse().det().reverse().det().trim()

    def min_fast(self):
        self = self.det().renumber()

        # calculate inverse of transition function (i.e., reverse arcs)
        inv = defaultdict(set)
        for i,a,j in self.arcs():
            inv[j,a].add(i)

        final = self.stop
        nonfinal = self.states - final

        P = [final, nonfinal]
        W = [final, nonfinal]

        while W:
            A = W.pop()
            for a in self.syms:
                X = {i for j in A for i in inv[j,a]}
                R = []
                for Y in P:
                    if X.isdisjoint(Y) or X >= Y:
                        R.append(Y)
                    else:
                        YX = Y & X
                        Y_X = Y - X
                        R.append(YX)
                        R.append(Y_X)
                        W.append(YX if len(YX) < len(Y_X) else Y_X)
                P = R

        # create new equivalence classes of states
        minstates = {}
        for i, qs in enumerate(P):
            #minstate = frozenset(qs)
            for q in qs:
                minstates[q] = i #minstate

        return self.rename(lambda i: minstates[i]).trim()

    def min_faster(self):
        self = self.det().renumber()

        # calculate inverse of transition function (i.e., reverse arcs)
        inv = defaultdict(set)
        for i,a,j in self.arcs():
            inv[j,a].add(i)

        final = self.stop
        nonfinal = self.states - final

        P = [final, nonfinal]
        W = [final, nonfinal]

        find = {i: block for block, elements in enumerate(P) for i in elements}

        while W:

            A = W.pop()
            for a in self.syms:

                X = {i for j in A for i in inv[j,a]}

                blocks = {find[i] for i in X}

                for block in blocks:
                    Y = P[block]

                    if X >= Y: continue

                    # TODO: use indexing to find nonempty (Y-X).
                    # Some notes:
                    #  - To be nonempty we need to find an element i* that is in Y
                    #    but not in X.  We already have an element i that is in Y
                    #    and X.
                    #  - We know that X and Y overlap (thanks to our indexing
                    #    trick).  Now, we want to filter out cases where X >= Y
                    #    because they do not need to be split.

                    YX = Y & X
                    Y_X = Y - X

                    # we will replace block with the intersection case (no
                    # need to update `find` index for YX elements)
                    P[block] = YX

                    new_block = len(P)
                    for i in Y_X:
                        find[i] = new_block

                    P.append(Y_X)
                    W.append(YX if len(YX) < len(Y_X) else Y_X)

        return self.rename(lambda i: find[i]).trim()

    min = lru_cache(None)(min_faster)

    def equal(self, other):
        return self.min()._dfa_isomorphism(other.min())

    def _dfa_isomorphism(self, other):
        "Find isomorphism between DFAs (if one exists)."

        # Requires that self and other are minimal DFAs

        # Theorem. If `self` and `other` are graphs with out-degree at most 1, then
        # the DFA works to determine whether G and H are isomorphic

        # A deterministic machine has exactly one start state

        # Two minimized DFAs are input
        # If the number of states is differs, these machines cannot be isomorphic
        if len(self.states) != len(other.states): return False
        if len(self.start) == 0: return len(other.start) == 0

        assert len(self.start) == 1 and len(other.start) == 1

        #self = self.renumber()
        #other = other.renumber()

        [p] = self.start
        [q] = other.start

        stack = [(p, q)]
        iso = {p: q}

        syms = self.syms | other.syms

        done = set()
        while stack:
            (p, q) = stack.pop()
            done.add((p,q))
            for a in syms:

                # presences of the arc has to be the same
                if (a in self.edges[p]) != (a in other.edges[q]):
                    return False

                if a not in self.edges[p]:
                    continue

                # machines are assumed deterministic
                [r] = self.edges[p][a]
                [s] = other.edges[q][a]

                if r in iso and iso[r] != s:
                    return False

                iso[r] = s
                if (r,s) not in done:
                    stack.append((r,s))

        return self.rename(iso.get) == other

#    def to_regex(self):
#        import numpy as np
#        from semirings.regex import Symbol
#        from semirings.kleene import kleene
#
#        n = len(self.states)
#
#        A = np.full((n,n), Symbol.zero)
#        start = np.full(n, Symbol.zero)
#        stop = np.full(n, Symbol.zero)
#
#        ix = Integerizer(list(self.states))
#
#        for i in self.states:
#            for a, j in self.arcs(i):
#                if a == eps:
#                    A[ix(i),ix(j)] += Symbol.one
#                else:
#                    A[ix(i),ix(j)] += Symbol(a)
#
#        for i in self.start:
#            start[ix(i)] += Symbol.one
#
#        for i in self.stop:
#            stop[ix(i)] += Symbol.one
#
#        return start @ kleene(A, Symbol) @ stop

    def __and__(self, other):
        "intersection"

        self = self.epsremoval().renumber()
        other = other.epsremoval().renumber()

        def product_arcs(Q):
            (q1, q2) = Q
            for a, j1 in self.arcs(q1):
                for j2 in other.edges[q2][a]:
                    yield a, (j1,j2)

        m = dfs({(q1, q2) for q1 in self.start for q2 in other.start},
                product_arcs)

        # final states
        for q1 in self.stop:
            for q2 in other.stop:
                m.add_stop((q1, q2))

        return m

    def add_sink(self, syms):
        "constructs a complete FSA"

        syms = set(syms)

        self = self.renumber()

        sink = len(self.states)
        for a in syms:
            self.add(sink, a, sink)

        for q in self.states:
            if q == sink: continue
            for a in syms - set(self.edges[q]):
                if a == eps: continue  # ignore epsilon
                self.add(q, a, sink)

        return self

    def __sub__(self, other):
        return self & other.invert(self.syms | other.syms)

    __or__ = __add__

    def __xor__(self, other):
        "Symmetric difference"
        return (self | other) - (self & other)

    def invert(self, syms):
        "create the complement of the machine"

        self = self.det().add_sink(syms)

        m = FSA()

        for i in self.states:
            for a, j in self.arcs(i):
                m.add(i, a, j)

        for q in self.start:
            m.add_start(q)

        for q in self.states - self.stop:
            m.add_stop(q)

        return m

    def __floordiv__(self, other):
        "left quotient self//other ≐ {y | ∃x ∈ other: x⋅y ∈ self}"

        # TODO: support NFA/epsilon arcs?
        self = self.epsremoval()
        other = other.epsremoval()

        # quotient arcs are very similar to product arcs except that the common
        # string is "erased" in the new machine.
        def quotient_arcs(Q):
            (q1, q2) = Q
            for a, j1 in self.arcs(q1):
                for j2 in other.edges[q2][a]:
                    yield eps, (j1, j2)

        m = dfs({(q1, q2) for q1 in self.start for q2 in other.start},
                quotient_arcs)

        # If we have managed to reach a final state of q2 then we can move into
        # the post-prefix set of states
        for (q1,q2) in set(m.states):
            if q2 in other.stop:
                m.add((q1, q2), eps, (q1,))

        # business as usual
        for q1 in self.states:
            for a, j1 in self.arcs(q1):
                m.add((q1,), a, (j1,))
        for q1 in self.stop:
            m.add_stop((q1,))

        return m

    def __truediv__(self, other):
        "right quotient self/other ≐ {x | ∃y ∈ other: x⋅y ∈ self}"
        return (self.reverse() // other.reverse()).reverse()   # reduce to left quotient on reversed languages

    def __lt__(self, other):
        "self ⊂ other"
        if self.equal(other): return False
        return (self & other).equal(self)

    def __le__(self, other):
        "self ⊆ other"
        return (self & other).equal(self)

    @classmethod
    def lift(cls, x):
        m = cls()
        m.add_start(0); m.add_stop(1); m.add(0,x,1)
        return m

    @classmethod
    def from_string(cls, xs):
        m = cls()
        m.add_start(xs[:0])
        for i in range(len(xs)):
            m.add(xs[:i], xs[i], xs[:i+1])
        m.add_stop(xs)
        return m

    @classmethod
    def from_strings(cls, Xs):
        m = cls()
        for xs in Xs:
            m.add_start(xs[:0])
            for i in range(len(xs)):
                m.add(xs[:i], xs[i], xs[:i+1])
            m.add_stop(xs)
        return m

    def __contains__(self, xs):
        d = self.det()
        [s] = d.start
        for x in xs:
            t = d.edges[s][x]
            if not t: break
            [s] = t
        return (s in d.stop)

    def merge(self, S, name=None):
        "merge states in `S` into a single state."
        if name is None: name = min(S)
        def f(s):
            return name if s in S else s
        m = FSA()
        for x in self.start:
            m.add_start(f(x))
        for x,a,y in self.arcs():
            m.add(f(x),a,f(y))
        for x in self.stop:
            m.add_stop(f(x))
        return m

    @staticmethod
    def universal(alphabet):
        u = FSA()
        u.add_start(0)
        for a in alphabet:
            u.add(0, a, 0)
        u.add_stop(0)
        return u


EPSILON = eps = ''

FSA.one = one = FSA()
one.add_start(0); one.add_stop(0)

FSA.zero = zero = FSA()
