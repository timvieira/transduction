from transduction.util import Integerizer
from collections import defaultdict, deque
from functools import lru_cache


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


class frozenset(frozenset):
    "Same as frozenset, but with a nicer printing method."
    def __repr__(self):
        return '{%s}' % (','.join(str(x) for x in self))


class FSA:
    """Finite-state automaton (acceptor) over a symbolic alphabet.

    An FSA is a directed graph whose arcs carry single labels.  A path
    from a start state to a stop state spells out a string; the set of
    all such strings is the FSA's language.

    Supports regular-language operations: union (``+`` / ``|``),
    concatenation (``*``), intersection (``&``), difference (``-``),
    complement (``invert``), Kleene star (``star``), left/right
    quotient (``//`` / ``/``), determinization (``det``),
    minimization (``min``), and language equivalence (``equal``).

    Attributes:
        states: Set of all states.
        start: Set of initial states.
        stop: Set of final (accepting) states.
        syms: Set of arc labels observed.
    """

    def __init__(self, start=(), arcs=(), stop=()):
        self.states = set()
        self.start = set()
        self.stop = set()
        self.syms = set()
        self.edges = defaultdict(lambda: defaultdict(set))
        # use the official methods for the constructor's initialization
        for i in start: self.add_start(i)
        for i in stop: self.add_stop(i)
        for i,a,j in arcs: self.add_arc(i,a,j)

    def materialize(self):
        # the FSA is already materialized
        return self

    def lazy(self):
        from transduction.lazy import LazyWrapper
        return LazyWrapper(self)

    def as_tuple(self):
        return (frozenset(self.states),
                frozenset(self.start),
                frozenset(self.stop),
                frozenset(self.arcs()))

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other):
        return self.as_tuple() == other.as_tuple()

    def __repr__(self):
        x = ['{']
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
        if not self.states:
            return {'text/html': '<center>∅</center>'}
        return self.graphviz()._repr_mimebundle_(*args, **kwargs)

    def graphviz(self, fmt_node=lambda x: x, sty_node=lambda x: {}, fmt_edge=lambda i,a,j: 'ε' if a == EPSILON else a):
        """Return a Graphviz digraph for visualization.

        Optional callbacks customize rendering: ``fmt_node(state)`` formats
        node labels, ``sty_node(state)`` returns a dict of Graphviz
        attributes, and ``fmt_edge(i, a, j)`` formats edge labels.
        """
        from transduction.viz import _render_graphviz
        return _render_graphviz(
            self.states, self.start, self.stop,
            arc_iter=self.arcs,
            fmt_node=fmt_node, fmt_edge=fmt_edge, sty_node=sty_node,
        )

    def D(self, x):
        """Left derivative: the language of strings ``y`` such that ``x·y`` is accepted."""
        e = self.epsremove()
        m = FSA(start = set(e.start), stop = set(e.stop))
        for i,a,j in e.arcs():
            if i in e.start and a == x:
                m.add(i,eps,j)
            else:
                m.add(i,a,j)
        return m

    def add(self, i, a, j):
        """Add arc from state ``i`` to state ``j`` with label ``a``. Creates states implicitly."""
        self.edges[i][a].add(j)
        self.states.add(i); self.syms.add(a); self.states.add(j)
        return self

    add_arc = add

    def add_start(self, i):
        """Mark state ``i`` as an initial state (creates it if needed)."""
        self.start.add(i)
        self.states.add(i)
        return self

    def add_stop(self, i):
        """Mark state ``i`` as a final (accepting) state (creates it if needed)."""
        self.stop.add(i)
        self.states.add(i)
        return self

    def is_final(self, i):
        """Return True if state ``i`` is a final (accepting) state."""
        return i in self.stop

    def arcs(self, i=None, a=None):
        """Iterate over arcs.

        - ``arcs()`` → yields all ``(i, a, j)`` triples.
        - ``arcs(i)`` → yields ``(a, j)`` pairs for arcs from state ``i``.
        - ``arcs(i, a)`` → yields destination states ``j`` for arcs ``i --a--> j``.
        """
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

    def arcs_x(self, i, x):
        """Return the set of destination states for arcs ``i --x--> *``."""
        return self.edges[i][x]

    def reverse(self):
        """Return the reversal: arcs are flipped, start and stop states are swapped."""
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
        """Return the set of states reachable from any start state."""
        return self._accessible(self.start)

    def trim(self):
        """Return a copy keeping only states on some start-to-stop path."""
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
        """Return a copy with states relabeled as consecutive integers."""
        return self.rename(Integerizer())

    def rename(self, f):
        """Return a new FSA with states relabeled by ``f(state)``.

        If ``f`` is not injective, distinct states may merge.
        """
        m = FSA()
        for i in self.start:
            m.add_start(f(i))
        for i in self.stop:
            m.add_stop(f(i))
        for i, a, j in self.arcs():
            m.add(f(i), a, f(j))
        return m

    def map_labels(self, f):
        "Transform arc labels by applying f to each label."
        m = FSA()
        for i in self.start:
            m.add_start(i)
        for i in self.stop:
            m.add_stop(i)
        for i, a, j in self.arcs():
            m.add(i, f(a), j)
        return m

    def rename_apart(self, other):
        """Rename states of ``self`` and ``other`` so their state sets are disjoint."""
        f = Integerizer()
        self = self.rename(lambda i: f((0, i)))
        other = other.rename(lambda i: f((1, i)))
        assert self.states.isdisjoint(other.states)
        return (self, other)

    def __mul__(self, other):
        """Concatenation: ``self * other`` accepts strings ``xy`` where ``x in self`` and ``y in other``."""
        self, other = self.rename_apart(other)
        m = FSA(
            start = self.start,
            stop = other.stop,
        )
        for i in self.stop:
            for j in other.start:
                m.add(i,eps,j)
        for i,a,j in self.arcs():
            m.add(i,a,j)
        for i,a,j in other.arcs():
            m.add(i,a,j)
        return m

    def __add__(self, other):
        """Union: ``self + other`` accepts strings in either language."""
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
        """Kleene plus: one or more repetitions of ``self``."""
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
        """Kleene star: zero or more repetitions of ``self``."""
        return one + self.p()

    def epsremove(self):
        """Return an equivalent FSA with all epsilon transitions removed."""
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

    def det(self):
        """Determinize via the powerset (subset) construction. Returns a DFA."""
        self = self.epsremove()

        def powerarcs(Q):
            for a in self.syms:
                yield a, frozenset(j for i in Q for j in self.edges[i][a])

        m = dfs([frozenset(self.start)], powerarcs)

        for powerstate in m.states:
            if powerstate & self.stop:
                m.add_stop(powerstate)

        return m

    def min_brzozowski(self):
        """Minimize via Brzozowski's algorithm (reverse-determinize-reverse-determinize).

        Works on NFAs directly (no prior determinization needed).  Simple but
        may be slow due to two subset constructions; can blow up if the
        intermediate DFA is exponentially larger.
        """
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
        """Minimize via Hopcroft's partition-refinement algorithm.

        Determinizes first, then iteratively splits equivalence classes.
        O(n log n) in theory.  See ``min_faster`` for an optimized variant.
        """
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
            for q in qs:
                minstates[q] = i #minstate

        return self.rename(lambda i: minstates[i]).trim()

    def min_faster(self):
        """Optimized Hopcroft minimization with a ``find`` index for O(1) block lookup.

        Same algorithm as ``min_fast`` but avoids scanning all blocks on each
        refinement step.  This is the default (aliased as ``min``).
        """
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

                # Group pre-images by their current block; this lets us
                # replace the O(|Y|) superset check `X >= Y` with an
                # O(1) length comparison.
                block_members = defaultdict(set)
                for j in A:
                    for i in inv[j, a]:
                        block_members[find[i]].add(i)

                for block, YX in block_members.items():
                    Y = P[block]

                    if len(YX) == len(Y): continue

                    Y_X = Y - YX

                    # we will replace block with the intersection case (no
                    # need to update `find` index for YX elements)
                    P[block] = YX

                    new_block = len(P)
                    for i in Y_X:
                        find[i] = new_block

                    P.append(Y_X)
                    W.append(YX if len(YX) < len(Y_X) else Y_X)

        return self.rename(lambda i: find[i]).trim()

    min = min_faster

    def equal(self, other):
        """Test language equivalence via minimal-DFA isomorphism."""
        if isinstance(other, (frozenset, list, set, tuple)):
            other = FSA.from_strings(other)
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

    def __and__(self, other):
        """Intersection: ``self & other`` accepts strings in both languages."""

        self = self.epsremove().renumber()
        other = other.epsremove().renumber()

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
        """Make the FSA complete over ``syms`` by adding a non-accepting sink state for missing transitions."""

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
        """Set difference: ``self - other`` accepts strings in ``self`` but not in ``other``."""
        return self & other.invert(self.syms | other.syms)

    __or__ = __add__

    def __xor__(self, other):
        "Symmetric difference"
        return (self | other) - (self & other)

    def invert(self, syms):
        """Complement over alphabet ``syms``: accepts all strings not in the language."""

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

        self = self.epsremove()
        other = other.epsremove()

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
        """Build a single-arc FSA accepting exactly the one-symbol string ``x``."""
        m = cls()
        m.add_start(0); m.add_stop(1); m.add(0,x,1)
        return m

    @classmethod
    def from_string(cls, xs):
        """Build a linear FSA accepting exactly the string ``xs``."""
        m = cls()
        m.add_start(xs[:0])
        for i in range(len(xs)):
            m.add(xs[:i], xs[i], xs[:i+1])
        m.add_stop(xs)
        return m

    @classmethod
    def from_strings(cls, Xs):
        """Build an FSA accepting the union of the given strings (trie structure)."""
        m = cls()
        for xs in Xs:
            m.add_start(xs[:0])
            for i in range(len(xs)):
                m.add(xs[:i], xs[i], xs[:i+1])
            m.add_stop(xs)
        return m

    def __contains__(self, xs):
        """Test if string ``xs`` is in the language: ``xs in fsa``."""
        d = self.det()
        [s] = d.start
        for x in xs:
            t = d.edges[s][x]
            if not t:
                return False
            [s] = t
        return (s in d.stop)

    def merge(self, S, name=None):
        """Merge all states in ``S`` into a single state (default name: ``min(S)``)."""
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
        """Build a single-state FSA accepting all strings over ``alphabet`` (Σ*)."""
        u = FSA()
        u.add_start(0)
        for a in alphabet:
            u.add(0, a, 0)
        u.add_stop(0)
        return u

    def language(self, max_length=None):
        "Enumerate strings in the language of this FSA."
        worklist = deque()
        worklist.extend([(i, ()) for i in self.start])
        while worklist:
            (i, x) = worklist.popleft()
            if i in self.stop:
                yield x
            if max_length is not None and len(x) >= max_length:
                continue
            for a, j in self.arcs(i):
                if a == EPSILON:
                    worklist.append((j, x))
                else:
                    worklist.append((j, x + (a,)))


EPSILON = eps = ''

FSA.one = one = FSA()
one.add_start(0); one.add_stop(0)

FSA.zero = zero = FSA()
