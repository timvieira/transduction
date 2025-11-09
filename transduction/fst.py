from collections import defaultdict, deque
from functools import cached_property
from itertools import zip_longest

from transduction.fsa import FSA, EPSILON


import html

from arsenal import Integerizer
from graphviz import Digraph


# TODO: technically, we need to ensure that these are unique objects.
ε_1 = f'{EPSILON}₁'
ε_2 = f'{EPSILON}₂'


class FST:
    def __init__(self, start=(), stop=()):
        self.A = set()
        self.B = set()
        self.states = set()
        self.delta = defaultdict(lambda: defaultdict(set))
        self.I = set()
        self.F = set()
        for i in start: self.add_I(i)
        for i in stop: self.add_F(i)

    def __repr__(self):
        return f'{__class__.__name__}({len(self.states)} states)'

    def __str__(self):
        output = []
        output.append('{')
        for p in self.states:
            output.append(f'  {p} \t\t({p in self.I}, {p in self.F})')
            for a, q in self.arcs(p):
                output.append(f'    {a}: {q}')
        output.append('}')
        return '\n'.join(output)

    def is_final(self, i):
        return i in self.F

    def add_arc(self, i, a, b, j):  # pylint: disable=arguments-renamed
        self.states.add(i)
        self.states.add(j)
        self.delta[i][(a,b)].add(j)   # TODO: change this data structure to separarate a and b.
        self.A.add(a)
        self.B.add(b)

    def add_I(self, q):
        self.states.add(q)
        self.I.add(q)

    def add_F(self, q):
        self.states.add(q)
        self.F.add(q)

    def arcs(self, i):
        for (a, b), T in self.delta[i].items():
            for j in T:
                yield a, b, j

    def rename(self, f):
        "Note: If `f` is not bijective, states may merge."
        m = self.spawn()
        for i in self.I:
            m.add_I(f(i))
        for i in self.F:
            m.add_F(f(i))
        for i in self.states:
            for a, b, j in self.arcs(i):
                m.add_arc(f(i), a, b, f(j))
        return m

    @cached_property
    def renumber(self):
        return self.rename(Integerizer())

    def spawn(self, *, keep_init=False, keep_arcs=False, keep_stop=False):
        "Returns a new WFSA in the same semiring."
        m = self.__class__()
        if keep_init:
            for q in self.I:
                m.add_I(q)
        if keep_arcs:
            for i in self.states:
                for a, b, j in self.arcs(i):
                    m.add_arc(i, a, b, j)
        if keep_stop:
            for q in self.F:
                m.add_F(q)
        return m

    def _repr_mimebundle_(self, *args, **kwargs):
        return self.graphviz()._repr_mimebundle_(*args, **kwargs)

    def graphviz(
        self,
        fmt_node=lambda x: ' ',
        fmt_edge=lambda i, a, j: f'{str(a[0] or "ε")}:{str(a[1] or "ε")}' if a[0] != a[1] else str(a[0]),
        sty_node=lambda i: {},
    ):
        if len(self.states) == 0:
            import warnings
            warnings.warn('empty visualization')

        g = Digraph(
            graph_attr=dict(rankdir='LR'),
            node_attr=dict(
                fontname='Monospace',
                fontsize='8',
                height='.05', width='.05',
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

        # Start pointers
        for i in self.I:
            start = f'<start_{i}>'
            g.node(start, label='', shape='point', height='0', width='0')
            g.edge(start, str(f(i)), label='')

        # Nodes
        for i in self.states:
            sty = dict(peripheries='2' if i in self.F else '1')
            sty.update(sty_node(i))
            g.node(str(f(i)), label=str(fmt_node(i)), **sty)

        # Collect parallel-edge labels by (i, j)
        by_pair = defaultdict(list)
        for i in self.states:
            for a, b, j in self.arcs(i):
                lbl = fmt_edge(i, (a, b), j)
                by_pair[(str(f(i)), str(f(j)))].append(lbl)

        # Emit one edge per (i, j) with stacked labels
        for (u, v), labels in by_pair.items():
            # Stack with literal newlines. Graphviz renders '\n' as a line break.
            g.edge(u, v, label='\n'.join(sorted(labels)))

        return g

    def __call__(self, x, y):
        """
        Compute the total weight of x:y under the FST's weighted relation.  If one
        of x or y is None, we return the weighted language that is the cross
        section (we do so efficiently by representing it as a WFSA).
        """

        if x is not None and y is not None:
            if isinstance(x, str):
                x = FST.from_string(x)
            if isinstance(y, str):
                y = FST.from_string(y)
            return (x @ self @ y)

        elif x is not None and y is None:
            if isinstance(x, str):
                x = FST.from_string(x)
            return (x @ self).project(1)

        elif x is None and y is not None:
            if isinstance(y, str):
                y = FST.from_string(y)
            return (self @ y).project(0)

        else:
            return self

    @classmethod
    def from_string(cls, xs):
        m = cls()
        m.add_I(xs[:0])
        for i in range(len(xs)):
            m.add_arc(xs[:i], xs[i], xs[i], xs[:i+1])
        m.add_F(xs)
        return m

    @staticmethod
    def from_pairs(pairs):
        p = FST()
        p.add_I(0)
        p.add_F(1)
        for i, (xs, ys) in enumerate(pairs):
            p.add_arc(0, EPSILON, EPSILON, (i, 0))
            for j, (x, y) in enumerate(zip_longest(xs, ys, fillvalue=EPSILON)):
                p.add_arc((i, j), x, y, (i, j + 1))
            p.add_arc((i, max(len(xs), len(ys))), EPSILON, EPSILON, 1)
        return p

    def project(self, axis):
        """
        Project the FST into a FSA when `component` is 0, we project onto the left,
        and with 1 we project onto the right.
        """
        assert axis in [0, 1]
        A = FSA()
        for i in self.states:
            for a, b, j in self.arcs(i):
                if axis == 0:
                    A.add_arc(i, a, j)
                else:
                    A.add_arc(i, b, j)
        for i in self.I:
            A.add_start(i)
        for i in self.F:
            A.add_stop(i)
        return A

    @cached_property
    def T(self):
        "transpose swap left <-> right"
        T = self.spawn()
        for i in self.states:
            for a, b, j in self.arcs(i):
                T.add_arc(i, b, a, j)  # (a,b) -> (b,a)
        for q in self.I:
            T.add_I(q)
        for q in self.F:
            T.add_F(q)
        return T

    def __matmul__(self, other):
        "Relation composition; may coerce `other` to an appropriate type if need be."

        if isinstance(other, FSA):
            other = FST.diag(other)

        # minor efficiency trick: it's slightly more efficient to associate the composition as follows
        if len(self.states) < len(other.states):
            return (
                self._augment_epsilon_transitions(0)  # rename epsilons on the right
                ._compose(
                    epsilon_filter_fst(self.B),
                )  # this FST carefully combines the special epsilons
                ._compose(
                    other._augment_epsilon_transitions(1)
                )  # rename epsilons on th left
            )

        else:
            return (
                self._augment_epsilon_transitions(0)
                ._compose(  # rename epsilons on the right
                    epsilon_filter_fst(self.B)._compose(  # this FST carefully combines the special epsilons
                        other._augment_epsilon_transitions(1),
                    )
                )  # rename epsilons on the left
            )

    # TODO: add assertions for the 'bad' epsilon cases to ensure users aren't using this method incorrectly.
    # TODO: use lazy machine pattern
    def _compose(self, other):
        """
        Implements the on-the-fly composition of the FST `self` with the FST `other`.
        """

        C = FST()

        # index arcs in `other` to so that they are fast against later
        tmp = defaultdict(list)
        for i in other.states:
            for a, b, j in other.arcs(i):
                tmp[i, a].append((b, j))

        visited = set()
        stack = []

        # add initial states
        for P in self.I:
            for Q in other.I:
                PQ = (P, Q)
                C.add_I(PQ)
                visited.add(PQ)
                stack.append(PQ)

        # traverse the machine using depth-first search
        while stack:
            P, Q = PQ = stack.pop()

            # (q,p) is simultaneously a final state in the respective machines
            if P in self.F and Q in other.F:
                C.add_F(PQ)
                # Note: final states are not necessarily absorbing -> fall thru

            # Arcs of the composition machine are given by a cross-product-like
            # construction that matches an arc labeled `a:b` with an arc labeled
            # `b:c` in the left and right machines respectively.
            for a, b, Pʼ in self.arcs(P):
                for c, Qʼ in tmp[Q, b]:
                    assert b != EPSILON

                    PʼQʼ = (Pʼ, Qʼ)

                    C.add_arc(PQ, a, c, PʼQʼ)

                    if PʼQʼ not in visited:
                        stack.append(PʼQʼ)
                        visited.add(PʼQʼ)

        return C

    # TODO: use lazy pattern here too.
    def _augment_epsilon_transitions(self, idx):
        """
        Augments the FST by changing the appropriate epsilon transitions to
        epsilon_1 or epsilon_2 transitions to be able to perform the composition
        correctly.  See Fig. 7 on p. 17 of Mohri, "Weighted Automata Algorithms".

        Args: `idx` (int): 1 if the FST is the first one in the composition, 2 otherwise.
        """
        assert idx in [0, 1]

        T = self.spawn(keep_init=True, keep_stop=True)

        for i in self.states:
            if idx == 0:
                T.add_arc(i, EPSILON, ε_1, i)
            else:
                T.add_arc(i, ε_2, EPSILON, i)
            for a, b, j in self.arcs(i):
                if idx == 0 and b == EPSILON:
                    b = ε_2
                elif idx == 1 and a == EPSILON:
                    a = ε_1
                T.add_arc(i, a, b, j)

        return T

    @classmethod
    def diag(cls, fsa):
        """
        Convert a FSA A to diagonal relation T wich that T(x,x) = A(x) for all strings x.
        """
        assert isinstance(fsa, FSA), type(fsa)
        fst = cls()
        for i, a, j in fsa.arcs():
            fst.add_arc(i, a, a, j)
        for i in fsa.I:
            fst.add_I(i)
        for i in fsa.F:
            fst.add_F(i)
        return fst

    def paths(self):
        "Enumerate paths in the FST using breadth-first order."
        worklist = deque()
        for i in self.I:
            worklist.append((i,))
        while worklist:
            path = worklist.popleft()
            i = path[-1]
            if self.is_final(i):
                yield path
            for a, b, j in self.arcs(i):
                worklist.append((*path, (a,b), j))
        return m


def epsilon_filter_fst(Sigma):
    """
    Returns the 3-state epsilon-filtered FST, that is used in to avoid
    epsilon-related ambiguity when composing WFST with epsilons.
    """

    F = FST()

    F.add_I(0)

    for a in Sigma:
        F.add_arc(0, a, a, 0)
        F.add_arc(1, a, a, 0)
        F.add_arc(2, a, a, 0)

    F.add_arc(0, ε_2, ε_1, 0)
    F.add_arc(0, ε_1, ε_1, 1)
    F.add_arc(0, ε_2, ε_2, 2)

    F.add_arc(1, ε_1, ε_1, 1)
    F.add_arc(2, ε_2, ε_2, 2)

    F.add_F(0)
    F.add_F(1)
    F.add_F(2)

    return F
