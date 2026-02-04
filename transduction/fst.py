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


eps = EPSILON


class FST:
    def __init__(self, start=(), arcs=(), stop=()):
        self.A = set()
        self.B = set()
        self.states = set()
        self.delta = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.I = set()
        self.F = set()
        for i in start: self.add_I(i)
        for i in stop: self.add_F(i)
        for i,a,b,j in arcs: self.add_arc(i,a,b,j)

    @property
    def start(self):
        return self.I

    @property
    def stop(self):
        return self.F

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

    # TODO: test this method
    def dump(self):
        print('m = FST()')
        for i in self.start:
            print(f'm.add_start({i!r})')
        for i in self.stop:
            print(f'm.add_stop({i!r})')
        for i in self.states:
            for x, y, j in self.arcs(i):
                print(f'm.add_arc({i!r}, {x!r}, {y!r}, {j!r})')

    def is_final(self, i):
        return i in self.F

    def add_arc(self, i, a, b, j):  # pylint: disable=arguments-renamed
        self.states.add(i)
        self.states.add(j)
        self.delta[i][a][b].add(j)   # TODO: change this data structure to separate a and b.
        self.A.add(a)
        self.B.add(b)

    def add_I(self, q):
        self.states.add(q)
        self.I.add(q)

    def add_F(self, q):
        self.states.add(q)
        self.F.add(q)

    # aliases
    add_start = add_I
    add_stop = add_F

    def arcs(self, i, x=None):
        if x is None:
            for a, A in self.delta[i].items():
                for b, B in A.items():
                    for j in B:
                        yield a, b, j
        else:
            A = self.delta[i][x]
            for b, B in A.items():
                for j in B:
                    yield b, j

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

    def renumber(self):
        return self.rename(Integerizer())

    def spawn(self, *, keep_init=False, keep_arcs=False, keep_stop=False):
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
        if not self.states:
            return {'text/html': '<center>∅</center>'}
        return self.graphviz()._repr_mimebundle_(*args, **kwargs)

    def graphviz(
        self,
        fmt_node=lambda x: x,
        fmt_edge=lambda i, a, j: f'{str(a[0] or "ε")}:{str(a[1] or "ε")}' if a[0] != a[1] else str(a[0]),
        sty_node=lambda i: {},
    ):

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
            g.node(str(f(i)), label=html.escape(str(fmt_node(i))), **sty)

        # Collect parallel-edge labels by (i, j)
        by_pair = defaultdict(list)
        for i in self.states:
            for a, b, j in self.arcs(i):
                lbl = html.escape(str(fmt_edge(i, (a, b), j)))
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

    # TODO: this function needs testing
    def make_total(self, marker):
        "If `self` is a partial function, this method will make it total by extending the range with a failure `marker`."
        assert marker not in self.B

        d = (self @ FSA.from_strings(self.B - {EPSILON}).star().min()).project(0)
        other = d.invert(self.A - {EPSILON}).min()

        # TODO: this is not guaranteed to be renamed apart
        def gensym(i): return ('other', i)
        m = self.spawn(keep_arcs=True, keep_init=True, keep_stop=True)

        # copy arcs from `other` such that they read the same symbol, but now
        # emit the empty string.  However, at the end of we generate a `marker`
        # symbol and terminate.
        for i in other.start:
            m.add_I(gensym(i))
        for i,a,j in other.arcs():
            m.add_arc(gensym(i), a, EPSILON, gensym(j))
        for j in other.stop:
            m.add_arc(gensym(j), EPSILON, marker, gensym(None))
        m.add_F(gensym(None))

        return m

    @cached_property
    def T(self):
        "transpose swap left <-> right"
        T = self.spawn()
        for i in self.states:
            for a, b, j in self.arcs(i):
                T.add_arc(i, b, a, j)  # pylint: disable=W1114
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
                )  # rename epsilons on the left
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

        Args: `idx` (int): 0 if the FST is the first one in the composition, 1 otherwise.
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
        for i in fsa.start:
            fst.add_I(i)
        for i in fsa.stop:
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

    def relation(self, max_length):
        "Enumerate string pairs in the relation of this FST up to length `max_length`."
        worklist = deque()
        worklist.extend([(i, '', '') for i in self.I])
        while worklist:
            (i, xs, ys) = worklist.popleft()
            if self.is_final(i):
                yield xs, ys
            if len(xs) >= max_length or len(ys) >= max_length:
                continue
            for x, y, j in self.arcs(i):
                worklist.append((j, xs + x, ys + y))

    def trim(self):
        """
        Return a new FST containing only the states and arcs lying on
        some start → stop path.
        """

        trimmed_states = self.reachable() & self.coreachable()

        # ---- collect arcs within trimmed states ----
        trimmed = FST(
            start=self.start & trimmed_states,
            stop=self.stop & trimmed_states,
        )
        for q in trimmed_states:
            for x, y, dst in self.arcs(q):
                if dst in trimmed_states:
                    trimmed.add_arc(q, x, y, dst)

        return trimmed

    def push_labels(self):
        """Push output labels toward initial states to reduce output delay.

        Returns a new FST that realizes the same relation but with output
        produced as early as possible.  This reduces powerset state diversity
        during decomposition by advancing the buffer position sooner.
        """
        t = self.trim()
        if not t.states:
            return t

        # --- 1. Compute output delays via least fixpoint ---
        # d(q) is the longest common prefix of all output strings reachable
        # from q to any final state.  We use None as "unconstrained" (top of
        # lattice) — meaning we haven't yet seen any path from q.

        delay = {}
        for q in t.states:
            if q in t.F:
                delay[q] = ()        # final state: empty path contributes ε
            else:
                delay[q] = None      # unconstrained until resolved

        # Build reverse adjacency so we can propagate backwards
        reverse_adj = defaultdict(list)   # q' -> list of (q, y_tuple)
        for q in t.states:
            for x, y, qp in t.arcs(q):
                y_tuple = (y,) if y != EPSILON else ()
                reverse_adj[qp].append((q, y_tuple))

        # Iterative fixpoint
        changed = True
        while changed:
            changed = False
            for q in t.states:
                # d(q) = LCP({concat(y, d(q')) for each arc (q, x, y, q')})
                new_val = None
                for x, y, qp in t.arcs(q):
                    y_tuple = (y,) if y != EPSILON else ()
                    d_qp = delay[qp]
                    if d_qp is None:
                        continue     # unconstrained successor, skip
                    candidate = y_tuple + d_qp
                    if new_val is None:
                        new_val = candidate
                    else:
                        new_val = _lcp_pair(new_val, candidate)
                # Also, if q is final, the empty path contributes ε
                if q in t.F:
                    if new_val is None:
                        new_val = ()
                    else:
                        new_val = _lcp_pair(new_val, ())

                if new_val is not None and new_val != delay[q]:
                    delay[q] = new_val
                    changed = True

        # Finalize any remaining None to ()
        for q in t.states:
            if delay[q] is None:
                delay[q] = ()

        # --- 2. Transform arcs ---
        result = FST()
        counter = [0]
        tag = object()   # unique per call so repeated pushes don't collide

        def fresh():
            counter[0] += 1
            return ('__push__', tag, counter[0])

        for q in t.states:
            for x, y, qp in t.arcs(q):
                y_tuple = (y,) if y != EPSILON else ()
                full_output = y_tuple + delay[qp]
                new_output = _strip_prefix(delay[q], full_output)

                if len(new_output) == 0:
                    result.add_arc(q, x, EPSILON, qp)
                elif len(new_output) == 1:
                    result.add_arc(q, x, new_output[0], qp)
                else:
                    # Multi-symbol output: insert ε-input intermediates
                    prev = q
                    for k, sym in enumerate(new_output):
                        if k == 0:
                            nxt = fresh()
                            result.add_arc(prev, x, sym, nxt)
                            prev = nxt
                        elif k < len(new_output) - 1:
                            nxt = fresh()
                            result.add_arc(prev, EPSILON, sym, nxt)
                            prev = nxt
                        else:
                            result.add_arc(prev, EPSILON, sym, qp)

        # --- 3. Handle start states ---
        for s in t.I:
            d_s = delay[s]
            if len(d_s) == 0:
                result.add_I(s)
            else:
                # Prepend ε-input chain producing d(s)
                prev = fresh()
                result.add_I(prev)
                for k, sym in enumerate(d_s):
                    if k < len(d_s) - 1:
                        nxt = fresh()
                        result.add_arc(prev, EPSILON, sym, nxt)
                        prev = nxt
                    else:
                        result.add_arc(prev, EPSILON, sym, s)

        # --- 4. Final states ---
        for q in t.F:
            result.add_F(q)

        return result.trim()

    # TODO: we can tighten up the code for reachable and coreachable
    # (e.g., no need to materialize `adj`).
    def reachable(self):
        reachable = set()
        dq = deque(self.start)
        while dq:
            s = dq.popleft()
            if s in reachable:
                continue
            reachable.add(s)
            for _,_,t in self.arcs(s):
                if t not in reachable:
                    dq.append(t)
        return reachable

    # TODO: should we just use `self.reverse().reachable()`?
    def coreachable(self):
        radj = defaultdict(set)
        for q in self.states:
            for _, _, dst in self.arcs(q):
                radj[dst].add(q)
        coreachable = set()
        dq = deque(self.stop)
        while dq:
            s = dq.popleft()
            if s in coreachable:
                continue
            coreachable.add(s)
            for t in radj[s]:
                if t not in coreachable:
                    dq.append(t)
        return coreachable

    #___________________________________________________________________________
    #

    def strongly_connected_components(self):
        """
        Return list of SCCs, each a list of states.
        """

        # Build adjacency
        adj = defaultdict(list)
        for q in self.states:
            for _, _, dst in self.arcs(q):
                adj[q].append(dst)

        index = {}
        lowlink = {}
        stack = []
        on_stack = set()
        current_index = [0]
        sccs = []

        def strongconnect(v):
            index[v] = current_index[0]
            lowlink[v] = current_index[0]
            current_index[0] += 1

            stack.append(v)
            on_stack.add(v)

            for w in adj.get(v, ()):
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)

        for v in self.states:
            if v not in index:
                strongconnect(v)

        return sccs


def _lcp_pair(a, b):
    """Longest common prefix of two tuples."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return a[:i]
    return a[:n]


def _strip_prefix(prefix, seq):
    """Remove prefix from seq, returning the remainder as a tuple."""
    assert seq[:len(prefix)] == prefix, f'{prefix} is not a prefix of {seq}'
    return seq[len(prefix):]


def check_all_input_universal(fst):
    """
    O(|arcs|) check: does the input projection of this FST accept Σ* from the
    start state?

    Works by checking that:
    1. The eps-closed start set contains a final state
    2. The start set has arcs for every source symbol (completeness)
    3. Every symbol's successor eps-closure contains the start set

    If (3) holds, all reachable DFA states of the input projection contain the
    start set, hence are all final and complete → the start state is universal.
    """
    source_alphabet = fst.A - {eps}
    if not source_alphabet:
        # Empty alphabet: universal iff some start state is final
        return any(fst.is_final(s) for s in fst.I)

    # Eps-close start states over input-side ε arcs
    def ip_eps_close(states):
        visited = set(states)
        worklist = deque(states)
        while worklist:
            s = worklist.popleft()
            for a, _b, j in fst.arcs(s):
                if a == eps and j not in visited:
                    visited.add(j)
                    worklist.append(j)
        return visited

    start_set = ip_eps_close(fst.I)

    # Must contain a final state
    if not any(fst.is_final(s) for s in start_set):
        return False

    # Group non-ε input arcs from start_set by input symbol
    by_symbol = defaultdict(set)
    for s in start_set:
        for a, _b, j in fst.arcs(s):
            if a != eps:
                by_symbol[a].add(j)

    # Must be complete on source alphabet
    if len(by_symbol) < len(source_alphabet):
        return False

    # Check that every symbol's successor eps-closure contains the start set
    for _sym, raw_dests in by_symbol.items():
        closed = ip_eps_close(raw_dests)
        if not start_set <= closed:
            return False

    return True


def compute_ip_universal_states(fst):
    """
    Greatest-fixpoint computation of ip-universal FST states.

    A state q is ip-universal if the input projection of the FST, started from
    eps_close({q}), accepts Sigma*. This is strictly more general than
    check_all_input_universal, which only checks the start set.

    Algorithm:
    1. Precompute eps_close({q}) for all FST states q
    2. Initialize candidates = set(fst.states)
    3. Iteratively remove states that violate universality:
       - eps_close({q}) must contain a final state
       - eps_close({q}) must have arcs for every symbol in Sigma
       - For each symbol, the successor eps-closure must contain >= 1 candidate
    4. Fixed point = set of ip-universal states
    """
    source_alphabet = fst.A - {eps}
    if not source_alphabet:
        return {q for q in fst.states if fst.is_final(q)}

    def ip_eps_close(states):
        visited = set(states)
        worklist = deque(states)
        while worklist:
            s = worklist.popleft()
            for a, _b, j in fst.arcs(s):
                if a == eps and j not in visited:
                    visited.add(j)
                    worklist.append(j)
        return frozenset(visited)

    # Precompute closures
    closures = {q: ip_eps_close({q}) for q in fst.states}

    # Precompute per-closure: successor sets by symbol
    # For each closure, for each symbol, the raw destinations (before eps-close)
    closure_symbol_succs = {}
    for q in fst.states:
        by_symbol = defaultdict(set)
        for s in closures[q]:
            for a, _b, j in fst.arcs(s):
                if a != eps:
                    by_symbol[a].add(j)
        closure_symbol_succs[q] = by_symbol

    candidates = set(fst.states)

    changed = True
    while changed:
        changed = False
        to_remove = set()
        for q in candidates:
            closure = closures[q]

            # Must contain a final state
            if not any(fst.is_final(s) for s in closure):
                to_remove.add(q)
                continue

            # Must be complete on source alphabet
            by_symbol = closure_symbol_succs[q]
            if not all(a in by_symbol for a in source_alphabet):
                to_remove.add(q)
                continue

            # For each symbol, successor eps-closure must contain a candidate
            ok = True
            for a in source_alphabet:
                succ_closure = ip_eps_close(by_symbol[a])
                if not (succ_closure & candidates):
                    ok = False
                    break
            if not ok:
                to_remove.add(q)

        if to_remove:
            candidates -= to_remove
            changed = True

    return frozenset(candidates)


class UniversalityFilter:
    """
    Encapsulates universality short-circuit optimizations:
    - Fast path: if check_all_input_universal, every final state is universal
    - ip-universal witness check via set intersection
    - Superset monotonicity: if S is universal, any S' ⊇ S is too
    - Subset monotonicity: if S is not universal, any S' ⊆ S isn't either
    - Fallback: BFS universality check on the DFA

    Monotonicity caches use element-indexed lookups rather than linear scans.
    """

    def __init__(self, fst, target, dfa, source_alphabet, *,
                 all_input_universal=None, witnesses=None):
        self.dfa = dfa
        self.source_alphabet = source_alphabet
        self.all_input_universal = (
            check_all_input_universal(fst) if all_input_universal is None
            else all_input_universal
        )
        if not self.all_input_universal:
            if witnesses is not None:
                self._witnesses = witnesses
            else:
                ip_univ = compute_ip_universal_states(fst)
                self._witnesses = frozenset((q, target) for q in ip_univ)
        # Element-indexed positive cache (known universal states).
        # _pos_index[element] = set of entry IDs whose stored set contains element.
        # A stored set u ⊆ dfa_state iff every element of u is in dfa_state,
        # i.e., the entry's hit count equals its size.
        self._pos_index = defaultdict(set)
        self._pos_sizes = {}   # entry_id -> len(stored set)
        self._pos_next = 0
        # Element-indexed negative cache (known non-universal states).
        # A stored set nu ⊇ dfa_state iff every element of dfa_state is in nu,
        # i.e., the intersection of entry-ID sets across all elements is non-empty.
        self._neg_index = defaultdict(set)
        self._neg_next = 0

    def _add_pos(self, s):
        eid = self._pos_next
        self._pos_next += 1
        self._pos_sizes[eid] = len(s)
        for e in s:
            self._pos_index[e].add(eid)

    def _add_neg(self, s):
        eid = self._neg_next
        self._neg_next += 1
        for e in s:
            self._neg_index[e].add(eid)

    def _has_pos_subset(self, dfa_state):
        """Is there a known-universal set u such that u ⊆ dfa_state?"""
        hits = {}
        for e in dfa_state:
            for eid in self._pos_index.get(e, ()):
                h = hits.get(eid, 0) + 1
                if h == self._pos_sizes[eid]:
                    return True
                hits[eid] = h
        return False

    def _has_neg_superset(self, dfa_state):
        """Is there a known-non-universal set nu such that dfa_state ⊆ nu?"""
        candidates = None
        for e in dfa_state:
            entry_ids = self._neg_index.get(e)
            if entry_ids is None:
                return False
            if candidates is None:
                candidates = set(entry_ids)
            else:
                candidates &= entry_ids
            if not candidates:
                return False
        return bool(candidates)

    def _bfs_universal(self, state):
        """BFS check: does `state` accept source_alphabet* in the DFA?"""
        visited = set()
        worklist = deque()
        visited.add(state)
        worklist.append(state)
        while worklist:
            i = worklist.popleft()
            if not self.dfa.is_final(i):
                return False
            dest = dict(self.dfa.arcs(i))
            for a in self.source_alphabet:
                if a not in dest:
                    return False
                j = dest[a]
                if j not in visited:
                    visited.add(j)
                    worklist.append(j)
        return True

    def is_universal(self, dfa_state):
        """Returns True/False for whether dfa_state accepts Sigma*."""

        # A state must be final to accept Sigma* (since epsilon is in Sigma*)
        if not self.dfa.is_final(dfa_state):
            return False

        # Fast path: all input universal means every final state is universal
        if self.all_input_universal:
            return True

        # ip-universal witness check: short-circuits on first common element
        if not self._witnesses.isdisjoint(dfa_state):
            self._add_pos(dfa_state)
            return True

        # Superset monotonicity: is dfa_state ⊇ some known-universal set?
        if self._has_pos_subset(dfa_state):
            return True

        # Subset monotonicity: is dfa_state ⊆ some known-non-universal set?
        if self._has_neg_superset(dfa_state):
            return False

        # BFS fallback
        result = self._bfs_universal(dfa_state)
        (self._add_pos if result else self._add_neg)(dfa_state)
        return result


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
