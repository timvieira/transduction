from collections import defaultdict, deque
from functools import cached_property
from itertools import zip_longest

from transduction.fsa import FSA, EPSILON, _render_graphviz

from arsenal import Integerizer


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
        self.start = set()
        self.stop = set()
        self._arcs_i = None      # lazy arc indexes (built on first arcs() call)
        self._arcs_ix = None
        for i in start: self.add_start(i)
        for i in stop: self.add_stop(i)
        for i,a,b,j in arcs: self.add_arc(i,a,b,j)

    # Deprecated aliases
    @property
    def I(self):
        return self.start

    @property
    def F(self):
        return self.stop

    def __repr__(self):
        return f'{__class__.__name__}({len(self.states)} states)'

    def __str__(self):
        output = []
        output.append('{')
        for p in self.states:
            output.append(f'  {p} \t\t({p in self.start}, {p in self.stop})')
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
        return i in self.stop

    def add_arc(self, i, a, b, j):  # pylint: disable=arguments-renamed
        self.states.add(i)
        self.states.add(j)
        self.delta[i][a][b].add(j)   # TODO: change this data structure to separate a and b.
        self.A.add(a)
        self.B.add(b)
        self._arcs_i = None   # invalidate arc indexes

    def add_start(self, q):
        self.states.add(q)
        self.start.add(q)

    def add_stop(self, q):
        self.states.add(q)
        self.stop.add(q)

    # Deprecated aliases
    add_I = add_start
    add_F = add_stop

    def ensure_arc_indexes(self):
        """Build secondary arc indexes for O(1) lookup by various key combinations.

        Naming scheme: ``index_{keys}_{values}`` where i=source state,
        x=input label, y=output label, j=destination state.  For example,
        ``index_iy_xj[(i, y)]`` returns ``{(x, j), ...}`` — all arcs from
        state ``i`` with output label ``y``.
        """
        if hasattr(self, 'index_iy_xj'):
            return
        index_iy_xj = {}
        index_i_xj = {}
        index_ix_j = {}
        index_ixy_j = {}
        for i in self.states:
            for x, y, j in self.arcs(i):
                index_iy_xj.setdefault((i, y), set()).add((x, j))
                index_i_xj.setdefault(i, set()).add((x, j))
                index_ix_j.setdefault((i, x), set()).add(j)
                index_ixy_j.setdefault((i, x, y), set()).add(j)
        self.index_iy_xj = index_iy_xj
        self.index_i_xj = index_i_xj
        self.index_ix_j = index_ix_j
        self.index_ixy_j = index_ixy_j

    def _build_arc_index(self):
        """Build flat arc indexes for O(1) lookup, called lazily on first arcs() call."""
        # state → tuple[(input, output, dest), ...]
        arcs_i = {}
        # (state, input) → tuple[(output, dest), ...]
        arcs_ix = {}
        for i, d in self.delta.items():
            all_arcs = []
            for a, A in d.items():
                by_input = []
                for b, B in A.items():
                    for j in B:
                        all_arcs.append((a, b, j))
                        by_input.append((b, j))
                arcs_ix[(i, a)] = tuple(by_input)
            arcs_i[i] = tuple(all_arcs)
        self._arcs_i = arcs_i
        self._arcs_ix = arcs_ix

    def arcs(self, i, x=None):
        """Iterate over arcs from state ``i``.

        With ``x=None``, yields ``(a, b, j)`` triples for all arcs from ``i``.
        With ``x`` provided, yields ``(b, j)`` pairs for arcs with input label ``x``.
        """
        if self._arcs_i is None:
            self._build_arc_index()
        if x is None:
            return self._arcs_i.get(i, ())
        else:
            return self._arcs_ix.get((i, x), ())

    def rename(self, f):
        "Note: If `f` is not bijective, states may merge."
        m = self.spawn()
        for i in self.start:
            m.add_start(f(i))
        for i in self.stop:
            m.add_stop(f(i))
        for i in self.states:
            for a, b, j in self.arcs(i):
                m.add_arc(f(i), a, b, f(j))
        return m

    def map_labels(self, f):
        "Transform arc labels by applying f(a, b) -> (a', b') to each arc."
        m = self.spawn(keep_init=True, keep_stop=True)
        for i in self.states:
            for a, b, j in self.arcs(i):
                a2, b2 = f(a, b)
                m.add_arc(i, a2, b2, j)
        return m

    def renumber(self):
        return self.rename(Integerizer())

    def spawn(self, *, keep_init=False, keep_arcs=False, keep_stop=False):
        m = self.__class__()
        if keep_init:
            for q in self.start:
                m.add_start(q)
        if keep_arcs:
            for i in self.states:
                for a, b, j in self.arcs(i):
                    m.add_arc(i, a, b, j)
        if keep_stop:
            for q in self.stop:
                m.add_stop(q)
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
        return _render_graphviz(
            self.states, self.start, self.stop,
            arc_iter=lambda i: (((a, b), j) for a, b, j in self.arcs(i)),
            fmt_node=fmt_node, fmt_edge=fmt_edge, sty_node=sty_node,
        )

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
        m.add_start(xs[:0])
        for i in range(len(xs)):
            m.add_arc(xs[:i], xs[i], xs[i], xs[:i+1])
        m.add_stop(xs)
        return m

    @staticmethod
    def from_pairs(pairs):
        p = FST()
        p.add_start(0)
        p.add_stop(1)
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
        for i in self.start:
            A.add_start(i)
        for i in self.stop:
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
            m.add_start(gensym(i))
        for i,a,j in other.arcs():
            m.add_arc(gensym(i), a, EPSILON, gensym(j))
        for j in other.stop:
            m.add_arc(gensym(j), EPSILON, marker, gensym(None))
        m.add_stop(gensym(None))

        return m

    @cached_property
    def T(self):
        "transpose swap left <-> right"
        return self.map_labels(lambda a, b: (b, a))

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
        for P in self.start:
            for Q in other.start:
                PQ = (P, Q)
                C.add_start(PQ)
                visited.add(PQ)
                stack.append(PQ)

        # traverse the machine using depth-first search
        while stack:
            P, Q = PQ = stack.pop()

            # (q,p) is simultaneously a final state in the respective machines
            if P in self.stop and Q in other.stop:
                C.add_stop(PQ)
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
            fst.add_start(i)
        for i in fsa.stop:
            fst.add_stop(i)
        return fst

    def paths(self):
        "Enumerate paths in the FST using breadth-first order."
        worklist = deque()
        for i in self.start:
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
        worklist.extend([(i, '', '') for i in self.start])
        while worklist:
            (i, xs, ys) = worklist.popleft()
            if self.is_final(i):
                yield xs, ys
            if len(xs) >= max_length or len(ys) >= max_length:
                continue
            for x, y, j in self.arcs(i):
                worklist.append((j, xs + x, ys + y))

    def transduce(self, input_seq):
        """Transduce input_seq through this FST via BFS NFA simulation.
        Returns one accepting output tuple, or raises ValueError if no
        accepting path exists.
        """
        queue = deque()
        parent = {}
        for s in self.start:
            key = (s, 0)
            parent[key] = None
            queue.append(key)
        while queue:
            state, pos = queue.popleft()
            if pos == len(input_seq) and state in self.stop:
                output = []
                k = (state, pos)
                while parent[k] is not None:
                    prev, out_sym = parent[k]
                    if out_sym != EPSILON:
                        output.append(out_sym)
                    k = prev
                output.reverse()
                return tuple(output)
            for a, b, j in self.arcs(state):
                if a == EPSILON:
                    new_pos = pos
                elif pos < len(input_seq) and a == input_seq[pos]:
                    new_pos = pos + 1
                else:
                    continue
                key = (j, new_pos)
                if key not in parent:
                    parent[key] = ((state, pos), b)
                    queue.append(key)
        raise ValueError("No accepting path found")

    def is_functional(self):
        """Check whether this FST defines a (partial) function.

        An FST is functional if every input string maps to at most one
        output string.  Returns ``(True, None)`` if functional, or
        ``(False, (x, y1, y2))`` with a witness input ``x`` that produces
        two distinct outputs ``y1 != y2``.

        Uses a product construction: two copies of the trimmed FST run
        on the same input while tracking the output-buffer difference.
        Terminates on all finite-state transducers (no length bound
        needed) by exploiting the twinning property: for a functional
        transducer the delay is bounded, and for a non-functional one
        we detect the pumping cycle.
        """
        t = self.trim()
        if not t.states:
            return (True, None)

        # --- Phase 1: Build product automaton (q1, q2) sharing input ---
        # Forward reachability from all start pairs.
        product_arcs = {}    # (q1,q2) -> list of (a, b1, b2, j1, j2)
        product_final = set()

        fwd = set()
        queue = deque()
        for s1 in t.start:
            for s2 in t.start:
                pair = (s1, s2)
                if pair not in fwd:
                    fwd.add(pair)
                    queue.append(pair)

        while queue:
            q1, q2 = pair = queue.popleft()
            if q1 in t.stop and q2 in t.stop:
                product_final.add(pair)
            arcs = []
            for a, b1, j1 in t.arcs(q1):
                for a2, b2, j2 in t.arcs(q2):
                    if a != a2:
                        continue
                    arcs.append((a, b1, b2, j1, j2))
                    dest = (j1, j2)
                    if dest not in fwd:
                        fwd.add(dest)
                        queue.append(dest)
            product_arcs[pair] = arcs

        if not product_final:
            return (True, None)

        # --- Phase 2: Backward reachability → co-accessible states ---
        rev = defaultdict(set)
        for pair, arcs in product_arcs.items():
            for _, _, _, j1, j2 in arcs:
                rev[(j1, j2)].add(pair)

        co = set(product_final)
        queue = deque(product_final)
        while queue:
            pair = queue.popleft()
            for prev in rev[pair]:
                if prev not in co:
                    co.add(prev)
                    queue.append(prev)

        # Only explore states on a start→final path in the product.
        trimmed_product = fwd & co

        # --- Phase 3: BFS with output-delay tracking ---
        # For a functional transducer the delay is bounded (twinning
        # property).  Bound B: if the delay exceeds |trimmed_product|*M,
        # some (q1,q2) pair was visited with strictly growing delay,
        # meaning a cycle is pumping.  Since every state in
        # trimmed_product is co-accessible, that cycle lies on a path
        # to acceptance → non-functional.
        M = max(
            (len(b) for q in t.states for _, b, _ in t.arcs(q) if b != EPSILON),
            default=1,
        )
        delay_bound = len(trimmed_product) * max(M, 1)

        visited = set()
        worklist = deque()
        exceeded = False

        for s1 in t.start:
            for s2 in t.start:
                if (s1, s2) not in trimmed_product:
                    continue
                key = (s1, s2, 0, '')
                if key not in visited:
                    visited.add(key)
                    worklist.append((s1, s2, 0, '', '', '', ''))

        while worklist:
            q1, q2, side, buf, x, y1, y2 = worklist.popleft()

            if (q1, q2) in product_final and (side != 0 or buf != ''):
                return (False, (x, y1, y2))

            if len(buf) > delay_bound:
                exceeded = True
                continue

            for a, b1, b2, j1, j2 in product_arcs.get((q1, q2), ()):
                if (j1, j2) not in trimmed_product:
                    continue
                new_side, new_buf = _advance_buf(side, buf, b1, b2)
                key = (j1, j2, new_side, new_buf)
                if key in visited:
                    continue
                visited.add(key)
                worklist.append((j1, j2, new_side, new_buf,
                                 x + a, y1 + b1, y2 + b2))

        if exceeded:
            # Delay exceeded bound on a co-accessible product state.
            # A cycle is pumping the delay on a path to acceptance
            # → non-functional, but the witness requires more pumping
            # than the bound allows.
            return (False, None)

        return (True, None)

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
            if q in t.stop:
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
                if q in t.stop:
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
        for s in t.start:
            d_s = delay[s]
            if len(d_s) == 0:
                result.add_start(s)
            else:
                # Prepend ε-input chain producing d(s)
                prev = fresh()
                result.add_start(prev)
                for k, sym in enumerate(d_s):
                    if k < len(d_s) - 1:
                        nxt = fresh()
                        result.add_arc(prev, EPSILON, sym, nxt)
                        prev = nxt
                    else:
                        result.add_arc(prev, EPSILON, sym, s)

        # --- 4. Final states ---
        for q in t.stop:
            result.add_stop(q)

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


def _advance_buf(side, buf, b1, b2):
    """Update the output-buffer difference after copy-1 emits b1 and copy-2 emits b2.

    The buffer tracks what one copy has emitted beyond the other:
      side=0, buf='' means the two copies are in sync.
      side=1, buf=s   means copy-1 is ahead by suffix s.
      side=2, buf=s   means copy-2 is ahead by suffix s.
    """
    # Build the full "ahead" string for each side
    if side == 0:
        ahead1, ahead2 = '', ''
    elif side == 1:
        ahead1, ahead2 = buf, ''
    else:
        ahead1, ahead2 = '', buf

    ahead1 += b1
    ahead2 += b2

    # Cancel common prefix
    n = min(len(ahead1), len(ahead2))
    common = 0
    for i in range(n):
        if ahead1[i] != ahead2[i]:
            break
        common = i + 1

    ahead1 = ahead1[common:]
    ahead2 = ahead2[common:]

    if ahead1:
        return (1, ahead1)
    elif ahead2:
        return (2, ahead2)
    else:
        return (0, '')


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


# Re-export universality API for backward compatibility
from transduction.universality import (   # noqa: F401
    check_all_input_universal,
    compute_ip_universal_states,
    UniversalityFilter,
)


def epsilon_filter_fst(Sigma):
    """
    Returns the 3-state epsilon-filtered FST, that is used in to avoid
    epsilon-related ambiguity when composing WFST with epsilons.
    """

    F = FST()

    F.add_start(0)

    for a in Sigma:
        F.add_arc(0, a, a, 0)
        F.add_arc(1, a, a, 0)
        F.add_arc(2, a, a, 0)

    F.add_arc(0, ε_2, ε_1, 0)
    F.add_arc(0, ε_1, ε_1, 1)
    F.add_arc(0, ε_2, ε_2, 2)

    F.add_arc(1, ε_1, ε_1, 1)
    F.add_arc(2, ε_2, ε_2, 2)

    F.add_stop(0)
    F.add_stop(1)
    F.add_stop(2)

    return F
