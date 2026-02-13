from collections import defaultdict, deque
from transduction.fsa import FSA, EPSILON, frozenset
from arsenal import Integerizer


def is_universal(dfa, state, alphabet):
    """Check whether ``state`` accepts the universal language (alphabet*)."

    A DFA state accepts alphabet* iff all reachable states are accepting
    and complete (i.e., have a transition for each symbol in ``alphabet``).

    Warning: If the reachable subset automaton is infinite, the search may
    not terminate (as expected, NFA universality is PSPACE-complete in
    general), but in many practical FSAs this halts quickly.
    """
    visited = set()
    worklist = deque()

    visited.add(state)
    worklist.append(state)

    while worklist:
        i = worklist.popleft()

        if not dfa.is_final(i):
            return False

        dest = dict(dfa.arcs(i))

        for a in alphabet:
            if a not in dest:
                return False
            j = dest[a]
            if j not in visited:
                visited.add(j)
                worklist.append(j)

    return True


class Lazy:
    """Abstract base class for lazy (on-demand) automata.

    Subclasses implement the four abstract methods below to define an
    automaton whose states and arcs are computed on the fly rather than
    stored in memory.  The concrete methods on this class (``materialize``,
    ``det``, ``epsremove``, etc.) compose lazy automata without eagerly
    expanding them.
    """

    #____________________________________________________________
    # Abstract interface

    def arcs(self, state):
        """Yield ``(label, dest)`` pairs for all outgoing arcs from ``state``."""
        raise NotImplementedError()

    def arcs_x(self, state, x):
        """Yield destination states reachable from ``state`` on input ``x``.

        Default implementation filters ``arcs()``; subclasses may override
        for efficiency.
        """
        import warnings; warnings.warn('using slow implementation of arcs_x')
        for X, j in self.arcs(state):
            if X == x:
                yield j

    def start(self):
        """Yield the start state(s) of this automaton."""
        raise NotImplementedError()

    def is_final(self, state):
        """Return True if ``state`` is an accepting state."""
        raise NotImplementedError()

    #____________________________________________________________
    # Abstract class provides the methods below

    def start_at(self, s):
        return StartAt(self, s)

    def det(self):
        return LazyDeterminize(self)

    def min(self):
        "Warning: not lazy."
        return self.materialize().min()

    def epsremove(self):
        return EpsilonRemove(self)

    def materialize(self, max_steps=float('inf')):
        "Materialized this automaton using a depth-first traversal from its initial states."
        m = FSA()
        worklist = deque()
        visited = set()
        for i in self.start():
            worklist.append(i)
            visited.add(i)
            m.add_start(i)
        steps = 0
        while worklist:
            steps += 1
            if steps >= max_steps: break
            i = worklist.popleft()
            if self.is_final(i):
                m.add_stop(i)
            for a, j in self.arcs(i):
                if j not in visited:
                    visited.add(j)
                    worklist.append(j)
                m.add(i,a,j)
        return m

    def accepts_universal(self, state, alphabet):
        """Check whether ``state`` accepts the universal language (alphabet*).

        Determinizes the NFA starting from ``state``, then delegates to the
        shared ``is_universal`` BFS.
        """
        dfa = LazyDeterminize(self.start_at(state))
        [start] = list(dfa.start())
        return is_universal(dfa, start, alphabet)

    def cache(self):
        return Cached(self)

    def renumber(self):
        return Renumber(self)


class EpsilonRemove(Lazy):
    """Lazy epsilon removal: wraps an NFA and presents an equivalent
    epsilon-free automaton by expanding epsilon closures on demand.

    For each state, non-epsilon arcs are followed and then the epsilon
    closure of the destination is computed (and cached).  Start states
    and finality are similarly lifted through epsilon closures.
    """

    def __init__(self, fsa):
        self.fsa = fsa
        self._closure_cache = {}    # NOTE: caching should be optional/configurable

    def start(self):
        for i in self.fsa.start():
            yield from self._closure(i)

    def is_final(self, i):
        return any(self.fsa.is_final(j) for j in self._closure(i))

    def arcs(self, i):
        for a, j in self.fsa.arcs(i):
            if a == EPSILON: continue
            for k in self._closure(j):
                yield a, k

    def arcs_x(self, i, x):
        if x == EPSILON: return
        for j in self.fsa.arcs_x(i, x):
            for k in self._closure(j):
                yield k

    def _closure(self, i):
        value = self._closure_cache.get(i)
        if value is None:
            value = set(self.__closure(i))
            self._closure_cache[i] = value
        return value

    def __closure(self, i):
        pushed = {i}
        worklist = [i]
        while worklist:
            i = worklist.pop()
            yield i
            for j in self.fsa.arcs_x(i, EPSILON):
                if j not in pushed:
                    worklist.append(j)
                    pushed.add(j)


class LazyDeterminize(Lazy):

    def __init__(self, fsa):
        self.fsa = fsa.epsremove()

    def start(self):
        yield frozenset(self.fsa.start())

    def is_final(self, Q):
        return any(self.fsa.is_final(i) for i in Q)

    def arcs(self, Q):
        tmp = defaultdict(set)
        for i in Q:
            for a, j in self.fsa.arcs(i):
                tmp[a].add(j)
        for a, j in tmp.items():
            yield a, frozenset(j)

    def arcs_x(self, Q, x):
        result = frozenset(j for i in Q for j in self.fsa.arcs_x(i, x))
        if result:
            yield result


class Cached(Lazy):
    """Caches arcs and finality per state -- a lazy version of materialize."""

    def __init__(self, base):
        self.base = base
        self._arcs_cache = {}
        self._final_cache = {}

    def start(self):
        return self.base.start()

    def arcs(self, state):
        if state not in self._arcs_cache:
            self._arcs_cache[state] = list(self.base.arcs(state))
        return self._arcs_cache[state]

    def is_final(self, state):
        if state not in self._final_cache:
            self._final_cache[state] = self.base.is_final(state)
        return self._final_cache[state]


class StartAt(Lazy):
    "Clone base FSA so that it starts at s."

    def __init__(self, base, s):
        self.base = base
        self.s = s

    def start(self):
        yield self.s

    def is_final(self, i):
        return self.base.is_final(i)

    def arcs(self, i):
        return self.base.arcs(i)

    def arcs_x(self, i, x):
        return self.base.arcs_x(i, x)


class Renumber(Lazy):

    def __init__(self, fsa):
        self.fsa = fsa
        self.m = Integerizer()

    def start(self):
        for i in self.fsa.start():
            yield self.m(i)

    def is_final(self, i):
        return self.fsa.is_final(self.m[i])

    def arcs(self, i):
        for a, j in self.fsa.arcs(self.m[i]):
            yield a, self.m(j)

    def arcs_x(self, i, x):
        for j in self.fsa.arcs_x(self.m[i], x):
            yield self.m(j)


class LazyWrapper(Lazy):

    def __init__(self, base):
        self.base = base

    def start(self):
        return iter(self.base.start)

    def is_final(self, i):
        return self.base.is_final(i)

    def arcs(self, i):
        return self.base.arcs(i)

    def arcs_x(self, i, x):
        return self.base.arcs_x(i, x)



#_______________________________________________________________________________
# EXPERIMENTAL SCC-based epsilon closure; written by ChatGPT but not tested extensively

if 0:
    from collections import defaultdict

    class LazyEpsSCCIndex:
        """
        Incremental Tarjan SCC discovery over the ε-subgraph, driven lazily by ensure(i).
        Discovers SCCs only in the ε-reachable region from queried states.
        """

        def __init__(self, fsa):
            self.fsa = fsa

            # Tarjan per-state metadata (only for discovered states)
            self._index = 0
            self._idx = {}          # state -> dfs index
            self._low = {}          # state -> lowlink
            self._onstack = set()
            self._stack = []

            # SCC data (only for discovered SCCs)
            self._cid = 0
            self.comp = {}          # state -> component id
            self.members = {}       # component id -> frozenset(states)
            self.eps_succ = defaultdict(set)   # component id -> set(component id)

            # For building eps_succ when target component not known yet
            self._pending_to_state = defaultdict(set)  # state -> set(source component ids)

            # Optional per-component aggregates you’ll likely want
            self.comp_is_final = {}  # component id -> bool

        def ensure(self, i):
            """Ensure Tarjan has indexed i and all ε-reachable states from i."""
            if i in self._idx:
                return
            self._strongconnect(i)

        def _eps_succ_states(self, i):
            # Iterates ε successors lazily from the underlying FSA.
            return self.fsa.arcs_x(i, EPSILON)

        def _strongconnect(self, v):
            # Standard Tarjan, but driven from v only.
            self._idx[v] = self._index
            self._low[v] = self._index
            self._index += 1

            self._stack.append(v)
            self._onstack.add(v)

            for w in self._eps_succ_states(v):
                if w not in self._idx:
                    self._strongconnect(w)
                    self._low[v] = min(self._low[v], self._low[w])
                elif w in self._onstack:
                    self._low[v] = min(self._low[v], self._idx[w])

            # If v is a root node, pop the stack and generate an SCC
            if self._low[v] == self._idx[v]:
                nodes = []
                while True:
                    w = self._stack.pop()
                    self._onstack.remove(w)
                    nodes.append(w)
                    if w == v:
                        break

                cid = self._cid
                self._cid += 1

                # Assign component id to members
                for s in nodes:
                    self.comp[s] = cid

                nodes_fs = frozenset(nodes)
                self.members[cid] = nodes_fs

                # Aggregate final-ness (cheap, and helps is_final)
                self.comp_is_final[cid] = any(self.fsa.is_final(s) for s in nodes_fs)

                # Now that cid is known, resolve any pending incoming edges
                for s in nodes_fs:
                    if s in self._pending_to_state:
                        for src_cid in self._pending_to_state.pop(s):
                            if src_cid != cid:
                                self.eps_succ[src_cid].add(cid)

                # Record outgoing component edges by scanning ε-arcs from SCC members.
                # Targets might not yet have a component id at this exact moment if they’re
                # not in the explored region — but in Tarjan DFS they *will* have been reached.
                # Still, handle defensively via pending.
                for s in nodes_fs:
                    for t in self._eps_succ_states(s):
                        tgt_cid = self.comp.get(t)
                        if tgt_cid is not None:
                            if tgt_cid != cid:
                                self.eps_succ[cid].add(tgt_cid)
                        else:
                            # Haven't assigned t to a component yet; remember cid wants an edge to it.
                            self._pending_to_state[t].add(cid)


    class LazySCCDAGClosure:
        def __init__(self, scc_index: LazyEpsSCCIndex):
            self.I = scc_index
            self._cache = {}  # cid -> frozenset[cid]

        def closure_cid(self, cid):
            got = self._cache.get(cid)
            if got is not None:
                return got
            out = {cid}
            for nxt in self.I.eps_succ.get(cid, ()):
                out |= self.closure_cid(nxt)
            out = frozenset(out)
            self._cache[cid] = out
            return out


    class EpsilonRemove(Lazy):

        def __init__(self, fsa):
            self.fsa = fsa
            self.I = LazyEpsSCCIndex(fsa)
            self.C = LazySCCDAGClosure(self.I)

        def _closure(self, i):
            self.I.ensure(i)                 # discover SCCs only as needed
            cid = self.I.comp[i]             # now i has a component id
            for u in self.C.closure_cid(cid):
                yield from self.I.members[u]

        def start(self):
            for i in self.fsa.start():
                yield from self._closure(i)

        def is_final(self, i):
            self.I.ensure(i)
            cid = self.I.comp[i]
            return any(self.I.comp_is_final[u] for u in self.C.closure_cid(cid))

        def arcs(self, i):
            for a, j in self.fsa.arcs(i):
                if a == EPSILON:
                    continue
                yield from ((a, k) for k in self._closure(j))

        def arcs_x(self, i, x):
            if x == EPSILON:
                return
            for j in self.fsa.arcs_x(i, x):
                yield from self._closure(j)
