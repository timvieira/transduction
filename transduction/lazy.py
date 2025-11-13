from collections import defaultdict, deque
from transduction.fsa import FSA, EPSILON


_frozenset = frozenset
class frozenset(_frozenset):
    "Same as frozenset, but with a nicer printing method."
    def __repr__(self):
        return '{%s}' % (','.join(str(x) for x in self))


class Lazy:

    #____________________________________________________________
    # Abstract interface

    def arcs(self, state):
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def is_final(self, state):
        raise NotImplementedError()

    #____________________________________________________________
    # Abstract class provides the methods below

    def start_at(self, s):
        return StartAt(self, s)

    def det(self):
        return LazyDeterminize(self)

    def epsremove(self):
        return EpsilonRemove(self)

    def materialize(self):
        "Materialized this automaton using a depth-first traversal from its initial states."
        m = FSA()
        worklist = deque()
        visited = set()
        for i in self.start():
            worklist.append(i)
            m.add_start(i)
        while worklist:
            i = worklist.popleft()
            if i in visited: continue
            visited.add(i)
            if self.is_final(i):
                m.add_stop(i)
            for a, j in self.arcs(i):
                worklist.append(j)
                m.add(i,a,j)
        return m

    # XXX: This method seems like it could be improved with caching since it's
    # likely that we will ask about later states (need to think about that
    # actually).
    def accepts_universal(self, state, alphabet):
        "[True/False] This state accepts the universal language (alphabet$^*$)."
        #
        # Rationale: a DFA accepts `alphabet`$^*$ iff all reachable states are
        # accepting and complete (i.e., has a transition for each symbol in
        # `alphabet`).
        #
        # Warning: If the reachable subset automaton is infinite, the search may
        # not terminate (as expected, NFA universality is PSPACE-complete in
        # general), but in many practical FSAs this halts quickly.
        #
        dfa = LazyDeterminize(self.start_at(state))

        visited = set()
        worklist = deque()

        # DFA start state
        for i in dfa.start():
            visited.add(i)
            worklist.append(i)

        assert len(worklist) == 1

        while worklist:
            i = worklist.popleft()

            # All-final check in the DFA view
            if not dfa.is_final(i):
                return False

            # Build a symbol-to-destination mapping
            dest = dict(dfa.arcs(i))

            # Completeness on Σ
            for a in alphabet:
                # if we're missing an arc labeled `a` in state `i`, then state
                # `i` is not universal!  Moreover, `state` is not universal.
                if a not in dest:
                    return False
                j = dest[a]
                if j not in visited:
                    visited.add(j)
                    worklist.append(j)

        return True


class EpsilonRemove(Lazy):

    def __init__(self, fsa):
        self.fsa = fsa

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

    def _closure(self, i):
        pushed = {i}
        worklist = {i}
        while worklist:
            i = worklist.pop()
            yield i
            for a, j in self.fsa.arcs(i):
                if a == EPSILON and j not in pushed:
                    worklist.add(j)
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
