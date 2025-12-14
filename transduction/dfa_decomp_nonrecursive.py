from transduction import FSA, EPSILON
from transduction.eager_nonrecursive import LazyPrecoverNFA
from collections import deque


class NonrecursiveDFADecomp:

    def __init__(self, fst, target):
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        # Implementation note: this is a truncated representation of the
        # Precover(fst, target).  The recursive algorithm attempts to do
        # something differently where the automaton allows the target buffer to
        # grow without bound.  This works in a surprising number of cases, but
        # it can fail to terminate (e.g., on the `triplets_of_doom`).
        dfa = LazyPrecoverNFA(fst, target).det()

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
                if dfa.accepts_universal(i, self.source_alphabet):
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
        self.dfa = dfa
        self.target = target
        self.quotient = Q
        self.remainder = R
