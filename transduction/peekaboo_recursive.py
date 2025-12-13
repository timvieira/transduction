from transduction.base import PrecoverDecomp
from transduction.eager_nonrecursive import Precover
from transduction.lazy import Lazy
from transduction.fsa import FSA, frozenset
from transduction.fst import EPSILON
#from transduction.eager_nonrecursive import EagerNonrecursive
#from transduction.lazy_recursive import LazyRecursive
#from transduction.lazy_nonrecursive import LazyNonrecursive

from arsenal import colors
from collections import deque

#_______________________________________________________________________________
#
# [2025-12-09 Tue] TRUNCATION STRATEGIES: COST-BENEFIT ANALYSIS - The strategy
#   that we have taken in the current implementation is truncate as early as
#   possible - this minimizes the work in the current iteration.  However, it
#   might lead to more work in a later iteration because more nodes are marked
#   as trauncated, meaning that they cannot be used in the later iterations.  If
#   we used a different truncation policy it might be the case that we could
#   share more work.  For example, if there is a small number of nodes that we
#   could in principle enumerate now (like the dfa_decomp strategy does) then we
#   could get away with that.  It is not possible, in general, to never truncate
#   if we want to terminate.  However, the truncation strategy has a
#   cost-benefit analysis, which I am trying to elucidate a bit.  The knob that
#   controls this is the "truncation policy" and there are smarter things than
#   truncating at N+1 (even for the triplets of doom example).
#   _______________________________________________________________________________
#
# Warning: The BufferedRelevance machine is not finite state!
#
# [2025-12-02 Tue] Possible issue: we could do an unbounded amount of useless
#   work by enumerating states in the buffered relevance machine that are not on
#   track to participate in any relevant target string's precover.
#
# [2025-12-04 Thu] I think we ned to do something similar ot the
#   PeekabooPrecover construction so that we can only explore a finite number of
#   states and, thus, terminate in finite time.  The key to this is going to be
#   to either tweak the state space as in PeekabooPrecover or to use something
#   like we did with `refine`.  The concert about the PeekabooPrecover strategy
#   is that we might add edges to the machine that need to be removed/ignored in
#   the next layer. (The picture that I have in my head is something like change
#   propagation where we have to invalidate some of the work that we done under
#   the assumption that the target string ended here.  Aesthetically and
#   practically, I would rather not require invalidate work.  There is some
#   guiding principle that is lacking here and I don't know how to think about
#   it.)  Maybe the best way to move forward with this is to make a few
#   visualizations of what I expected the layered structure to look like (e.g.,
#   by grafting a set of correct machines for each next symbol together and
#   seeing where things diverge from from that idealization).
#_______________________________________________________________________________
#

class Peekaboo:
    """
    Recursive, batched computation of next-target-symbol optimal DFA-decomposition.
    """
    def __init__(self, fst):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

    def __call__(self, target):
        s = PeekabooState(self.fst, '', parent=None)

        ARCS = [(i,x,j) for j, arcs in s._arcs.items() for x,i in arcs]
        for x in target:
            s >>= x
            ARCS.extend((i,x,j) for j, arcs in s._arcs.items() for x,i in arcs)

        foo = {}
        for y in self.target_alphabet:
            q = FSA(start=set(s.dfa.start()), arcs=ARCS, stop=s.foo[y].quotient)
            r = FSA(start=set(s.dfa.start()), arcs=ARCS, stop=s.foo[y].remainder)
            foo[y] = PrecoverDecomp(q, r)

        return foo

    def graphviz(self, target):
        #
        # TODO: [2025-12-06 Sat] The funny thing about this picture is that the
        # "plates" are technically for the wrong target string.  specifically,
        # they are the precover of the next-target symbol extension of the
        # target target context thus, we have in each plate the *union* of the
        # precovers.
        #
        # TODO: use the integerizer here so that nodes are not improperly
        # equated with thier string representations.
        #
        # TODO: show the active nodes in the graph of the outer most plate
        # (e.g., by coloring them yellow (#f2d66f), as in the precover
        # visualization); inactive node are white.  An additional option would
        # be to color the active vs. inactive edges differently as there is some
        # possibility of misinterpretation.
        #
        # TODO: another useful option would be for each plate to have "output
        # ports" for the nodes that should be expose to the next plate.  (I did
        # something like this in my dissertation code base, which was based on
        # using HTML tables inside node internals.)
        #
        from graphviz import Digraph

        def helper(target, outer):
            print(repr(target))
            with outer.subgraph(name="cluster") as inner:
                inner.attr(label=target,
                           style='rounded, filled', color='black', fillcolor='white')

                if target == '':
                    curr = PeekabooState(self.fst, '', parent=None)

                    for j, arcs in curr._arcs.items():
                        for x,i in arcs:
                            #inner.node(str(i))
                            #inner.node(str(j))
                            inner.edge(str(i), str(j), label=x)

                else:

                    prev = helper(target[:-1], inner)

                    curr = prev >> target[-1]
                    #ARCS.append([(i,x,j) for j, arcs in curr._arcs.items() for x,i in arcs])

                    for j, arcs in curr._arcs.items():
                        for x,i in arcs:
                            #inner.node(str(i))
                            inner.edge(str(i), str(j), label=x)

                for y in curr.foo:
                    tmp = curr.foo[y]
                    for j in tmp.quotient:
                        inner.node(str(j), fillcolor='#90EE90')
                    for j in tmp.remainder:
                        inner.node(str(j), fillcolor='#f26fec')

                return curr

        dot = Digraph(
            graph_attr=dict(rankdir='LR'),
            node_attr=dict(
                fontname='Monospace',
                fontsize='8',
                height='.05',
                width='.05',
                margin="0.055,0.042",
                shape='box',
                style='rounded, filled',
            ),
            edge_attr=dict(
                arrowsize='0.3',
                fontname='Monospace',
                fontsize='8'
            ),
        )

        with dot.subgraph(name='outer') as outer:
            helper(target, outer)

        return dot

    def check(self, target):
        from transduction import display_table
        from IPython.display import HTML

        Have = self(target)

        for y in self.target_alphabet:

            want = Precover(self.fst, target + y)
            have = Have[y]

            q_ok = have.quotient.equal(want.quotient)
            r_ok = have.remainder.equal(want.remainder)

            if q_ok and r_ok:
                print(colors.mark(True), 'sym:', repr(y))
            else:
                print(colors.mark(False), 'sym:', repr(y), 'q:', colors.mark(q_ok), 'r:', colors.mark(r_ok))

                #display_table([
                #    ['quotient', have.quotient.min(), want.quotient.min()],
                #    ['remainder', have.remainder.min(), want.remainder.min()],
                #], headings=['', 'have', 'want'])

                display_table([
                    [HTML('<b>quotient</b>'), have.quotient, want.quotient],
                    [HTML('<b>remainder</b>'), have.remainder, want.remainder],
                ], headings=['', 'have', 'want'])


class PeekabooState:

    def __init__(self, fst, target, parent):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.target = target
        self.parent = parent

        assert parent is None or parent.target == target[:-1]

        def debug(*_): pass
        #debug = print

        dfa = PeekabooPrecover(self.fst, target).det()

        self.dfa = dfa

        if len(target) == 0:
            assert parent is None
            worklist = deque()
            self._arcs = {}
            for state in dfa.start():
                worklist.append(state)
                self._arcs[state] = set()
                debug(colors.orange % 'init:', state)

        else:

            debug(colors.orange % 'target:', repr(self.target))
            debug('GOO:', parent.goo)

            p = parent.goo[target[-1]]
            debug('we need to pick up from the following states:', p)

            # put previous and Q and R states on the worklist
            worklist = deque()
            self._arcs = {}
            for state in p:
                assert not any(truncated for _, ys, truncated in state)
                worklist.append(state)
                self._arcs[state] = set()

        precover = {y: PrecoverDecomp(set(), set()) for y in self.target_alphabet}
        goo = {y: set() for y in self.target_alphabet}

        #def refine(frontier):
        #    "For caching and cycle-detection, we truncate the state representation after N+1 target symbols."
        #    #return frontier
        #    return frozenset((i, ys[:N+1]) for i, ys in frontier)

        def state_relevant_symbols(state):
            return {ys[N] for _, ys, _ in state if len(ys) > N}

        N = len(target)
        while worklist:
            state = worklist.popleft()
            debug()
            debug(colors.cyan % 'work:', state)

            relevant_symbols = state_relevant_symbols(state)
            debug(f'  {relevant_symbols=}')

            # Shortcut: At most one of the `relevant_symbols` can be
            # continuous. If we find one, we can stop expanding.
            continuous = set()
            for y in relevant_symbols:

                # XXX: we may have already constructed this machine
                dfa_truncated = TruncatedDFA(dfa=dfa, fst=self.fst, target=target + y)

                if dfa_truncated.accepts_universal(state, self.source_alphabet):
                    debug('  universal for', repr(y))
                    precover[y].quotient.add(state)
                    continuous.add(y)

                elif dfa_truncated.is_final(state):
                    debug('  accepting for', repr(y))
                    precover[y].remainder.add(state)

                else:
                    debug('  pass on', repr(y))

            assert len(continuous) <= 1
            #debug(f'{continuous=}')
            if continuous:
                continue    # we have found a quotient and can skip

            for x, next_state in dfa.arcs(state):

                if next_state not in self._arcs:
                    assert isinstance(next_state, frozenset), [type(x), x]
                    worklist.append(next_state)
                    self._arcs[next_state] = set()
                    debug('  pushed', next_state)
                else:
                    debug('  pushed-repeat', next_state)

                self._arcs[next_state].add((x, state))

                if not any(truncated for _, ys, truncated in state):
                    for _, ys, truncated in next_state:
                        if truncated:
                            y = ys[-1]
                            debug(colors.light.red % 'goo', state, repr(y), next_state)
                            goo[y].add(state)

        self.foo = precover
        self.goo = goo

        for y in self.foo:
            for state in self.foo[y].quotient | self.foo[y].remainder:
                if not any(truncated for _, ys, truncated in state):
                    debug(colors.light.red % 'goo+', state)
                    goo[y].add(state)

    def __rshift__(self, y):
        assert y in self.target_alphabet, repr(y)
        return PeekabooState(self.fst, self.target + y, parent=self)


# TODO: in order to predict EOS, we need to extract the preimage from Q and R
#
# TODO: should we unify this class with `peekabo.PeekabooPrecover`?
#
#    I think the non-recursive algorithm doesn tneed to worry about the
#    truncation bits, so we probably do not need to unify them.  That said, we
#    might want to have a collection of all the different Precover encodings so
#    that we can test them / compare them independently of the algorithms that
#    use them.
#
class PeekabooPrecover(Lazy):

    def __init__(self, f, target):
        self.f = f
        self.target = target
        self.N = len(target)

    def arcs(self, state):
        (i, ys, truncated) = state   # TODO: use truncated bit to speed some of this up
        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:      # target and ys are not prefixes of one another.
            return
        if m >= self.N:                    # i.e, target <= ys
            for x, y, j in self.f.arcs(i):

                if y == EPSILON or truncated:
                    yield (x, (j, ys, truncated))

                else:
                    assert not truncated
                    was = (ys + y)

                    # XXX: truncation policy
                    now = was[:self.N+1]
                    #now = was[:self.N+2]

                    # mark truncated nodes because they need to be removed in the next iteration
                    yield (x, (j, now, truncated or (was != now)))

        else:                              # i.e, ys < target)
            assert not truncated
            for x, y, j in self.f.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys, truncated))
                elif y == self.target[n]:
                    yield (x, (j, self.target[:n+1], truncated))

    def start(self):
        for i in self.f.I:
            yield (i, self.target[:0], False)

    def is_final(self, state):
        (i, ys, _) = state
        return self.f.is_final(i) and ys.startswith(self.target) and len(ys) > self.N


class TruncatedDFA(Lazy):
    """NOTE: This class augments a determinized `PeekabooPrecover` semi-automaton by
    adding an appropriate `is_final` method so that it is a valid finite-state
    automaton that encodes `Precover(fst, target)`.


    # In other words, for all `target` strings:
    # dfa_truncated = TruncatedDFA(dfa=dfa, fst=self.fst, target=target)
    # assert dfa_filtered.materialize().equal(Precover(self.fst, target).dfa)


    """

    def __init__(self, *, dfa, fst, target):
        self.dfa = dfa
        self.fst = fst
        self.target = target

    def start(self):
        return self.dfa.start()

    # TODO: I think this is optional, but possible less efficient
    def refine(self, frontier):
        # clip the target side side to `y` in order to mimick the states of the
        # composition machine that we used in the new lazy, nonrecursive
        # algorithm.
        N = len(self.target)
        return frozenset(
            (i, ys[:N], truncated) for i, ys, truncated in frontier     # TODO: can we use the truncated bit here?
            if ys[:min(N, len(ys))] == self.target[:min(N, len(ys))]
        )
#        return frontier

    def arcs(self, state):
        for x, next_state in self.dfa.arcs(state):
            yield x, self.refine(next_state)

    def is_final(self, state):
        return any(ys.startswith(self.target) and self.fst.is_final(i) for (i, ys, _) in state)
