from transduction.base import PrecoverDecomp
from transduction.eager_nonrecursive import Precover
from transduction.lazy import Lazy
from transduction.fsa import FSA, frozenset
from transduction.fst import FST, EPSILON
from transduction.eager_nonrecursive import EagerNonrecursive
from transduction.lazy_recursive import LazyRecursive
#from transduction.lazy_nonrecursive import LazyNonrecursive

from arsenal import colors
from collections import deque


class Peekaboo:
    """
    Recursive, batched computation of next-target-symbol optimal DFA-decomposition.
    """
    def __init__(self, fst, max_steps=float('inf')):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.max_steps = max_steps

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
        from graphviz import Digraph

        # TODO: [2025-12-06 Sat] The funny thing about this picture is that the
        # "plates" are technically for the wrong target string.  specifically,
        # they are the precover of the next-target symbol extension of the
        # target target context thus, we have in each plate the *union* of the
        # precovers

        # TODO: use the integerizer here so that nodes are not improperly
        # equated with thier string representations.
        from arsenal import Integerizer
        m = Integerizer()

        def helper(target, outer):
            print(repr(target))
            with outer.subgraph(name="cluster") as inner:
                inner.attr(label=target,
                           style='rounded, filled', color='black', fillcolor='#e0ffe0')

                if target == '':
                    curr = PeekabooState(self.fst, '', parent=None, max_steps=self.max_steps)

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
                        inner.node(str(j), fillcolor='red')
                    for j in tmp.remainder:
                        inner.node(str(j), fillcolor='magenta')

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
        from transduction import Precover, display_table
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

class PeekabooState:

    def __init__(self, fst, target, parent, max_steps=float('inf')):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.target = target
        self.parent = parent

        if self.parent is not None:
            self.max_steps = min(self.parent.max_steps, max_steps)
        else:
            self.max_steps = max_steps

        assert parent is None or parent.target == target[:-1]

        def debug(*args): pass
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

        verbosity = 0
        N = len(target)
        t = 0
        visited = set()
        while worklist:
            state = worklist.popleft()
            t += 1
            debug()
            debug(colors.cyan % 'work:', state)

            if t >= self.max_steps:
                print(colors.light.red % 'TOOK TOO LONG')
                break

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

                # [2025-12-02 Tue] unfortunately, these graphs aren't always
                # cleanly layered; we can have arcs that go backward (e.g., when
                # Q and R are cyclical) [TODO: add examples] We can also have
                # empty layers (these are layers where the nodes are in previous
                # layers - just imagine a case where Q(abc) = Q(ab)).

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
class PeekabooPrecover(Lazy):

    def __init__(self, f, target):
        self.f = f
        self.target = target
        self.N = len(target)

    def arcs(self, state):
        (i, ys, truncated) = state   # TODO: use truncated bit to speed some of this up
        if ys.startswith(self.target):
            for x, y, j in self.f.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys, truncated))
                else:
                    was = (ys + y)
                    now = was[:self.N+1]
                    if was == now:
                        yield (x, (j, was, False))
                    else:
                        # mark truncated nodes because they need to be removed in the next iteration
                        yield (x, (j, now, True))
        else:
            n = len(ys)
            assert not truncated
            for x, y, j in self.f.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys, False))
                elif y == self.target[n]:
                    yield (x, (j, ys + y, False))

    def start(self):
        for i in self.f.I:
            yield (i, '', False)

    def is_final(self, state):
        (i, ys) = state
        return self.f.is_final(i) and ys.startswith(self.target)



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

    def refine(self, frontier):
        # clip the target side side to `y` in order to mimick the states of the
        # composition machine that we used in the new lazy, nonrecursive
        # algorithm.
        N = len(self.target)
        return frozenset(
            (i, ys[:N], truncated) for i, ys, truncated in frontier     # TODO: can we use the truncated bit here?
            if ys[:min(N, len(ys))] == self.target[:min(N, len(ys))]
        )

    def arcs(self, state):
        for x, next_state in self.dfa.arcs(state):
            yield x, self.refine(next_state)

    def is_final(self, state):
        return any(ys.startswith(self.target) and self.fst.is_final(i) for (i, ys, _) in state)


#_______________________________________________________________________________
# TESTING CODE BELOW

from transduction import examples


class recursive_testing:
    """
    Utility function for testing the `Peekaboo` method against a slower method.
    """
    def __init__(self, fst, target, depth, verbosity=0):
        self.fst = fst
        self.target_alphabet = self.fst.B - {EPSILON}
        self.depth = depth
        self.peekaboo = Peekaboo(fst)
        self.reference = lambda target: Precover(fst, target)
#        self.reference = LazyNonrecursive(fst)
#        self.reference = EagerNonrecursive(fst)
        self.verbosity = verbosity
        self.run(target, depth)

    def run(self, target, depth):
        if depth == 0: return
        want = {y: self.reference(target + y) for y in self.target_alphabet}
        have = self.peekaboo(target)
        assert_equal_decomp_map(have, want)
        for y in want:
            if self.verbosity > 0: print('>', repr(target + y))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                self.run(target + y, depth - 1)


def assert_equal_decomp_map(have, want):
    for y in have | want:
        assert have[y].quotient.equal(want[y].quotient)
        assert have[y].remainder.equal(want[y].remainder)


def test_abc():
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])
    recursive_testing(fst, '', depth=4)


def test_samuel():
    fst = examples.samuel_example()
    recursive_testing(fst, '', depth=5)


def test_small():
    fst = examples.small()
    recursive_testing(fst, '', depth=5)


def test_sdd1():
    fst = examples.sdd1_fst()
    recursive_testing(fst, '', depth=5)


def test_duplicate():
    fst = examples.duplicate(set('12345'))
    recursive_testing(fst, '', depth=5)


def test_number_comma_separator():
    #import string
    #fst = examples.number_comma_separator(set(string.printable) - set('\t\n\r\x0b\x0c'))
    fst = examples.number_comma_separator({'a', ',', ' ', '0'}, Digit={'0'})
    recursive_testing(fst, '', depth=4, verbosity=1)
    recursive_testing(fst, '0,| 0,', depth=1, verbosity=1)
    recursive_testing(fst, '0,| 0,|', depth=1, verbosity=1)


def test_newspeak2():
    fst = examples.newspeak2()
    p = Peekaboo(fst, max_steps=500)
    recursive_testing(fst, '', depth=1)
    recursive_testing(fst, 'ba', depth=1)
    recursive_testing(fst, 'bad', depth=1)


def test_lookahead():
    fst = examples.lookahead()
    recursive_testing(fst, '', depth=6, verbosity=1)


def test_weird_copy():
    fst = examples.weird_copy()
    recursive_testing(fst, '', depth=5, verbosity=0)


def test_triplets_of_doom():
    fst = examples.triplets_of_doom()
    recursive_testing(fst, '', depth=13, verbosity=0)


def test_infinite_quotient():
    fst = examples.infinite_quotient()
    recursive_testing(fst, '', depth=5, verbosity=1)


def test_parity():
    fst = examples.parity({'a', 'b'})
    recursive_testing(fst, '', depth=5, verbosity=1)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
