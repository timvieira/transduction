from transduction.base import AbstractAlgorithm, EPSILON, PrecoverDecomp
from transduction.lazy import Lazy
from transduction.fst import FST, EPSILON
from transduction.eager_nonrecursive import EagerNonrecursive
from transduction.lazy_recursive import LazyRecursive
from transduction.lazy_nonrecursive import LazyNonrecursive

from arsenal import colors
from collections import deque, defaultdict


class Peekaboo(AbstractAlgorithm):

    def __init__(self, fst, empty_source = '', extend = lambda x,y: x + y, max_steps=float('inf')):
        self.fst = fst
        self.empty_source = empty_source
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.extend = extend
        self.max_steps = max_steps
        # the variables below need to be used very carefully
        self.state = None
        self.dfa = None
        self.nfa = None

    def __call__(self, target):
        precover = {y: PrecoverDecomp(set(), set()) for y in self.target_alphabet}
        worklist = deque()
        for xs in self.initialize(target):
            worklist.append(xs)
        t = 0
        N = len(target)
        while worklist:
            xs = worklist.popleft()
            t += 1
            if t > self.max_steps:
                print(colors.light.red % 'stopped early')
                break

            # XXX: I think that both the continuity test and discontinuity test
            # here are hacky.  We need to test that the relevant target-side
            # strings that we encounter once they look "ripe" are ready for
            # their tests.  Much like the recursive algorithm, it is best to
            # view the peekaboo machine as a semi-automatan.  The universality
            # test and---more fundamentally---the acceptance tests run can't be
            # accurately translated
            #

            # get Y, the relvant symbols
            relevant_symbols = set()
            for _, ys in self.state[xs]:
                if len(ys) > N:
                    relevant_symbols.add(ys[N])

            if not relevant_symbols: # Early exit
                for next_xs in self.candidates(xs, target):
                    worklist.append(next_xs)
                continue

            found_continuous = False
            for y in relevant_symbols: # 
                if self.continuity(xs, y):
                    precover[y].quotient.add(xs)
                    found_continuous = True
                    break  # At most one can be continuous
            if found_continuous:
                continue # we have found a quotient and can skip      

            for y in relevant_symbols:
                if self.discontinuity(xs, y):
                    precover[y].remainder.add(xs)            
            for next_xs in self.candidates(xs, target):
                worklist.append(next_xs)
        return precover

    def initialize(self, target):
        self.state = {}
        self.nfa = PeekabooPrecover(self.fst, target)
        self.dfa = self.nfa.det()#.trim()   # XXX: if we don't trim here, there will be a dead state that causes us to run forever!
        for state in self.dfa.start():
            self.state[self.empty_source] = state
            yield self.empty_source

    def candidates(self, xs, target):
        state = self.state[xs]
        for source_symbol, next_state in self.dfa.arcs(state):
            next_xs = self.extend(xs, source_symbol)
            self.state[next_xs] = next_state
            yield next_xs

    # def discontinuity(self, xs, target):
    #     return self.dfa.is_final(self.state[xs])

    # def continuity(self, xs, target):
    #     return self.dfa.accepts_universal(self.state[xs], self.source_alphabet)

    def discontinuity(self, xs, target_symbol):
        # Filtered finality
        dfa_filtered = FilteredDFA(self.dfa, self.nfa, target_symbol, len(self.nfa.target))
        return dfa_filtered.is_final(self.state[xs])

    def continuity(self, xs, target_symbol):
        # Filtered finality
        dfa_filtered = FilteredDFA(self.dfa, self.nfa, target_symbol, len(self.nfa.target))
        return dfa_filtered.accepts_universal(self.state[xs], self.source_alphabet)


class FilteredDFA(Lazy):
    """filter a DFA's is_final to only accept states that emit specific target symbol."""
    
    def __init__(self, dfa, nfa, target_symbol, target_length):
        self.dfa = dfa
        self.nfa = nfa
        self.target_symbol = target_symbol
        self.N = target_length
    
    def start(self):
        return self.dfa.start()
    
    def arcs(self, state):
        return self.dfa.arcs(state)
    
    def is_final(self, state):
        # filter NFA states with the target symbol
        for nfa_state in state:
            _, ys = nfa_state
            if len(ys) == self.N + 1 and ys[self.N] == self.target_symbol:
                if self.nfa.is_final(nfa_state):
                    return True
        return False


class PeekabooPrecover(Lazy):

    def __init__(self, f, target):
        self.f = f
        self.target = target

    def arcs(self, state):
        (i, ys) = state
        n = len(ys)
        N = len(self.target)
        if ys == self.target and n == N:
            for x,y,j in self.f.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                else:
                    yield (x, (j, ys + y))

        elif ys.startswith(self.target) and n == N + 1:
            for a,b,j in self.f.arcs(i):
                yield (a, (j, ys))

        elif self.target.startswith(ys) and n < N:
            for x,y,j in self.f.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                elif y == self.target[len(ys)]:
                    yield (x, (j, ys + y))

    def start(self):
        for i in self.f.I:
            yield (i, self.target[:0])

    def is_final(self, state):
        (i, ys) = state
        return (i in self.f.F) and ys.startswith(self.target) and len(ys) == len(self.target) + 1



#_______________________________________________________________________________
# TESTING CODE BELOW

from transduction import examples


class recursive_testing:
    """
    Utility function for testing the `Peekaboo` method against a slower method.
    """
    def __init__(self, fst, target, depth):
        self.fst = fst
        self.depth = depth
        self.peekaboo = Peekaboo(fst)
        self.reference = LazyRecursive(fst)
#        self.reference = LazyNonrecursive(fst)
#        self.reference = EagerNonrecursive(fst)
        self.run(target, depth)

    def run(self, target, depth):
        if depth == 0: return
        want = {y: self.reference(target + y) for y in self.reference.target_alphabet}
        have = self.peekaboo(target)
        assert have == want, f"""\ntarget = {target!r}\nhave = {have}\nwant = {want}\n"""
        for y in want:
            if want[y].quotient or want[y].remainder:   # nonempty
                self.run(target + y, depth - 1)


def test_abc():
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])

    p = Peekaboo(fst)
    target = ''
    have = p(target)
    tmp = EagerNonrecursive(fst)
    want = {y: tmp(target + y) for y in tmp.target_alphabet}
    assert have == want

    target = 'abc'
    have = p(target)
    tmp = EagerNonrecursive(fst)
    want = {y: tmp(target + y) for y in tmp.target_alphabet}
    assert have == want

    recursive_testing(fst, '', depth=5)


def test_samuel():
    fst = examples.samuel_example()

    p = Peekaboo(fst)
    target = 'y'
    have = p(target)
    tmp = EagerNonrecursive(fst)
    want = {y: tmp(target + y) for y in tmp.target_alphabet}

    print(have)
    print(want)

    assert have == want

    recursive_testing(fst, '', depth=5)


def test_small():

    fst = FST()
    fst.add_I(0)
    fst.add_F(0)

    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'x', 2)

    fst.add_arc(2, 'a', 'a', 3)
    fst.add_arc(2, 'b', 'b', 3)

    fst.add_arc(3, 'a', 'a', 3)
    fst.add_arc(3, 'b', 'b', 3)

    fst.add_F(1)
    fst.add_F(3)

    recursive_testing(fst, '', depth=5)


def test_sdd1():
    fst = examples.sdd1_fst()
    recursive_testing(fst, '', depth=5)


def test_duplicate():
    fst = examples.duplicate(set('12345'))
    recursive_testing(fst, '', depth=5)


def test_number_comma_separator():
    import string
    #fst = examples.number_comma_separator(set(string.printable) - set('\t\n\r\x0b\x0c'))
    fst = examples.number_comma_separator({'a','b',',',' ','0','1'}, Digit={'0', '1'})

    recursive_testing(fst, '', depth=5)


def test_newspeak2():
    from transduction import Precover
    m = examples.newspeak2()
    p = Peekaboo(m, max_steps=500)
    target = ''
    have = p(target)
    tmp = EagerNonrecursive(m)
    want = {y: tmp(target + y) for y in tmp.target_alphabet}

    #print('have=', have)
    #print('want=', want)

    for y in have | want:
        if have.get(y) == want.get(y):
            print(colors.mark(True), repr(y))
        else:
            print(colors.mark(False), repr(y))
            print('  have=', have.get(y))
            print('  want=', want.get(y))
            #Precover(m, target + y).check_decomposition(*want[y], throw=True)
            Precover(m, target + y).check_decomposition(*have[y], throw=False)
    assert have == want


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
