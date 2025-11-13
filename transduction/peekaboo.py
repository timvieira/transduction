from transduction.base import AbstractAlgorithm, EPSILON, PrecoverDecomp
from transduction.fst import FST, EPSILON
from transduction.eager_nonrecursive import EagerNonrecursive
from transduction.lazy_recursive import LazyRecursive

from arsenal import colors


def get_string_from_state(state):
    # the epsilon filtering trick does some annoying stuff with how states are
    # named here we do a little guess work.
    if isinstance(state, tuple):
        _, ys = state
    else:
        ys = state
    return ys


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
        worklist = []
        for xs in self.initialize(target):
            worklist.append(xs)
        t = 0
        N = len(target)
        while worklist:
            xs = worklist.pop()
            t += 1
            if t > self.max_steps:
                print(colors.light.red % 'stopped early')
                break
            if self.continuity(xs, target):
                # samuel pointed out that at most one of the target-side
                # extensions can have `xs` in its quotient.  Given that the
                # continuity test was true, we just need to figure out which
                # target-side extension is responsible for it!
                count = set()
                for _, ys in self.state[xs]:
                    if len(ys) > N:
                        precover[ys[N]].quotient.add(xs)
                        count.add(ys[N])
                    #else:
                    #    print(colors.light.red % 'incomplete state:', repr(ys), 'in', self.state[xs])
                assert len(count) == 1, f"""
{count = }
{xs = }
{target = }
state = {self.state[xs]}
"""
                continue
            if self.discontinuity(xs, target):
                for _, ys in self.state[xs]:
                    if len(ys) > N:
                        precover[ys[N]].remainder.add(xs)
                    else:
                        print(colors.light.red % 'incomplete state:', repr(ys), 'in', self.state[xs])
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

    def discontinuity(self, xs, target):
        return self.dfa.is_final(self.state[xs])

    def continuity(self, xs, target):
        return self.dfa.accepts_universal(self.state[xs], self.source_alphabet)


#def PeekabooPrecover(fst, target):
#    "FSA representing the complete precover."
#    target_alphabet = fst.B - {EPSILON}
#    # this is a copy machine for target \targetAlphabet^+
#    m = FST()
#    m.add_I(target[:0])
#    N = len(target)
#    for i in range(N):
#        m.add_arc(target[:i], target[i], target[i], target[:i+1])
#    for y in target_alphabet:
#        m.add_arc(target, y, y, target + y)
#        # XXX: we have separate final states for each target extension
#        m.add_F(target + y)
#        for yy in target_alphabet:
#            m.add_arc(target + y, yy, yy, target + y)
#    have = LazyPeekabooPrecover(fst, target).materialize()
#    want = (fst @ m).project(0)   # this version does not bypass the epsilon filter
#    assert have.equal(want)
#    return want


from transduction.lazy import Lazy
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


class recursive_testing:
    """
    Utility function for testing the `Peekaboo` method against a slower method.
    """
    def __init__(self, fst, target, depth):
        self.fst = fst
        self.depth = depth
        self.peekaboo = Peekaboo(fst)
        self.reference = LazyRecursive(fst)
        self.run(target, depth)

    def run(self, target, depth):
        if depth == 0: return
        want = {y: self.reference(target + y) for y in self.reference.target_alphabet}
        have = self.peekaboo(target)
        assert have == want, f"""\ntarget = {target!r}\nhave = {have}\nwant = {want}\n"""
        for y in want:
            if want[y].quotient or want[y].remainder:   # nonempty
                self.run(target + y, depth - 1)


from transduction import examples

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


# TODO: needs support for `bytes`
#def test_newspeak():
#    fst = examples.newspeak()
#    recursive_testing(fst, b'', depth=3)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
