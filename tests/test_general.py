import pytest
from transduction import examples, FSA, EPSILON, Precover
from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp
from transduction.dfa_decomp_recursive import RecursiveDFADecomp
from transduction.token_decompose import TokenDecompose
from transduction.fst import check_all_input_universal
from transduction import peekaboo_nonrecursive
from transduction import peekaboo_recursive

try:
    from transduction.rust_bridge import RustDecomp
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


class run_recursive_dfa_decomp:
    """
    Utility function for testing a precover decomposition method against a reference
    implementation method.
    """
    def __init__(self, fst, target, depth, verbosity=0):
        self.fst = fst
        self.target_alphabet = self.fst.B - {EPSILON}
        self.depth = depth
        self.reference = lambda target: Precover(fst, target)
        self.verbosity = verbosity
        self.run(target, depth, RecursiveDFADecomp(fst, target))

    def run(self, target, depth, state):
        if depth == 0: return
        want = {y: self.reference(target + y) for y in self.target_alphabet}
        have = {y: (state >> y) for y in self.target_alphabet}
        assert_equal_decomp_map(have, want)
        for y in want:
            if self.verbosity > 0: print('>', repr(target + y))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                self.run(target + y, depth - 1, have[y])


class run_peekaboo_recursive:
    """
    Utility function for testing the `Peekaboo` method against a slower method.
    """
    def __init__(self, fst, target, depth, verbosity=0):
        self.fst = fst
        self.target_alphabet = self.fst.B - {EPSILON}
        self.depth = depth
        self.peekaboo = peekaboo_recursive.Peekaboo(fst)
        self.reference = lambda target: Precover(fst, target)
        self.verbosity = verbosity
        self.run(target, depth)

    def run(self, target, depth):
        if depth == 0: return

        # Check that the peekaboo machine matches the reference implementation
        have = peekaboo_recursive.PeekabooPrecover(self.fst, target).materialize()
        want = (self.fst @ (FSA.from_string(target) * FSA.from_strings(self.target_alphabet).p())).project(0)
        assert have.equal(want)

        # Check that the decomposition matches the reference implementation
        want = {y: self.reference(target + y) for y in self.target_alphabet}
        have = self.peekaboo(target)
        assert_equal_decomp_map(have, want)

        # recurse
        for y in want:
            if self.verbosity > 0: print('>', repr(target + y))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                self.run(target + y, depth - 1)


class run_peekaboo_nonrecursive:
    """
    Utility function for testing the `Peekaboo` method against a slower method.
    """
    def __init__(self, fst, target, depth, verbosity=0):
        self.fst = fst
        self.target_alphabet = self.fst.B - {EPSILON}
        self.depth = depth
        self.peekaboo = peekaboo_nonrecursive.Peekaboo(fst)
        self.reference = lambda target: Precover(fst, target)
        self.verbosity = verbosity
        self.run(target, depth)

    def run(self, target, depth):
        if depth == 0: return

        # Check that the peekaboo machine matches the reference implementation
        have = peekaboo_nonrecursive.PeekabooPrecover(self.fst, target).materialize()
        want = (self.fst @ (FSA.from_string(target) * FSA.from_strings(self.target_alphabet).p())).project(0)
        assert have.equal(want)

        # Check that the decomposition matches the reference implementation
        want = {y: self.reference(target + y) for y in self.target_alphabet}
        have = self.peekaboo(target)
        assert_equal_decomp_map(have, want)

        # Recurse
        for y in want:
            if self.verbosity > 0: print('>', repr(target + y))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                self.run(target + y, depth - 1)


class run_nonrecursive_dfa_decomp:
    """
    Utility function for testing the `Peekaboo` method against a slower method.
    """
    def __init__(self, fst, target, depth, verbosity=0):
        self.fst = fst
        self.target_alphabet = self.fst.B - {EPSILON}
        self.depth = depth
        self.peekaboo = lambda target: NonrecursiveDFADecomp(fst, target)
        self.reference = lambda target: Precover(fst, target)
        self.verbosity = verbosity
        self.run(target, depth)

    def run(self, target, depth):
        if depth == 0: return

        # Check that the decomposition matches the reference implementation
        want = {y: self.reference(target + y) for y in self.target_alphabet}
        have = {y: self.peekaboo(target + y) for y in self.target_alphabet}
        assert_equal_decomp_map(have, want)

        # Recurse
        for y in want:
            if self.verbosity > 0: print('>', repr(target + y))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                self.run(target + y, depth - 1)


class run_token_decompose:
    """
    Utility function for testing the token-level decomposition against the
    reference implementation. Falls back to NonrecursiveDFADecomp when
    all_input_universal is False.
    """
    def __init__(self, fst, target, depth, verbosity=0):
        self.fst = fst
        self.target_alphabet = self.fst.B - {EPSILON}
        self.depth = depth
        self.all_univ = check_all_input_universal(fst)
        self.decompose = (
            (lambda target: TokenDecompose(fst, target))
            if self.all_univ
            else (lambda target: NonrecursiveDFADecomp(fst, target))
        )
        self.reference = lambda target: Precover(fst, target)
        self.verbosity = verbosity
        self.run(target, depth)

    def run(self, target, depth):
        if depth == 0: return

        # Check that the decomposition matches the reference implementation
        want = {y: self.reference(target + y) for y in self.target_alphabet}
        have = {y: self.decompose(target + y) for y in self.target_alphabet}
        assert_equal_decomp_map(have, want)

        # Recurse
        for y in want:
            if self.verbosity > 0: print('>', repr(target + y))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                self.run(target + y, depth - 1)


class run_rust_decomp:
    """
    Utility function for testing the Rust-accelerated decomposition against the
    reference implementation.
    """
    def __init__(self, fst, target, depth, verbosity=0):
        self.fst = fst
        self.target_alphabet = self.fst.B - {EPSILON}
        self.depth = depth
        self.rust_decomp = lambda target: RustDecomp(fst, target)
        self.reference = lambda target: Precover(fst, target)
        self.verbosity = verbosity
        self.run(target, depth)

    def run(self, target, depth):
        if depth == 0: return

        # Check that the decomposition matches the reference implementation
        want = {y: self.reference(target + y) for y in self.target_alphabet}
        have = {y: self.rust_decomp(target + y) for y in self.target_alphabet}
        assert_equal_decomp_map(have, want)

        # Recurse
        for y in want:
            if self.verbosity > 0: print('>', repr(target + y))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                self.run(target + y, depth - 1)


IMPLEMENTATIONS = [
    pytest.param(run_recursive_dfa_decomp, id="recursive_dfa_decomp"),
    pytest.param(run_nonrecursive_dfa_decomp, id="nonrecursive_dfa_decomp"),
    pytest.param(run_peekaboo_recursive, id="peekaboo_recursive"),
    pytest.param(run_peekaboo_nonrecursive, id="peekaboo_nonrecursive"),
    pytest.param(run_token_decompose, id="token_decompose"),
]

if HAS_RUST:
    IMPLEMENTATIONS.append(
        pytest.param(run_rust_decomp, id="rust_decomp"),
    )


def assert_equal_decomp_map(have, want):
    for y in have | want:
        assert have[y].quotient.equal(want[y].quotient)
        assert have[y].remainder.equal(want[y].remainder)


@pytest.fixture(params=IMPLEMENTATIONS)
def run(request):
    return request.param


def test_abc(run):
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])
    run(fst, '', depth=4)


def test_delete_b(run):
    fst = examples.delete_b()
    run(fst, '', depth=10)


def test_samuel(run):
    fst = examples.samuel_example()
    run(fst, '', depth=5)


def test_small(run):
    fst = examples.small()
    run(fst, '', depth=5)


def test_sdd1(run):
    fst = examples.sdd1_fst()
    run(fst, '', depth=5)


def test_duplicate(run):
    fst = examples.duplicate(set('12345'))
    run(fst, '', depth=5)


def test_number_comma_separator(run):
    #import string
    #fst = examples.number_comma_separator(set(string.printable) - set('\t\n\r\x0b\x0c'))
    fst = examples.number_comma_separator({'a', ',', ' ', '0'}, Digit={'0'})
    run(fst, '', depth=4, verbosity=0)
    run(fst, '0,| 0,', depth=1, verbosity=0)
    run(fst, '0,| 0,|', depth=1, verbosity=0)


def test_newspeak2(run):
    fst = examples.newspeak2()
    run(fst, '', depth=1)
    run(fst, 'ba', depth=1)
    run(fst, 'bad', depth=1)


def test_lookahead(run):
    fst = examples.lookahead()
    run(fst, '', depth=6, verbosity=0)


def test_weird_copy(run):
    fst = examples.weird_copy()
    run(fst, '', depth=5, verbosity=0)


def test_triplets_of_doom(run):
    if run is run_recursive_dfa_decomp:
        pytest.xfail("recursive_dfa_decomp does not terminate on this input")
    from arsenal import timelimit
    fst = examples.triplets_of_doom()
    with timelimit(5):
        run(fst, '', depth=13, verbosity=0)


def test_infinite_quotient(run):
    fst = examples.infinite_quotient()
    run(fst, '', depth=5, verbosity=0)


def test_parity(run):
    fst = examples.parity({'a', 'b'})
    run(fst, '', depth=5, verbosity=0)


if __name__ == '__main__':
    from arsenal import testing_framework

    algs = [
        run_recursive_dfa_decomp,
        run_nonrecursive_dfa_decomp,
        run_peekaboo_recursive,
        run_peekaboo_nonrecursive,
        run_token_decompose,
    ]
    if HAS_RUST:
        algs.append(run_rust_decomp)

    options = {}
    env = dict(globals())
    for f in env:
        if f.startswith('test_'):
            for alg in algs:
                options[f'{f}[{alg.__name__}]'] = lambda f=f, alg=alg: env[f](alg)

    testing_framework(options)
