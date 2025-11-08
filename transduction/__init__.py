from transduction.util import *
from transduction.base import AbstractAlgorithm, PrecoverDecomp
from transduction.lazy_recursive import BuggyLazyRecursive, LazyRecursive
from transduction.lazy_nonrecursive import LazyNonrecursive, LazyPrecoverNFA
from transduction.eager_nonrecursive import EagerNonrecursive
from transduction import examples

from arsenal import colors

from transduction.fst import FST, EPSILON
from transduction.fsa import FSA
from transduction.eager_nonrecursive import build_precover_dfa


def check_properties(fst, target, Q, R, throw=False):
    source_alphabet = fst.A - {EPSILON}
    target_alphabet = fst.B - {EPSILON}
    P = build_precover_dfa(fst, target, target_alphabet)

    U = FSA.universal(source_alphabet)

    ok = True

    z = P.equal(FSA.from_strings(Q) * U + FSA.from_strings(R))
    ok &= z
    print('check properties:')
    print('├─ valid:', colors.mark(z), 'equal to precover')
    assert z

    if Q: print('├─ quotient:')
    for xs in Q:
        z = FSA.from_string(xs) * U <= P
        print('├─', colors.mark(z), repr(xs), 'is a valid cylinder')
        ok &= z

        # minimality -- if there is a strict prefix that is a cylinder of the precover, then `xs` is not minimal.
        # in terms of the precover automaton, `xs` leads to a universal state, but we want to make
        # sure that there wasn't an earlier time that it arrived there.
        for t in range(len(xs)):
            xss = xs[:t]
            if xss == xs: continue  # skip equality case
            if FSA.from_string(xss) * U <= P:
                print('├─', colors.mark(False), repr(xs), 'is not minimal because its strict prefix', repr(xss), 'is a cylinder of the precover')
                ok &= False

    if len(R): print('├─ remainder:')
    for xs in R:
        z = (xs in P)
        zz = not (FSA.from_string(xs) * U <= P)
        print('├─', colors.mark(z & zz), repr(xs), 'in precover and not a cylinder')
        ok &= z
        ok &= zz
    print('└─ overall:', colors.mark(ok))
    assert not throw or ok
