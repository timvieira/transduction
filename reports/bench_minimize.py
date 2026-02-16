"""Benchmark: min_fast vs old min_faster vs new min_faster.

Compares three Hopcroft variants:
  min_fast        — no find index, scans all blocks each iteration
  min_faster_old  — find index, but O(|Y|) superset check (X >= Y)
  min_faster_new  — find index + O(1) block_members grouping
"""
import time, random
from collections import defaultdict
from transduction.fsa import FSA
from transduction.util import set_memory_limit

set_memory_limit(4)


def min_faster_old(self):
    """Old min_faster: find index + X >= Y superset check."""
    self = self.det().renumber()
    inv = defaultdict(set)
    for i, a, j in self.arcs():
        inv[j, a].add(i)
    final = self.stop
    nonfinal = self.states - final
    P = [final, nonfinal]
    W = [final, nonfinal]
    find = {i: block for block, elements in enumerate(P) for i in elements}
    while W:
        A = W.pop()
        for a in self.syms:
            X = {i for j in A for i in inv[j, a]}
            blocks = {find[i] for i in X}
            for block in blocks:
                Y = P[block]
                if X >= Y:
                    continue
                YX = Y & X
                Y_X = Y - X
                P[block] = YX
                new_block = len(P)
                for i in Y_X:
                    find[i] = new_block
                P.append(Y_X)
                W.append(YX if len(YX) < len(Y_X) else Y_X)
    return self.rename(lambda i: find[i]).trim()


def bench(label, fsa, n_trials=5, skip_min_fast=False):
    """Benchmark minimization methods on an FSA (already a DFA)."""
    dfa = fsa.det().renumber()
    n_arcs = sum(1 for _ in dfa.arcs())
    print(f"\n{label}: {len(dfa.states)} DFA states, "
          f"{n_arcs} arcs, {len(dfa.syms)} symbols")

    methods = []
    if not skip_min_fast:
        methods.append(("min_fast", lambda: dfa.min_fast()))
    methods.append(("min_faster_old", lambda: min_faster_old(dfa)))
    methods.append(("min_faster_new", lambda: dfa.min_faster()))

    results = {}
    for name, fn in methods:
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            result = fn()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        best = min(times)
        results[name] = (best, result)
        print(f"  {name:20s}: {best*1000:8.2f} ms  "
              f"({len(result.states)} min states)")

    sizes = {name: len(r.states) for name, (_, r) in results.items()}
    assert len(set(sizes.values())) == 1, f"Size mismatch: {sizes}"


def make_dfa_deep_refinement(n, k=4, n_final_classes=3):
    """DFA with deep refinement: states that look similar but differ at depth.

    Creates a DFA where equivalence classes require multiple rounds of
    refinement to distinguish.  This stresses the per-block superset check.
    """
    random.seed(42)
    fsa = FSA()
    fsa.start = {0}
    # Assign states to final classes based on state mod n_final_classes
    for i in range(n):
        if i % n_final_classes == 0:
            fsa.stop.add(i)
    # Build DFA transitions: mostly uniform structure but with subtle differences
    for i in range(n):
        for a in range(k):
            # Most states go to (i + a + 1) % n, creating lots of equivalent-looking blocks
            j = (i * (a + 2) + a + 1) % n
            fsa.add_arc(i, a, j)
    return fsa


def make_dfa_wide_alphabet(n, k):
    """DFA with wide alphabet — many symbols but few states.

    Each refinement step iterates over k symbols, with most blocks
    satisfying X >= Y (the no-split fast path).
    """
    random.seed(42)
    fsa = FSA()
    fsa.start = {0}
    fsa.stop = {n - 1}
    for i in range(n):
        for a in range(k):
            j = random.randint(0, n - 1)
            fsa.add_arc(i, a, j)
    return fsa


def make_powerset_nfa(n, k=3, density=2):
    """NFA that blows up under determinization, producing a large DFA to minimize.

    Each state has `density` successors per symbol, creating powerset explosion.
    """
    random.seed(42)
    fsa = FSA()
    fsa.start = {0}
    fsa.stop = {n - 1}
    for i in range(n):
        for a in range(k):
            for _ in range(density):
                j = random.randint(0, n - 1)
                fsa.add_arc(i, a, j)
    return fsa


if __name__ == "__main__":
    print("=" * 65)
    print("Deep-refinement DFAs (many rounds of splitting)")
    print("=" * 65)
    for n in [200, 500, 1000, 2000]:
        fsa = make_dfa_deep_refinement(n, k=4)
        bench(f"deep(n={n}, k=4)", fsa, skip_min_fast=(n > 1000))

    print("\n" + "=" * 65)
    print("Wide-alphabet DFAs (many symbols, fast-path superset checks)")
    print("=" * 65)
    for n, k in [(100, 26), (100, 52), (200, 26), (500, 10)]:
        fsa = make_dfa_wide_alphabet(n, k)
        bench(f"wide(n={n}, k={k})", fsa)

    print("\n" + "=" * 65)
    print("Powerset NFA → DFA (realistic blowup)")
    print("=" * 65)
    for n in [8, 10, 12]:
        fsa = make_powerset_nfa(n, k=3, density=2)
        bench(f"powerset(n={n}, k=3, d=2)", fsa)
