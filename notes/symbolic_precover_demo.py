"""Demo: rho-arc factoring keeps the determinized PrecoverNFA small at BPE scale.

Compares standard PrecoverNFA + LazyDeterminize (which blows up at large
vocabs) against SymbolicLazyDeterminize (which stays small via rho arcs).
The rho-factored DFA is exact (same language as the standard DFA).

Uses synthetic BPE FSTs (no transformers dependency) to demonstrate the
arc reduction at various vocab sizes and target lengths.
"""
import time
import random

from transduction.fsa import EPSILON
from transduction.fst import FST
from transduction.lazy import LazyDeterminize
from transduction.precover_nfa import PrecoverNFA
from transduction.symbolic_precover import (
    RHO, SymbolicLazyDeterminize, ExpandRho,
)
from transduction.util import set_memory_limit
set_memory_limit(4)

try:
    from transduction.rust_bridge import RustRhoDeterminize
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def synthetic_bpe_fst(tokens):
    """Build a BPE FST from byte-string tokens."""
    m = FST()
    m.add_start(())
    for i, tok in enumerate(tokens):
        bx = tuple(tok)
        for j in range(len(bx)):
            m.add_arc(bx[:j], EPSILON, bx[j], bx[:j+1])
        m.add_arc(bx, i, EPSILON, ())
    m.add_stop(())
    return m


def make_vocab(n_single=95, n_two=0, n_three=0, n_four=0, seed=42):
    """Generate a synthetic BPE vocabulary."""
    random.seed(seed)
    tokens = set()
    for b in range(32, 32 + min(n_single, 95)):
        tokens.add(bytes([b]))
    for _ in range(n_two):
        tokens.add(bytes([random.choice(range(32, 127)),
                          random.choice(range(32, 127))]))
    for _ in range(n_three):
        tokens.add(bytes([random.choice(range(32, 127)) for _ in range(3)]))
    for _ in range(n_four):
        tokens.add(bytes([random.choice(range(32, 127)) for _ in range(4)]))
    return sorted(tokens)


# =============================================================================
print('=' * 90)
print('DEMO: Rho-arc factoring for BPE precover DFA')
print('=' * 90)

# --- Small/medium scale with equality verification ---
print()
print('--- Equality verified: varying vocab size (target = "Hi") ---')
for n2, n3 in [(0, 0), (50, 0), (200, 50), (500, 200)]:
    v = make_vocab(95, n2, n3)
    fst = synthetic_bpe_fst(v)
    alphabet = fst.A - {EPSILON}
    target = tuple(b'Hi')

    t0 = time.perf_counter()
    sym_dfa = SymbolicLazyDeterminize(PrecoverNFA(fst, target), alphabet)
    sym_mat = sym_dfa.materialize()
    py_elapsed = (time.perf_counter() - t0) * 1000
    sym_arcs = sum(1 for s in sym_mat.states for _ in sym_mat.arcs(s))
    rho_arcs = sum(1 for s in sym_mat.states for a, _ in sym_mat.arcs(s) if a is RHO)

    std_mat = LazyDeterminize(PrecoverNFA(fst, target)).materialize()
    std_arcs = sum(1 for s in std_mat.states for _ in std_mat.arcs(s))

    rho_expanded = ExpandRho(sym_dfa, alphabet).materialize()
    eq = std_mat.min().equal(rho_expanded.min())
    ok = "OK" if eq else "FAIL"

    rust_part = ''
    if HAS_RUST:
        rust_rho = RustRhoDeterminize(fst, target)
        rust_arcs = rust_rho.total_arcs
        rust_ms = rust_rho.total_ms
        speedup = py_elapsed / rust_ms if rust_ms > 0 else float('inf')
        rust_part = f' | Rust: {rust_arcs:5d} arcs, {rust_ms:.1f}ms, speedup={speedup:.0f}x'

    print(f'  V={len(v):5d} | Python: {sym_arcs:5d} arcs ({rho_arcs} RHO), {py_elapsed:.1f}ms | '
          f'Std: {std_arcs:5d} arcs | ratio={sym_arcs/std_arcs*100:.1f}% | eq={ok}{rust_part}')

# --- Varying target length ---
print()
print('--- Equality verified: varying target length (V~400) ---')
v = make_vocab(95, 200, 100)
fst = synthetic_bpe_fst(v)
alphabet = fst.A - {EPSILON}
for target_bytes in [b'H', b'Hi', b'Hel', b'Hell', b'Hello']:
    target = tuple(target_bytes)

    t0 = time.perf_counter()
    sym_dfa = SymbolicLazyDeterminize(PrecoverNFA(fst, target), alphabet)
    sym_mat = sym_dfa.materialize()
    py_elapsed = (time.perf_counter() - t0) * 1000
    sym_arcs = sum(1 for s in sym_mat.states for _ in sym_mat.arcs(s))

    std_mat = LazyDeterminize(PrecoverNFA(fst, target)).materialize()
    std_arcs = sum(1 for s in std_mat.states for _ in std_mat.arcs(s))

    rho_expanded = ExpandRho(sym_dfa, alphabet).materialize()
    eq = std_mat.min().equal(rho_expanded.min())
    ok = "OK" if eq else "FAIL"

    rust_part = ''
    if HAS_RUST:
        rust_rho = RustRhoDeterminize(fst, target)
        rust_arcs = rust_rho.total_arcs
        rust_ms = rust_rho.total_ms
        speedup = py_elapsed / rust_ms if rust_ms > 0 else float('inf')
        rust_part = f' | Rust: {rust_arcs} arcs, {rust_ms:.1f}ms, {speedup:.0f}x'

    print(f'  target={repr(target_bytes):10s} ({len(target)}B) | '
          f'{len(sym_mat.states)} st, {sym_arcs:3d} rho-arcs vs {std_arcs:4d} std-arcs | '
          f'ratio={sym_arcs/std_arcs*100:.1f}% | eq={ok}{rust_part}')

# --- Large scale (rho only, standard would OOM) ---
print()
print('--- Large scale: rho-factored only (standard DFA would OOM) ---')
for label, n2, n3, n4 in [
    ('2.6K', 2000, 500, 200),
    ('4K',   3000, 1000, 500),
]:
    v = make_vocab(95, n2, n3, n4)
    fst = synthetic_bpe_fst(v)
    alphabet = fst.A - {EPSILON}
    for target in [b'Hi', b'Hello']:
        target_t = tuple(target)
        t0 = time.perf_counter()
        sym_dfa = SymbolicLazyDeterminize(PrecoverNFA(fst, target_t), alphabet)
        sym_mat = sym_dfa.materialize()
        py_elapsed = (time.perf_counter() - t0) * 1000
        sym_arcs = sum(1 for s in sym_mat.states for _ in sym_mat.arcs(s))
        rho_arcs = sum(1 for s in sym_mat.states for a, _ in sym_mat.arcs(s) if a is RHO)

        rust_part = ''
        if HAS_RUST:
            rust_rho = RustRhoDeterminize(fst, target_t)
            rust_arcs = rust_rho.total_arcs
            rust_ms = rust_rho.total_ms
            speedup = py_elapsed / rust_ms if rust_ms > 0 else float('inf')
            rust_part = f' | Rust: {rust_arcs} arcs, {rust_ms:.1f}ms, {speedup:.0f}x'

        print(f'  V={len(v):5d} ({label}) target={repr(target):10s} | '
              f'{len(sym_mat.states)} st, {sym_arcs:3d} arcs ({rho_arcs} RHO), {py_elapsed:.0f}ms'
              f'{rust_part}')

print()
print('Key observations:')
print('  - Rho-factored DFA has 1-3% of the arcs of the standard DFA')
print('  - DFA state count = len(target) + 1, independent of vocab size')
print('  - Memory-efficient: handles V=4000+ without OOM (standard OOMs at V~1500)')
if HAS_RUST:
    print('  - Rust backend provides significant speedup over Python')
