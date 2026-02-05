"""
Benchmark and profiling for peekaboo decomposition implementations.

Compares:
  - Python peekaboo_recursive.Peekaboo
  - Rust rust_peekaboo (via RustPeekaboo bridge)

Reports timing, per-step DFA state counts, arena sizes, and other metrics.
"""

import time
import sys
from collections import defaultdict

from transduction import examples, EPSILON
from transduction.peekaboo_recursive import Peekaboo as PythonPeekaboo
from transduction.rust_bridge import RustPeekaboo, to_rust_fst
import transduction_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_targets(fst, depths):
    """Generate target strings by DFS through output symbols."""
    alphabet = sorted(fst.B - {EPSILON})
    targets = {'': 0}
    frontier = ['']
    for d in range(max(depths)):
        next_frontier = []
        for t in frontier:
            for y in alphabet:
                s = t + y
                if s not in targets:
                    targets[s] = d + 1
                    next_frontier.append(s)
        frontier = next_frontier
        if not frontier:
            break
    return targets


def time_python_peekaboo(fst, target, warmup=1, trials=3):
    """Time the Python peekaboo on a single target call."""
    peekaboo = PythonPeekaboo(fst)
    # warmup
    for _ in range(warmup):
        peekaboo(target)
    # timed
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        peekaboo(target)
        times.append(time.perf_counter() - t0)
    return min(times)


def time_rust_peekaboo(fst, target, warmup=1, trials=3):
    """Time the Rust peekaboo on a single target call, returning (time, stats)."""
    rust_fst, sym_map, state_map = to_rust_fst(fst)
    target_u32 = [sym_map(y) for y in target]
    # warmup
    for _ in range(warmup):
        transduction_core.rust_peekaboo(rust_fst, target_u32)
    # timed
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        result = transduction_core.rust_peekaboo(rust_fst, target_u32)
        times.append(time.perf_counter() - t0)
    stats = result.stats()
    return min(times), stats


def time_rust_peekaboo_incremental(fst, targets, warmup=0, trials=1):
    """Time multiple sequential calls (simulating autoregressive decoding)."""
    rust_fst, sym_map, state_map = to_rust_fst(fst)

    for _ in range(warmup):
        for t in targets:
            target_u32 = [sym_map(y) for y in t]
            transduction_core.rust_peekaboo(rust_fst, target_u32)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for t in targets:
            target_u32 = [sym_map(y) for y in t]
            transduction_core.rust_peekaboo(rust_fst, target_u32)
        times.append(time.perf_counter() - t0)
    return min(times)


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

EXAMPLES = {
    'replace_5':       lambda: examples.replace([('1','a'),('2','b'),('3','c'),('4','d'),('5','e')]),
    'delete_b':        lambda: examples.delete_b(),
    'samuel':          lambda: examples.samuel_example(),
    'small':           lambda: examples.small(),
    'sdd1':            lambda: examples.sdd1_fst(),
    'duplicate_5':     lambda: examples.duplicate(set('12345')),
    'number_comma':    lambda: examples.number_comma_separator({'a', ',', ' ', '0'}, Digit={'0'}),
    'newspeak2':       lambda: examples.newspeak2(),
    'lookahead':       lambda: examples.lookahead(),
    'weird_copy':      lambda: examples.weird_copy(),
    'triplets_of_doom': lambda: examples.triplets_of_doom(),
    'infinite_quotient': lambda: examples.infinite_quotient(),
    'parity_ab':       lambda: examples.parity({'a', 'b'}),
}

TARGET_DEPTHS = {
    'replace_5':       [0, 1, 2, 3, 4],
    'delete_b':        [0, 2, 5, 10],
    'samuel':          [0, 1, 3, 5],
    'small':           [0, 1, 3, 5],
    'sdd1':            [0, 1, 3, 5],
    'duplicate_5':     [0, 1, 3, 5],
    'number_comma':    [0, 1, 3],
    'newspeak2':       [0, 1, 3],
    'lookahead':       [0, 1, 3, 6],
    'weird_copy':      [0, 1, 3, 5],
    'triplets_of_doom': [0, 3, 6, 10, 13],
    'infinite_quotient': [0, 1, 3, 5],
    'parity_ab':       [0, 1, 3, 5],
}


def run_single_benchmark(name, fst, target, py_timeout=10.0):
    """Run benchmark on a single (fst, target) pair."""
    target_len = len(target)

    # Rust
    rust_time, stats = time_rust_peekaboo(fst, target, warmup=1, trials=3)

    # Python (skip if likely too slow)
    py_time = None
    if target_len <= 15:
        try:
            # Quick test if it's fast enough
            t0 = time.perf_counter()
            PythonPeekaboo(fst)(target)
            probe = time.perf_counter() - t0
            if probe < py_timeout:
                py_time = time_python_peekaboo(fst, target, warmup=1, trials=3)
        except Exception:
            pass

    return rust_time, py_time, stats


def format_time(seconds):
    if seconds is None:
        return '   N/A    '
    if seconds < 0.001:
        return f'{seconds*1e6:7.1f} us'
    elif seconds < 1.0:
        return f'{seconds*1e3:7.2f} ms'
    else:
        return f'{seconds:7.2f} s '


def main():
    print("=" * 100)
    print("Peekaboo Decomposition Benchmark")
    print("=" * 100)
    print()

    # --- Part 1: Timing comparison ---
    print("PART 1: Timing Comparison (Rust vs Python)")
    print("-" * 100)
    print(f"{'Example':<20} {'target_len':>10} {'Rust':>12} {'Python':>12} {'Speedup':>10}")
    print("-" * 100)

    all_results = []

    for name in EXAMPLES:
        fst = EXAMPLES[name]()
        alphabet = sorted(fst.B - {EPSILON})
        depths = TARGET_DEPTHS.get(name, [0, 1, 3])

        for depth in depths:
            # Build a target at the given depth
            target = ''
            for d in range(depth):
                target += alphabet[d % len(alphabet)]

            rust_time, py_time, stats = run_single_benchmark(name, fst, target)

            speedup = ''
            if py_time is not None and rust_time > 0:
                s = py_time / rust_time
                speedup = f'{s:>8.1f}x'

            print(f"{name:<20} {len(target):>10} {format_time(rust_time):>12} {format_time(py_time):>12} {speedup:>10}")
            all_results.append((name, len(target), rust_time, py_time, stats))

        print()

    # --- Part 2: Rust Profiling Details ---
    print()
    print("PART 2: Rust Profiling Details")
    print("-" * 100)
    print(f"{'Example':<20} {'tgt_len':>7} {'total':>8} {'init':>8} {'bfs':>8} {'extract':>8} "
          f"{'steps':>5} {'visited':>8} {'arena':>6} {'max_ps':>6} {'avg_ps':>6} "
          f"{'uni_t':>5} {'uni_f':>5} {'inc_arcs':>8}")
    print("-" * 100)

    for name, tgt_len, rust_time, py_time, stats in all_results:
        print(f"{name:<20} {tgt_len:>7} "
              f"{stats.total_ms:>7.2f}m "
              f"{stats.init_ms:>7.2f}m "
              f"{stats.bfs_ms:>7.2f}m "
              f"{stats.extract_ms:>7.2f}m "
              f"{stats.num_steps:>5} "
              f"{stats.total_bfs_visited:>8} "
              f"{stats.arena_size:>6} "
              f"{stats.max_powerset_size:>6} "
              f"{stats.avg_powerset_size:>6.1f} "
              f"{stats.universal_true:>5} "
              f"{stats.universal_false:>5} "
              f"{stats.merged_incoming_arcs:>8}")

    # --- Part 3: Per-step breakdown for selected examples ---
    print()
    print("PART 3: Per-Step BFS Breakdown (selected examples)")
    print("-" * 100)

    interesting = [r for r in all_results if r[1] >= 3]
    for name, tgt_len, rust_time, py_time, stats in interesting[:8]:
        print(f"\n{name} (target_len={tgt_len}):")
        print(f"  Step  Frontier  Visited")
        for i, (fsize, visited) in enumerate(zip(stats.per_step_frontier_size, stats.per_step_visited)):
            print(f"  {i:>4}  {fsize:>8}  {visited:>7}")

    # --- Part 4: BFS time breakdown ---
    print()
    print("PART 4: BFS Time Breakdown")
    print("-" * 100)
    print(f"{'Example':<20} {'tgt_len':>7} {'bfs_ms':>8} {'arcs_ms':>8} {'arcs_n':>7} "
          f"{'intern_ms':>9} {'intern_n':>8} {'univ_ms':>8} {'univ_n':>7}")
    print("-" * 100)

    for name, tgt_len, rust_time, py_time, stats in all_results:
        print(f"{name:<20} {tgt_len:>7} "
              f"{stats.bfs_ms:>7.2f}m "
              f"{stats.compute_arcs_ms:>7.2f}m "
              f"{stats.compute_arcs_calls:>7} "
              f"{stats.intern_ms:>8.2f}m "
              f"{stats.intern_calls:>8} "
              f"{stats.universal_ms:>7.2f}m "
              f"{stats.universal_calls:>7}")

    # --- Part 5: Incremental sequence timing ---
    print()
    print("PART 5: Incremental Sequence Timing (autoregressive simulation)")
    print("-" * 100)

    for name in ['triplets_of_doom', 'duplicate_5', 'parity_ab', 'lookahead']:
        fst = EXAMPLES[name]()
        alphabet = sorted(fst.B - {EPSILON})

        # Build sequence of targets: '', 'a', 'ab', 'abc', ...
        max_len = min(TARGET_DEPTHS[name][-1], 13)
        targets = []
        t = ''
        for d in range(max_len + 1):
            targets.append(t)
            if d < max_len:
                t += alphabet[d % len(alphabet)]

        total_rust = time_rust_peekaboo_incremental(fst, targets, warmup=1, trials=3)
        avg_rust = total_rust / len(targets)

        try:
            peekaboo = PythonPeekaboo(fst)
            t0 = time.perf_counter()
            for t in targets:
                peekaboo(t)
            total_py = time.perf_counter() - t0
            avg_py = total_py / len(targets)
        except Exception:
            total_py = None
            avg_py = None

        print(f"{name:<20}  {len(targets)} calls (targets len 0..{max_len})")
        print(f"  Rust total: {format_time(total_rust)},  avg/call: {format_time(avg_rust)}")
        if total_py is not None:
            speedup = total_py / total_rust if total_rust > 0 else float('inf')
            print(f"  Python total: {format_time(total_py)},  avg/call: {format_time(avg_py)}")
            print(f"  Speedup: {speedup:.1f}x")
        else:
            print(f"  Python: too slow / skipped")
        print()


if __name__ == '__main__':
    main()
