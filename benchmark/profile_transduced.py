#!/usr/bin/env python
"""
TransducedLM profiling benchmark.

Profiles TransducedLM's autoregressive decoding using the PTB tokenizer FST
and a character-level n-gram LM.  Measures per-step timing breakdown:
  - PeekabooState BFS (_ensure_bfs)
  - _compute_logp_next (priority queue expansion)
  - LM state advance (lm_state >> x)
  - Beam management

Usage:
    python -m benchmark.profile_transduced                 # default: short text
    python -m benchmark.profile_transduced --steps 20      # decode 20 steps
    python -m benchmark.profile_transduced --cprofile      # full cProfile dump
    python -m benchmark.profile_transduced --text "Hello"  # custom input text
    python -m benchmark.profile_transduced --simple        # use examples.small() instead of PTB
"""

import argparse
import time
import cProfile
import pstats
import io
import heapq
import numpy as np
from collections import defaultdict

from transduction.lm.ngram import CharNgramLM
from transduction.lm.transduced import TransducedLM, TransducedState, BeamItem, logsumexp
from transduction.lm.base import LogpNext


# ---------------------------------------------------------------------------
# Instrumented TransducedState
# ---------------------------------------------------------------------------

class ProfiledTransducedState(TransducedState):
    """TransducedState with timing instrumentation on _compute_logp_next."""

    # Class-level timing accumulators (reset between runs)
    timings = defaultdict(list)

    @classmethod
    def reset_timings(cls):
        cls.timings = defaultdict(list)

    def __rshift__(self, y):
        """Override to return ProfiledTransducedState instead of TransducedState."""
        if y == self.eos:
            raise ValueError(f"Out of vocabulary: {y!r}")
        self._ensure_computed()

        from transduction.lm.transduced import _to_key
        key = _to_key(y)
        if key is None or key not in self._logp_next_cache:
            raise ValueError(f"Out of vocabulary: {y!r}")

        lp_y = self._logp_next_cache[y]
        new_peekaboo = self._peekaboo_state >> key
        new_beam = self._carry_forward_cache.get(key, [])

        if len(new_beam) > self.tlm.max_beam:
            new_beam = sorted(new_beam, key=lambda it: it.weight, reverse=True)
            new_beam = new_beam[:self.tlm.max_beam]

        return ProfiledTransducedState(
            self.tlm, new_peekaboo, new_beam,
            self.logp + lp_y,
            history=(self.history, y),
        )

    def _compute_logp_next(self):
        """Instrumented version: times each phase of the computation."""

        t_start = time.perf_counter()

        # --- Phase 1: Access peekaboo decomposition (triggers BFS if needed) ---
        t0 = time.perf_counter()
        decomp = self._peekaboo_state.decomp
        dfa = self._peekaboo_state.dfa
        _target = self._peekaboo_state.target
        t_bfs = time.perf_counter() - t0

        # --- Phase 2: Build reverse lookups ---
        t0 = time.perf_counter()
        q_lookup = {}
        r_lookup = {}
        for y, d in decomp.items():
            for state in d.quotient:
                q_lookup.setdefault(state, set()).add(y)
            for state in d.remainder:
                r_lookup.setdefault(state, set()).add(y)

        resume_states = {}
        for y, states in self._peekaboo_state.resume_frontiers.items():
            for state in states:
                resume_states.setdefault(state, set()).add(y)

        preimage_lookup = self._peekaboo_state.preimage_stops
        t_lookup = time.perf_counter() - t0

        # --- Phase 3: Priority queue expansion ---
        scores = {}
        eos_scores = []
        carry_forward = {}

        EOS = self.tlm.inner_lm.eos if hasattr(self.tlm.inner_lm, 'eos') else self.tlm.inner_lm.initial().eos

        queue = list(self._beam)
        heapq.heapify(queue)

        t0 = time.perf_counter()
        steps = 0
        t_lm_advance = 0.0
        t_lm_logp = 0.0
        n_arcs_expanded = 0
        n_quotient_hits = 0
        n_remainder_hits = 0

        while queue and steps < self.tlm.max_steps:
            steps += 1
            item = heapq.heappop(queue)
            dfa_state = item.dfa_state
            lm_state = item.lm_state
            weight = item.weight

            t_lm0 = time.perf_counter()
            lm_logp_next = lm_state.logp_next
            t_lm_logp += time.perf_counter() - t_lm0

            if dfa_state in preimage_lookup:
                eos_scores.append(weight + lm_logp_next[EOS])

            is_quotient = False
            q_syms = q_lookup.get(dfa_state, set())
            if q_syms:
                for y in q_syms:
                    scores.setdefault(y, []).append(weight)
                    carry_forward.setdefault(y, []).append(item)
                is_quotient = True
                n_quotient_hits += 1

            if dfa_state in r_lookup:
                for y in r_lookup[dfa_state]:
                    if y not in q_syms:
                        scores.setdefault(y, []).append(weight + lm_logp_next[EOS])
                    carry_forward.setdefault(y, []).append(item)
                n_remainder_hits += 1

            if is_quotient:
                continue

            for x, next_dfa_state in dfa.arcs(dfa_state):
                next_weight = float(weight + lm_logp_next[x])
                if next_weight == -np.inf:
                    continue
                t_adv0 = time.perf_counter()
                next_lm_state = lm_state >> x
                t_lm_advance += time.perf_counter() - t_adv0
                next_item = BeamItem(
                    dfa_state=next_dfa_state,
                    lm_state=next_lm_state,
                    weight=next_weight,
                )
                heapq.heappush(queue, next_item)
                n_arcs_expanded += 1

        t_expand = time.perf_counter() - t0

        # --- Phase 4: Drain queue ---
        t0 = time.perf_counter()
        n_drained = 0
        while queue:
            n_drained += 1
            item = heapq.heappop(queue)
            dfa_state = item.dfa_state
            if dfa_state in preimage_lookup:
                lm_logp_next_drain = item.lm_state.logp_next
                eos_scores.append(item.weight + lm_logp_next_drain[EOS])
            if dfa_state in resume_states:
                for y in resume_states[dfa_state]:
                    carry_forward.setdefault(y, []).append(item)
            q_syms = q_lookup.get(dfa_state, set())
            if q_syms:
                for y in q_syms:
                    scores.setdefault(y, []).append(item.weight)
                    carry_forward.setdefault(y, []).append(item)
            if dfa_state in r_lookup:
                lm_logp_next = item.lm_state.logp_next
                for y in r_lookup[dfa_state]:
                    if y not in q_syms:
                        scores.setdefault(y, []).append(item.weight + lm_logp_next[EOS])
                    carry_forward.setdefault(y, []).append(item)
        t_drain = time.perf_counter() - t0

        # --- Phase 5: Normalize ---
        t0 = time.perf_counter()
        all_raw = []
        for y, s_list in scores.items():
            all_raw.append(logsumexp(s_list))
        eos_raw = logsumexp(eos_scores)
        if eos_raw > -np.inf:
            all_raw.append(eos_raw)
        Z = logsumexp(all_raw)

        normalized = {}
        for y, s_list in scores.items():
            normalized[y] = logsumexp(s_list) - Z
        eos_logp = eos_raw - Z if Z > -np.inf else -np.inf
        normalized[self.tlm.eos] = eos_logp
        t_normalize = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start

        self._logp_next_cache = LogpNext(normalized)
        self._carry_forward_cache = carry_forward

        # Record timing
        ProfiledTransducedState.timings['bfs_ms'].append(t_bfs * 1000)
        ProfiledTransducedState.timings['lookup_ms'].append(t_lookup * 1000)
        ProfiledTransducedState.timings['expand_ms'].append(t_expand * 1000)
        ProfiledTransducedState.timings['drain_ms'].append(t_drain * 1000)
        ProfiledTransducedState.timings['normalize_ms'].append(t_normalize * 1000)
        ProfiledTransducedState.timings['total_ms'].append(t_total * 1000)
        ProfiledTransducedState.timings['lm_advance_ms'].append(t_lm_advance * 1000)
        ProfiledTransducedState.timings['lm_logp_ms'].append(t_lm_logp * 1000)
        ProfiledTransducedState.timings['steps'].append(steps)
        ProfiledTransducedState.timings['arcs_expanded'].append(n_arcs_expanded)
        ProfiledTransducedState.timings['quotient_hits'].append(n_quotient_hits)
        ProfiledTransducedState.timings['remainder_hits'].append(n_remainder_hits)
        ProfiledTransducedState.timings['drained'].append(n_drained)
        ProfiledTransducedState.timings['beam_size'].append(len(self._beam))
        ProfiledTransducedState.timings['n_symbols'].append(len(normalized) - 1)  # exclude EOS
        ProfiledTransducedState.timings['carry_forward_total'].append(
            sum(len(v) for v in carry_forward.values())
        )


class ProfiledTransducedLM(TransducedLM):
    """TransducedLM that uses ProfiledTransducedState."""

    def initial(self):
        from transduction.peekaboo_incremental import PeekabooState as _DefaultDecompState
        from transduction.peekaboo_incremental import FstUniversality as _DefaultUniv

        peekaboo = self._decomp_state_cls(self.fst, (), parent=None, univ=self._univ)
        dfa = peekaboo.dfa
        start_states = list(dfa.start())
        inner_initial = self.inner_lm.initial()
        beam = [
            BeamItem(dfa_state=s, lm_state=inner_initial, weight=0.0)
            for s in start_states
        ]
        return ProfiledTransducedState(self, peekaboo, beam, 0.0)


# ---------------------------------------------------------------------------
# Benchmark setup
# ---------------------------------------------------------------------------

def remap_fst_to_single_chars(fst):
    """Remap an FST's multi-character symbols (like '84', '258') to single
    Unicode characters.  This reduces the alphabet size and keeps symbol
    representations compact, which improves performance for FSTs with large
    numeric token IDs.

    Returns (new_fst, fwd_map, inv_map) where fwd_map[old_sym] = new_char
    and inv_map[new_char] = old_sym.
    """
    from transduction.fst import FST as FSTClass
    from transduction.fsa import EPSILON

    # Build bijection: old symbol -> single Unicode char
    # Use private-use-area chars starting at U+E000 to avoid collisions
    fwd = {}
    inv = {}
    code = 0xE000
    for sym in sorted(fst.A | fst.B):
        if sym == EPSILON:
            continue
        fwd[sym] = chr(code)
        inv[chr(code)] = sym
        code += 1

    # Rebuild FST with remapped symbols
    new_fst = FSTClass()
    for s in fst.start:
        new_fst.add_start(s)
    for s in fst.stop:
        new_fst.add_stop(s)
    for s in fst.states:
        for x, y, j in fst.arcs(s):
            new_x = fwd.get(x, x) if x != EPSILON else EPSILON
            new_y = fwd.get(y, y) if y != EPSILON else EPSILON
            new_fst.add_arc(s, new_x, new_y, j)

    return new_fst, fwd, inv


def build_ptb_setup(text=None, max_chars=100):
    """Build PTB FST, remap to single-char symbols, train inner LM, and
    prepare a target sequence."""
    from transduction.applications.ptb import build_ptb_fst_pynini, string_to_byte_strs, decode_ptb_output
    from transduction.fst import FST
    from transduction.fsa import EPSILON

    print("Building PTB FST...")
    t0 = time.perf_counter()
    raw_fst = build_ptb_fst_pynini()
    print(f"  Built in {time.perf_counter() - t0:.2f}s")
    print(f"  States: {len(raw_fst.states)}, |A|={len(raw_fst.A)}, |B|={len(raw_fst.B)}")

    if text is None:
        text = "The quick brown fox jumps over the lazy dog."
    text = text[:max_chars]
    print(f"  Input text: {text!r}")

    # Remap multi-char symbols to single Unicode chars for compact representation
    print("  Remapping FST symbols to single chars...")
    fst, fwd_map, inv_map = remap_fst_to_single_chars(raw_fst)
    print(f"  Remapped FST: |A|={len(fst.A)}, |B|={len(fst.B)}")

    # Transduce to get a valid target sequence (in remapped symbols)
    byte_strs = string_to_byte_strs(text)
    remapped_input = tuple(fwd_map[s] for s in byte_strs)
    input_fst = FST.from_string(remapped_input)
    output_fsa = (input_fst @ fst).project(1)
    target_seq = next(output_fsa.language())
    print(f"  Target sequence length: {len(target_seq)}")

    # Decode for display
    decoded_syms = [inv_map.get(c, c) for c in target_seq]
    print(f"  Decoded: {decode_ptb_output(tuple(decoded_syms))!r}")

    # Train inner LM on source-side symbols (remapped single chars).
    training_text = (
        "The quick brown fox jumps over the lazy dog. "
        "A stitch in time saves nine. To be or not to be, that is the question. "
        "All that glitters is not gold. Actions speak louder than words. "
        "The pen is mightier than the sword. Knowledge is power. "
        "Practice makes perfect. Where there is a will, there is a way. "
        "It was the best of times, it was the worst of times. "
        "Call me Ishmael. In the beginning was the Word."
    )
    training_bytes = [fwd_map[s] for s in string_to_byte_strs(training_text)]
    # Ensure all remapped source symbols appear at least once
    source_alpha = fst.A - {EPSILON}
    for sym in source_alpha:
        training_bytes.append(sym)

    print(f"\n  Training CharNgramLM on {len(training_bytes)} symbols...")
    inner_lm = CharNgramLM.train(training_bytes, n=3, alpha=0.5)
    print(f"  LM: {inner_lm}  (alphabet size: {len(inner_lm.alphabet)})")

    return fst, inner_lm, target_seq


def build_example_setup(example_name='lowercase'):
    """Build an example FST setup for profiling."""
    from transduction import examples
    from transduction.fst import FST
    from transduction.fsa import EPSILON

    fst_builders = {
        'lowercase': lambda: examples.lowercase(),
        'duplicate': lambda: examples.duplicate(set('12345')),
        'number_comma': lambda: examples.number_comma_separator({'a', ',', ' ', '0'}, Digit={'0'}),
        'lookahead': lambda: examples.lookahead(),
        'parity': lambda: examples.parity({'a', 'b'}),
        'weird_copy': lambda: examples.weird_copy(),
        'sdd1': lambda: examples.sdd1_fst(),
    }

    if example_name not in fst_builders:
        raise ValueError(f"Unknown example: {example_name}. Available: {list(fst_builders)}")

    fst = fst_builders[example_name]()
    source_alpha = sorted(fst.A - {EPSILON})
    target_alpha = sorted(fst.B - {EPSILON})
    print(f"FST: {example_name}, States: {len(fst.states)}, |A|={len(fst.A)}, |B|={len(fst.B)}")
    print(f"  Source: {source_alpha[:10]}{'...' if len(source_alpha) > 10 else ''}")
    print(f"  Target: {target_alpha[:10]}{'...' if len(target_alpha) > 10 else ''}")

    # Train LM over source alphabet with enough data for realistic priors
    training_data = source_alpha * 100
    inner_lm = CharNgramLM.train(training_data, n=2, alpha=0.5)
    print(f"LM: {inner_lm}")

    # Generate a valid target by transducing a source string
    # Try progressively longer source strings until we get a long enough target
    target_seq = ()
    for length in [10, 20, 50]:
        source_str = tuple(source_alpha[i % len(source_alpha)] for i in range(length))
        try:
            input_fst = FST.from_string(source_str)
            output_fsa = (input_fst @ fst).project(1)
            target_seq = next(output_fsa.language())
            if len(target_seq) >= 10:
                break
        except StopIteration:
            continue

    if not target_seq:
        raise RuntimeError(f"Could not generate valid target for {example_name}")

    print(f"Target sequence ({len(target_seq)} symbols): {''.join(target_seq[:20])}{'...' if len(target_seq) > 20 else ''}")
    return fst, inner_lm, target_seq


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def run_profiled_decode(fst, inner_lm, target_seq, max_steps_per_symbol=2000,
                        max_beam=200, decode_steps=10):
    """Run TransducedLM decoding with profiling instrumentation."""

    ProfiledTransducedState.reset_timings()

    tlm = ProfiledTransducedLM(
        inner_lm, fst,
        max_steps=max_steps_per_symbol,
        max_beam=max_beam,
    )

    print(f"\nDecoding {decode_steps} steps (max_steps={max_steps_per_symbol}, max_beam={max_beam})...")

    state = tlm.initial()
    decoded = []
    step_times = []

    for i in range(min(decode_steps, len(target_seq))):
        y = target_seq[i]
        t0 = time.perf_counter()
        lp = state.logp_next[y]
        state = state >> y
        step_time = time.perf_counter() - t0
        step_times.append(step_time)
        decoded.append(y)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Step {i+1}: symbol={y!r}, logp={lp:.4f}, time={step_time*1000:.1f}ms")

    print(f"\n  Total decode time: {sum(step_times)*1000:.1f}ms")
    print(f"  Avg per step: {np.mean(step_times)*1000:.1f}ms")
    return step_times


def print_timing_report():
    """Print a detailed breakdown of timing from ProfiledTransducedState."""
    T = ProfiledTransducedState.timings
    n = len(T['total_ms'])

    if n == 0:
        print("No timing data collected.")
        return

    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN (per _compute_logp_next call)")
    print("=" * 80)

    def fmt_col(label, values, unit='ms'):
        arr = np.array(values)
        return f"  {label:<22s}  mean={arr.mean():8.2f}{unit}  "  \
               f"median={np.median(arr):8.2f}{unit}  "  \
               f"max={arr.max():8.2f}{unit}  "  \
               f"sum={arr.sum():8.1f}{unit}"

    print(f"\n  Calls: {n}")
    print(fmt_col("Total", T['total_ms']))
    print(fmt_col("BFS (peekaboo)", T['bfs_ms']))
    print(fmt_col("Lookup build", T['lookup_ms']))
    print(fmt_col("Queue expand", T['expand_ms']))
    print(fmt_col("  LM advance", T['lm_advance_ms']))
    print(fmt_col("  LM logp_next", T['lm_logp_ms']))
    print(fmt_col("Queue drain", T['drain_ms']))
    print(fmt_col("Normalize", T['normalize_ms']))

    print(f"\n  {'SEARCH STATS':<22s}")
    print(fmt_col("Expansion steps", T['steps'], unit=''))
    print(fmt_col("Arcs expanded", T['arcs_expanded'], unit=''))
    print(fmt_col("Quotient hits", T['quotient_hits'], unit=''))
    print(fmt_col("Remainder hits", T['remainder_hits'], unit=''))
    print(fmt_col("Queue drained", T['drained'], unit=''))
    print(fmt_col("Beam size (in)", T['beam_size'], unit=''))
    print(fmt_col("Symbols scored", T['n_symbols'], unit=''))
    print(fmt_col("Carry-fwd total", T['carry_forward_total'], unit=''))

    # Per-step table
    print("\n" + "-" * 100)
    print(f"  {'Step':>4s}  {'total':>8s}  {'BFS':>8s}  {'expand':>8s}  "
          f"{'LM>>':>8s}  {'LM.lp':>8s}  {'drain':>8s}  "
          f"{'steps':>5s}  {'arcs':>5s}  {'beam':>5s}  {'syms':>4s}")
    print("-" * 100)

    for i in range(n):
        print(f"  {i+1:>4d}  "
              f"{T['total_ms'][i]:>7.1f}m  "
              f"{T['bfs_ms'][i]:>7.1f}m  "
              f"{T['expand_ms'][i]:>7.1f}m  "
              f"{T['lm_advance_ms'][i]:>7.1f}m  "
              f"{T['lm_logp_ms'][i]:>7.1f}m  "
              f"{T['drain_ms'][i]:>7.1f}m  "
              f"{T['steps'][i]:>5.0f}  "
              f"{T['arcs_expanded'][i]:>5.0f}  "
              f"{T['beam_size'][i]:>5.0f}  "
              f"{T['n_symbols'][i]:>4.0f}")

    # Fraction breakdown
    total_sum = sum(T['total_ms'])
    if total_sum > 0:
        print(f"\n  TIME FRACTION:")
        for label, key in [
            ('BFS (peekaboo)', 'bfs_ms'),
            ('Lookup build', 'lookup_ms'),
            ('Queue expand', 'expand_ms'),
            ('  LM advance', 'lm_advance_ms'),
            ('  LM logp_next', 'lm_logp_ms'),
            ('Queue drain', 'drain_ms'),
            ('Normalize', 'normalize_ms'),
        ]:
            frac = sum(T[key]) / total_sum * 100
            bar = '#' * int(frac / 2)
            print(f"  {label:<22s} {frac:5.1f}%  {bar}")


def run_cprofile(fst, inner_lm, target_seq, max_steps_per_symbol, max_beam, decode_steps):
    """Run with cProfile and print top functions."""
    tlm = ProfiledTransducedLM(
        inner_lm, fst,
        max_steps=max_steps_per_symbol,
        max_beam=max_beam,
    )

    def do_decode():
        state = tlm.initial()
        for i in range(min(decode_steps, len(target_seq))):
            y = target_seq[i]
            _ = state.logp_next[y]
            state = state >> y

    pr = cProfile.Profile()
    pr.enable()
    do_decode()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(40)
    print("\n" + "=" * 80)
    print("cProfile (top 40 by cumulative time)")
    print("=" * 80)
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(30)
    print("=" * 80)
    print("cProfile (top 30 by total time)")
    print("=" * 80)
    print(s2.getvalue())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TransducedLM Profiling Benchmark")

    parser.add_argument("--example", type=str, default=None,
                        help="Use an example FST (lowercase, duplicate, number_comma, "
                             "lookahead, parity, weird_copy, sdd1)")
    parser.add_argument("--text", type=str, default=None,
                        help="Custom input text for PTB transduction")
    parser.add_argument("--max-chars", type=int, default=100,
                        help="Max characters of input text (default: 100)")
    parser.add_argument("--steps", type=int, default=15,
                        help="Number of decode steps (default: 15)")
    parser.add_argument("--max-search-steps", type=int, default=2000,
                        help="Max search steps per logp_next (default: 2000)")
    parser.add_argument("--max-beam", type=int, default=200,
                        help="Max beam size (default: 200)")
    parser.add_argument("--cprofile", action="store_true",
                        help="Run cProfile analysis")

    args = parser.parse_args()

    print("=" * 80)
    print("TransducedLM Profiling Benchmark")
    print("=" * 80)

    if args.example:
        fst, inner_lm, target_seq = build_example_setup(args.example)
    else:
        fst, inner_lm, target_seq = build_ptb_setup(
            text=args.text, max_chars=args.max_chars,
        )

    step_times = run_profiled_decode(
        fst, inner_lm, target_seq,
        max_steps_per_symbol=args.max_search_steps,
        max_beam=args.max_beam,
        decode_steps=args.steps,
    )

    print_timing_report()

    if args.cprofile:
        ProfiledTransducedState.reset_timings()
        run_cprofile(
            fst, inner_lm, target_seq,
            max_steps_per_symbol=args.max_search_steps,
            max_beam=args.max_beam,
            decode_steps=args.steps,
        )


if __name__ == "__main__":
    main()
