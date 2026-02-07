#!/usr/bin/env python
"""
PTB (Penn Treebank) Tokenizer Benchmarking

Benchmarks precover decomposition algorithms on the PTB tokenizer FST.
Loads text from WikiText, transduces through the PTB FST, and measures
decomposition performance at various prefix lengths.

Usage:
    python -m benchmark.run_ptb --help
    python -m benchmark.run_ptb -n 2 -c 200 -m rust nonrecursive_dfa
    python -m benchmark.run_ptb -m rust -t 10 --max-memory 4096

Available methods:
    rust            - Rust-accelerated decomposition (fastest)
    rust_peekaboo   - Rust peekaboo algorithm
    nonrecursive_dfa - Python NonrecursiveDFADecomp
    peekaboo        - Python Peekaboo algorithm
    precover        - Python Precover (reference implementation)
"""

import argparse
import csv
import json
import resource
import signal
import time
from pathlib import Path

import tqdm

# Dataset config
import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

# Local imports
from benchmark.fsts.ptb_pynini import (
    build_ptb_fst_pynini,
    string_to_byte_strs,
    decode_ptb_output,
    SEP,
    MARKER,
)
from benchmark.data import load_wikitext, wikitext_detokenize
from transduction.fsa import EPSILON
from transduction.fst import FST

# Decomposition methods
from transduction.rust_bridge import RustDecomp, RustPeekaboo
from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp
from transduction.peekaboo_recursive import Peekaboo
from transduction.eager_nonrecursive import Precover

# Methods that take (fst, target) directly
DIRECT_METHODS = {
    'rust': RustDecomp,
    'nonrecursive_dfa': NonrecursiveDFADecomp,
    'precover': Precover,
}

# Methods that take (fst) then are called with (target) -> per-symbol results
PEEKABOO_METHODS = {
    'rust_peekaboo': RustPeekaboo,
    'peekaboo': Peekaboo,
}

METHODS = {**DIRECT_METHODS, **PEEKABOO_METHODS}

DEFAULT_METHOD = 'rust'

# Cache for peekaboo instances (they can be reused across targets)
_peekaboo_cache = {}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------



def decode_with_boundaries(output_tuple, boundary_char='|'):
    """
    Decode PTB FST output showing explicit word boundaries.

    Converts byte-string symbols back to UTF-8 text, inserting
    boundary_char between tokens.
    """
    tokens = []
    current_token = []

    for sym in output_tuple:
        if sym == SEP:
            if current_token:
                byte_vals = [int(b) for b in current_token]
                tokens.append(bytes(byte_vals).decode('utf-8', errors='replace'))
                current_token = []
            tokens.append(boundary_char)
        elif sym != MARKER and sym != EPSILON:
            current_token.append(sym)

    if current_token:
        byte_vals = [int(b) for b in current_token]
        tokens.append(bytes(byte_vals).decode('utf-8', errors='replace'))

    return ''.join(tokens)


# -----------------------------------------------------------------------------
# Resource limits
# -----------------------------------------------------------------------------

class TimeoutError(Exception):
    """Raised when an operation exceeds its time limit."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def set_memory_limit(max_mb):
    """Set process memory limit in megabytes."""
    if max_mb is not None and max_mb > 0:
        max_bytes = max_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_wikitext_paragraphs_ptb(fst, split, n=4, max_chars=None):
    """
    Load WikiText paragraphs and transduce through PTB FST.

    Args:
        fst: PTB tokenizer FST
        split: Dataset split ('train', 'test', 'validation')
        n: Number of paragraphs to load
        max_chars: Maximum characters per paragraph (None for unlimited)

    Returns:
        (paragraphs, original_texts, total_symbols)
        - paragraphs: list of output symbol tuples
        - original_texts: list of original text strings
        - total_symbols: total number of output symbols
    """
    dataset = load_wikitext(split)
    paragraphs = []
    original = []

    for item in dataset:
        text = item["text"].strip()
        if not text or text.startswith("="):
            continue

        detokenized = wikitext_detokenize(text)
        if max_chars is not None:
            detokenized = detokenized[:max_chars]

        byte_strs = string_to_byte_strs(detokenized)

        try:
            input_fst = FST.from_string(byte_strs)
            output_fsa = (input_fst @ fst).project(1)
            transduced = next(output_fsa.language(tuple=True))
            paragraphs.append(transduced)
            original.append(detokenized)
        except StopIteration:
            print(f"  Skipping (FST rejected): {detokenized[:50]}...")
            continue
        except Exception as e:
            print(f"  Skipping (error: {e}): {detokenized[:50]}...")
            continue

        if len(paragraphs) >= n:
            break

    total_len = sum(len(p) for p in paragraphs)
    for i, para in enumerate(paragraphs):
        cumulative = sum(len(p) for p in paragraphs[:i + 1])
        print(f"Paragraph {i + 1}: {len(para)} symbols (cumulative: {cumulative})")

    return paragraphs, original, total_len


# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------

def benchmark_precover(fst, target, method=None, timeout_sec=None):
    """
    Benchmark a single precover decomposition.

    Args:
        fst: The FST to decompose
        target: Target output prefix (tuple of symbols)
        method: Decomposition method name (default: DEFAULT_METHOD)
        timeout_sec: Timeout in seconds (None for no timeout)

    Returns:
        dict with keys:
            - method: method name
            - time_ms: execution time in milliseconds
            - quotient_states: number of states in quotient FSA
            - quotient_final: number of final states in quotient
            - remainder_states: number of states in remainder FSA
            - remainder_final: number of final states in remainder
            - num_symbols: (peekaboo only) number of symbols computed
        Or on error:
            - method: method name
            - error: error description
    """
    if method is None:
        method = DEFAULT_METHOD

    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    # Set up timeout handler
    old_handler = None
    if timeout_sec is not None:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_sec))

    try:
        t0 = time.perf_counter()

        if method in PEEKABOO_METHODS:
            # Peekaboo methods: reuse cached instance, call with target
            cache_key = (method, id(fst))
            if cache_key not in _peekaboo_cache:
                _peekaboo_cache[cache_key] = PEEKABOO_METHODS[method](fst)
            peekaboo = _peekaboo_cache[cache_key]
            per_symbol = peekaboo(target)
            t1 = time.perf_counter()

            # Aggregate stats across all symbols
            total_q_states = 0
            total_q_final = 0
            total_q_arcs = 0
            total_r_states = 0
            total_r_final = 0
            total_r_arcs = 0
            non_empty = 0
            for y, decomp in per_symbol.items():
                Q, R = decomp.quotient, decomp.remainder
                total_q_states += len(Q.states)
                total_q_final += len(Q.stop)
                total_q_arcs += sum(len(list(Q.arcs(s))) for s in Q.states)
                total_r_states += len(R.states)
                total_r_final += len(R.stop)
                total_r_arcs += sum(len(list(R.arcs(s))) for s in R.states)
                if len(Q.states) > 0 or len(R.states) > 0:
                    non_empty += 1

            return {
                'method': method,
                'time_ms': (t1 - t0) * 1000,
                'quotient_states': total_q_states,
                'quotient_final': total_q_final,
                'quotient_arcs': total_q_arcs,
                'remainder_states': total_r_states,
                'remainder_final': total_r_final,
                'remainder_arcs': total_r_arcs,
                'num_symbols': len(per_symbol),
                'non_empty_symbols': non_empty,
            }
        else:
            # Direct methods: call with (fst, target)
            result = DIRECT_METHODS[method](fst, target)
            Q, R = result.quotient, result.remainder
            t1 = time.perf_counter()

            q_arcs = sum(len(list(Q.arcs(s))) for s in Q.states)
            r_arcs = sum(len(list(R.arcs(s))) for s in R.states)

            return {
                'method': method,
                'time_ms': (t1 - t0) * 1000,
                'quotient_states': len(Q.states),
                'quotient_final': len(Q.stop),
                'quotient_arcs': q_arcs,
                'remainder_states': len(R.states),
                'remainder_final': len(R.stop),
                'remainder_arcs': r_arcs,
            }
    except TimeoutError:
        return {'method': method, 'error': f'timeout ({timeout_sec}s)'}
    except MemoryError:
        return {'method': method, 'error': 'out of memory'}
    finally:
        if timeout_sec is not None:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(n_pgs=1, max_chars=None, max_prefix_len=None,
         methods=None, timeout_sec=None, max_memory_mb=None, verbose=False,
         output_file=None):
    """
    Run PTB tokenizer benchmarking.

    Args:
        n_pgs: Number of paragraphs to load
        max_chars: Maximum characters per paragraph
        max_prefix_len: Maximum prefix length to test
        methods: List of method names to benchmark (default: all)
        timeout_sec: Timeout per decomposition in seconds
        max_memory_mb: Memory limit in MB
    """
    if methods is None:
        methods = list(METHODS.keys())

    # Validate methods
    invalid = [m for m in methods if m not in METHODS]
    if invalid:
        print(f"Error: Unknown methods {invalid}. Available: {list(METHODS.keys())}")
        return {}

    # Clear peekaboo cache
    _peekaboo_cache.clear()

    # Print configuration
    print("=" * 60)
    print("PTB Tokenizer Benchmarking")
    print("=" * 60)
    print(f"Methods: {', '.join(methods)}")
    if timeout_sec:
        print(f"Timeout: {timeout_sec}s per decomposition")
    if max_memory_mb:
        print(f"Memory limit: {max_memory_mb} MB")
        set_memory_limit(max_memory_mb)

    # Build PTB FST
    print("\nBuilding PTB FST...")
    t0 = time.perf_counter()
    fst = build_ptb_fst_pynini()
    build_time = time.perf_counter() - t0
    print(f"  Built in {build_time:.2f}s")
    print(f"  States: {len(fst.states)}")
    print(f"  Input alphabet: {len(fst.A)} symbols")
    print(f"  Output alphabet: {len(fst.B)} symbols")

    # Sanity test
    test_str = "Hello, world!"
    try:
        test_bytes = string_to_byte_strs(test_str)
        input_fst = FST.from_string(test_bytes)
        output_fsa = (input_fst @ fst).project(1)
        test_output = next(output_fsa.language(tuple=True))
        print(f"\n  Sanity test: '{test_str}' -> '{decode_ptb_output(test_output)}'")
    except Exception as e:
        print(f"\n  Sanity test FAILED: {e}")

    # Load data
    chars_desc = f"max {max_chars} chars" if max_chars else "unlimited"
    print(f"\nLoading {n_pgs} WikiText paragraphs ({chars_desc})...")
    pgs, original, total_len = load_wikitext_paragraphs_ptb(
        fst, "test", n=n_pgs, max_chars=max_chars
    )
    print(f"Total: {total_len} output symbols")

    if not pgs:
        print("No paragraphs loaded, exiting.")
        return {}

    # Run benchmarks
    results = {}

    for idx, pg in enumerate(pgs):
        results[idx] = []
        print(f"\n{'─' * 60}")
        print(f"Paragraph {idx + 1}/{len(pgs)}")
        print(f"{'─' * 60}")
        print(f"Text ({len(original[idx])} chars): {original[idx][:80]}...")

        sep_count = sum(1 for sym in pg if sym == SEP)
        print(f"Tokens: {sep_count + 1}, Symbols: {len(pg)}")
        print(f"Preview: {decode_with_boundaries(pg)[:80]}...")

        max_len = len(pg) if max_prefix_len is None else min(len(pg), max_prefix_len)
        test_lengths = list(range(1, max_len + 1))

        for prefix_len in tqdm.tqdm(test_lengths, desc=f"Paragraph {idx + 1}"):
            target = pg[:prefix_len]

            for method in methods:
                try:
                    metrics = benchmark_precover(
                        fst, target, method=method, timeout_sec=timeout_sec
                    )
                    metrics['target_len'] = prefix_len
                    results[idx].append(metrics)

                    if verbose:
                        if 'error' in metrics:
                            print(f"\n  [{method}] len={prefix_len}: ERROR - {metrics['error']}")
                        else:
                            m = metrics
                            q_info = f"Q: {m['quotient_states']} states, {m['quotient_arcs']} arcs, {m['quotient_final']} final"
                            r_info = f"R: {m['remainder_states']} states, {m['remainder_arcs']} arcs, {m['remainder_final']} final"
                            time_info = f"{m['time_ms']:.1f}ms"
                            if 'num_symbols' in m:
                                sym_info = f" | {m['non_empty_symbols']}/{m['num_symbols']} symbols non-empty"
                            else:
                                sym_info = ""
                            print(f"\n  [{method}] len={prefix_len}:")
                            print(f"    {q_info}")
                            print(f"    {r_info}")
                            print(f"    time: {time_info}{sym_info}")

                except Exception as e:
                    print(f"\n  Error: len={prefix_len} method={method}: {e}")
                    results[idx].append({
                        'target_len': prefix_len,
                        'method': method,
                        'error': str(e),
                    })

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for idx, res_list in results.items():
        print(f"\nParagraph {idx + 1}:")
        for r in res_list:
            method = r.get('method', '?')
            if 'error' in r:
                print(f"  len={r['target_len']:3d} [{method:16s}]: ERROR - {r['error']}")
            else:
                if 'num_symbols' in r:
                    extra = f" | {r['non_empty_symbols']}/{r['num_symbols']} syms"
                else:
                    extra = ""
                print(f"  len={r['target_len']:3d} [{method:16s}]: "
                      f"Q={r['quotient_states']:5d} st/{r['quotient_arcs']:5d} arcs ({r['quotient_final']:2d} final), "
                      f"R={r['remainder_states']:5d} st/{r['remainder_arcs']:5d} arcs, "
                      f"time={r['time_ms']:7.1f}ms{extra}")

    # Write output file if requested
    if output_file:
        output_path = Path(output_file)

        # Flatten results for output
        flat_results = []
        for idx, res_list in results.items():
            for r in res_list:
                flat_results.append({'paragraph': idx + 1, **r})

        if output_path.suffix == '.json':
            output_data = {
                'config': {
                    'n_paragraphs': n_pgs,
                    'max_chars': max_chars,
                    'max_prefix_len': max_prefix_len,
                    'methods': methods,
                    'timeout_sec': timeout_sec,
                },
                'fst_stats': {
                    'states': len(fst.states),
                    'input_alphabet': len(fst.A),
                    'output_alphabet': len(fst.B),
                },
                'results': flat_results,
            }
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults written to {output_path}")

        elif output_path.suffix == '.csv':
            if flat_results:
                fieldnames = list(flat_results[0].keys())
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flat_results)
                print(f"\nResults written to {output_path}")
        else:
            print(f"\nWarning: Unknown output format {output_path.suffix}, use .json or .csv")

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PTB Tokenizer Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available methods: {', '.join(METHODS.keys())}"
    )

    # Data options
    parser.add_argument("-n", "--n-paragraphs", type=int, default=1,
                        help="Number of paragraphs to load (default: 1)")
    parser.add_argument("-c", "--max-chars", type=int, default=None,
                        help="Max characters per paragraph (default: unlimited)")

    # Prefix options
    parser.add_argument("-p", "--max-prefix", type=int, default=None,
                        help="Max prefix length to test (default: full paragraph)")

    # Method options
    parser.add_argument("-m", "--methods", type=str, nargs="+",
                        choices=list(METHODS.keys()),
                        help="Methods to benchmark (default: all)")

    # Resource limits
    parser.add_argument("-t", "--timeout", type=float, default=None,
                        help="Timeout per decomposition in seconds")
    parser.add_argument("--max-memory", type=int, default=None,
                        help="Memory limit in MB")

    # Output options
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print each result as it completes")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (JSON if .json, CSV if .csv)")

    args = parser.parse_args()

    main(
        n_pgs=args.n_paragraphs,
        max_chars=args.max_chars,
        max_prefix_len=args.max_prefix,
        methods=args.methods,
        timeout_sec=args.timeout,
        max_memory_mb=args.max_memory,
        verbose=args.verbose,
        output_file=args.output,
    )
