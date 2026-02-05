"""
Benchmarking script for PTB (Penn Treebank) tokenizer transducer.

Uses the fastest available decomposition algorithm (Rust backend if available,
otherwise falls back to Python).
"""
from transduction.benchmarking.fsts.ptb_pynini import (
    build_ptb_fst_pynini,
    string_to_byte_strs,
    decode_ptb_output,
    SEP,
    MARKER,
)
from transduction.fsa import EPSILON
from transduction.benchmarking.data import load_wikitext, wikitext_detokenize

import tqdm
import time

import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


def decode_with_boundaries(output_tuple, boundary_char='|'):
    """Decode PTB FST output showing explicit word boundaries."""
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

# Try to import Rust backend, fall back to Python if not available
try:
    import transduction_core  # Test if Rust module is built
    from transduction.rust_bridge import RustDecomp
    HAS_RUST = True
    DECOMP_IMPL = "rust"
    print("Using Rust backend for decomposition")
except ImportError:
    HAS_RUST = False
    # Use NonrecursiveDFADecomp - faster than eager_nonrecursive.Precover
    from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp
    DECOMP_IMPL = "nonrecursive_dfa"
    print("Rust backend not available, using NonrecursiveDFADecomp (Python)")


def load_wikitext_paragraphs_ptb(fst, split, n=4, max_chars=None):
    """
    Load wikitext paragraphs and transduce through PTB FST.
    """
    dataset = load_wikitext(split)
    paragraphs = []
    original = []

    for item in dataset:
        text = item["text"].strip()
        if text and not text.startswith("="):
            detokenized = wikitext_detokenize(text)
            if max_chars is not None:
                detokenized = detokenized[:max_chars]

            byte_strs = string_to_byte_strs(detokenized)

            try:
                output_fsa = fst(byte_strs, None)
                transduced = next(output_fsa.language(tuple=True))
                paragraphs.append(transduced)
                original.append(detokenized)
            except StopIteration:
                print(f"  Skipping paragraph (FST rejected): {detokenized[:50]}...")
                continue
            except Exception as e:
                print(f"  Skipping paragraph (error: {e}): {detokenized[:50]}...")
                continue

            if len(paragraphs) >= n:
                break

    total_len = sum(len(p) for p in paragraphs)
    for i, para in enumerate(paragraphs):
        print(f"Paragraph {i+1} len {len(para)} cumulative length {sum(len(p) for p in paragraphs[:i+1])}")

    return paragraphs, original, total_len


def benchmark_precover(fst, target):
    """
    Benchmark precover computation using the fastest available backend.
    """
    t0 = time.perf_counter()

    if HAS_RUST:
        result = RustDecomp(fst, target)
        Q = result.quotient
        R = result.remainder
    else:
        result = NonrecursiveDFADecomp(fst, target)
        Q = result.quotient
        R = result.remainder

    t1 = time.perf_counter()

    return {
        'time_ms': (t1 - t0) * 1000,
        'quotient_states': len(Q.states),
        'quotient_final': len(Q.stop),
        'remainder_states': len(R.states),
        'remainder_final': len(R.stop),
    }


def main():
    print("=" * 60)
    print("PTB Tokenizer Benchmarking")
    print("=" * 60)

    # Build PTB FST
    print("\nBuilding PTB FST...")
    t0 = time.perf_counter()
    fst = build_ptb_fst_pynini()
    t1 = time.perf_counter()
    print(f"PTB FST built in {t1-t0:.2f}s")
    print(f"  States: {len(fst.states)}")
    print(f"  Input alphabet size: {len(fst.A)}")
    print(f"  Output alphabet size: {len(fst.B)}")

    # Quick sanity test
    test_str = "Hello, world!"
    test_bytes = string_to_byte_strs(test_str)
    try:
        output_fsa = fst(test_bytes, None)
        test_output = next(output_fsa.language(tuple=True))
        print(f"\n  Sanity test: '{test_str}' -> '{decode_ptb_output(test_output)}'")
    except Exception as e:
        print(f"\n  Sanity test failed: {e}")

    # Load data
    n_pgs = 10
    max_chars = 1000
    print(f"\nLoading {n_pgs} wikitext paragraphs (max {max_chars} chars each)...")

    pgs, original, total_len = load_wikitext_paragraphs_ptb(
        fst, "test", n=n_pgs, max_chars=max_chars
    )
    print(f"Loaded {total_len} total output symbols")

    if not pgs:
        print("No paragraphs loaded, exiting.")
        return {}

    # Run benchmarking
    result = {}

    for idx, pg in enumerate(pgs):
        result[idx] = []
        print(f"\n--- Paragraph {idx + 1}/{len(pgs)} ---")
        print(f"Original ({len(original[idx])} chars):")
        print(f"  {original[idx][:100]}...")

        # Show tokenized output with explicit boundaries
        sep_count = sum(1 for sym in pg if sym == SEP)
        print(f"\nTokenized ({sep_count + 1} tokens, {len(pg)} symbols):")
        tokenized = decode_with_boundaries(pg)
        # Show first 150 chars of tokenized output
        print(f"  {tokenized}")

        # Test various prefix lengths
        test_lengths = list(range(2, min(len(pg), 30), 4))

        for i in tqdm.tqdm(test_lengths, desc=f"Paragraph {idx+1}"):
            target = pg[:i]

            try:
                metrics = benchmark_precover(fst, target)
                result[idx].append({
                    'target_len': i,
                    **metrics
                })

            except Exception as e:
                print(f"\n  Error at len={i}: {e}")
                import traceback
                traceback.print_exc()
                result[idx].append({
                    'target_len': i,
                    'error': str(e),
                })

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for idx, res_list in result.items():
        print(f"\nParagraph {idx + 1}:")
        for r in res_list:
            if 'error' in r:
                print(f"  len={r['target_len']}: ERROR - {r['error']}")
            else:
                print(f"  len={r['target_len']}: Q={r['quotient_states']} states "
                      f"({r['quotient_final']} final), R={r['remainder_states']} states, "
                      f"time={r['time_ms']:.1f}ms")

    return result


if __name__ == "__main__":
    main()
