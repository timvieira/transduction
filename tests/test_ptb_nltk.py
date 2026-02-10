"""
Test PTB FST against NLTK TreebankWordTokenizer.

This script compares the FST tokenization against NLTK on wikitext paragraphs and checks for differences.

Usage:
    python -m pytest tests/test_ptb_nltk.py
"""
from collections import defaultdict

from nltk.tokenize import TreebankWordTokenizer
import pytest

from transduction.fst import FST
from transduction.applications.ptb import (
    build_ptb_fst_pynini,
    string_to_byte_strs,
    SEP,
    MARKER,
)
from transduction.applications.wikitext import load_wikitext, wikitext_detokenize
from transduction.fsa import EPSILON


def fst_tokenize(fst, text):
    """Tokenize text using the FST."""
    byte_strs = string_to_byte_strs(text)
    try:
        input_fst = FST.from_string(byte_strs)
        output_fsa = (input_fst @ fst).project(1)
        output = next(output_fsa.language(tuple=True))
    except StopIteration:
        return None  # FST rejected input

    tokens = []
    current = []
    for sym in output:
        if sym == SEP:
            if current:
                tokens.append(bytes(current).decode('utf-8', errors='replace'))
                current = []
        elif sym != MARKER and sym != EPSILON:
            current.append(sym)
    if current:
        tokens.append(bytes(current).decode('utf-8', errors='replace'))
    return tokens


def nltk_tokenize(tokenizer, text):
    """Tokenize text using NLTK."""
    return tokenizer.tokenize(text)


def categorize_difference(text, fst_tokens, nltk_tokens):
    """Categorize the type of difference between FST and NLTK."""
    # Find first differing token
    diff_idx = None
    for i in range(max(len(fst_tokens), len(nltk_tokens))):
        f = fst_tokens[i] if i < len(fst_tokens) else '<END>'
        n = nltk_tokens[i] if i < len(nltk_tokens) else '<END>'
        if f != n:
            diff_idx = i
            break

    if diff_idx is None:
        return 'unknown', None, None

    fst_tok = fst_tokens[diff_idx] if diff_idx < len(fst_tokens) else '<END>'
    nltk_tok = nltk_tokens[diff_idx] if diff_idx < len(nltk_tokens) else '<END>'

    # Categorize based on the difference
    if fst_tok == '.' or nltk_tok.endswith('.'):
        # Period handling difference
        if nltk_tok.endswith('.') and fst_tok == nltk_tok[:-1]:
            return 'period_not_separated', fst_tok, nltk_tok
        elif fst_tok == '.':
            return 'period_separated', fst_tok, nltk_tok

    if fst_tok in ['``', "''"] or nltk_tok in ['``', "''"]:
        return 'quote', fst_tok, nltk_tok

    if "'" in fst_tok or "'" in nltk_tok:
        return 'apostrophe', fst_tok, nltk_tok

    if fst_tok in list(',:;!?') or nltk_tok in list(',:;!?'):
        return 'punctuation', fst_tok, nltk_tok

    if fst_tok in list('[](){}') or nltk_tok in list('[](){}'):
        return 'bracket', fst_tok, nltk_tok

    return 'other', fst_tok, nltk_tok


def run_comparison(n_paragraphs=100, max_chars=500, verbose=True, fst=None):
    """Run comparison between FST and NLTK on wikitext paragraphs."""
    if fst is None:
        fst = build_ptb_fst_pynini()
    nltk_tok = TreebankWordTokenizer()

    print(f"Loading {n_paragraphs} wikitext paragraphs...")
    dataset = load_wikitext("test")

    paragraphs = []
    for item in dataset:
        text = item["text"].strip()
        if text and not text.startswith("="):
            detokenized = wikitext_detokenize(text)[:max_chars]
            if len(detokenized) > 20:
                paragraphs.append(detokenized)
            if len(paragraphs) >= n_paragraphs:
                break

    print(f"Testing {len(paragraphs)} paragraphs...\n")

    # Track results
    matches = 0
    mismatches = []
    errors = []
    categories = defaultdict(list)

    for i, text in enumerate(paragraphs):
        fst_tokens = fst_tokenize(fst, text)

        if fst_tokens is None:
            errors.append({'idx': i, 'text': text, 'error': 'FST rejected input'})
            continue

        nltk_tokens = nltk_tokenize(nltk_tok, text)

        if fst_tokens == nltk_tokens:
            matches += 1
        else:
            cat, fst_tok, nltk_tok_diff = categorize_difference(text, fst_tokens, nltk_tokens)
            mismatch = {
                'idx': i,
                'text': text,
                'fst': fst_tokens,
                'nltk': nltk_tokens,
                'category': cat,
                'fst_tok': fst_tok,
                'nltk_tok': nltk_tok_diff,
            }
            mismatches.append(mismatch)
            categories[cat].append(mismatch)

    # Print results
    total = len(paragraphs) - len(errors)
    print("=" * 70)
    print(f"RESULTS: {matches}/{total} paragraphs match ({100*matches/total:.1f}%)")
    print("=" * 70)

    if errors:
        print(f"\nErrors: {len(errors)}")
        for e in errors[:3]:
            print(f"  - {e['text'][:50]}... : {e['error']}")

    print(f"\nMismatches by category:")
    for cat in sorted(categories.keys(), key=lambda c: -len(categories[c])):
        print(f"  {cat}: {len(categories[cat])}")

    if verbose and mismatches:
        print("\n" + "=" * 70)
        print("SAMPLE MISMATCHES BY CATEGORY")
        print("=" * 70)

        for cat in sorted(categories.keys(), key=lambda c: -len(categories[c])):
            samples = categories[cat][:3]  # Show up to 3 examples per category
            print(f"\n--- {cat.upper()} ({len(categories[cat])} total) ---")

            for m in samples:
                print(f"\nText: {m['text'][:80]}...")
                print(f"FST:  {' '.join(m['fst'][:15])}{'...' if len(m['fst']) > 15 else ''}")
                print(f"NLTK: {' '.join(m['nltk'][:15])}{'...' if len(m['nltk']) > 15 else ''}")
                print(f"Diff: FST='{m['fst_tok']}' vs NLTK='{m['nltk_tok']}'")

    return {
        'matches': matches,
        'total': total,
        'mismatches': mismatches,
        'categories': dict(categories),
        'errors': errors,
    }


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ptb_fst():
    return build_ptb_fst_pynini()


def test_ptb_fst_matches_nltk(ptb_fst):
    """Test that PTB FST matches NLTK on wikitext paragraphs."""
    results = run_comparison(
        n_paragraphs=100, max_chars=500, verbose=True, fst=ptb_fst
    )
    total = results['total']
    matches = results['matches']
    assert total > 0, "No paragraphs were tested"
    # Allow up to 5% mismatch rate for known divergences
    match_rate = matches / total
    assert match_rate >= 0.95, (
        f"Match rate {match_rate:.1%} ({matches}/{total}) is below 95%. "
        f"Categories: {dict((k, len(v)) for k, v in results['categories'].items())}"
    )


def test_ptb_fst_simple_sentence(ptb_fst):
    """Test FST tokenization on a simple sentence."""
    nltk_tok = TreebankWordTokenizer()
    text = "Hello, world!"
    fst_tokens = fst_tokenize(ptb_fst, text)
    nltk_tokens = nltk_tok.tokenize(text)
    assert fst_tokens is not None, "FST rejected simple input"
    assert fst_tokens == nltk_tokens, f"FST={fst_tokens} vs NLTK={nltk_tokens}"


def test_ptb_fst_contractions(ptb_fst):
    """Test FST tokenization handles contractions."""
    nltk_tok = TreebankWordTokenizer()
    text = "I can't believe it's not butter."
    fst_tokens = fst_tokenize(ptb_fst, text)
    nltk_tokens = nltk_tok.tokenize(text)
    assert fst_tokens is not None, "FST rejected contractions input"
    assert fst_tokens == nltk_tokens, f"FST={fst_tokens} vs NLTK={nltk_tokens}"


def test_ptb_fst_quotes(ptb_fst):
    """Test FST tokenization handles quotes."""
    nltk_tok = TreebankWordTokenizer()
    text = 'He said "hello" to her.'
    fst_tokens = fst_tokenize(ptb_fst, text)
    nltk_tokens = nltk_tok.tokenize(text)
    assert fst_tokens is not None, "FST rejected quotes input"
    assert fst_tokens == nltk_tokens, f"FST={fst_tokens} vs NLTK={nltk_tokens}"


def main():
    # Test on 1000 paragraphs by default
    # Increase n_paragraphs for more thorough testing
    results = run_comparison(n_paragraphs=1000, max_chars=1000, verbose=True)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Match rate: {results['matches']}/{results['total']} ({100*results['matches']/results['total']:.1f}%)")

    if results['matches'] == results['total']:
        print("\n*** ALL TESTS PASS! FST matches NLTK. ***")
    else:
        print(f"\nTo fix: Address {len(results['mismatches'])} mismatches")
        print("Categories to fix:")
        for cat, items in sorted(results['categories'].items(), key=lambda x: -len(x[1])):
            print(f"  - {cat}: {len(items)} cases")


if __name__ == "__main__":
    main()
