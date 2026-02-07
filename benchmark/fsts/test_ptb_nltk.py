"""
Test PTB FST against NLTK TreebankWordTokenizer.

This script compares the FST tokenization against NLTK on wikitext paragraphs and checks for differences.

Usage:
    python -m benchmark.fsts.test_ptb_nltk
"""
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer

from benchmark.fsts.ptb_pynini import (
    build_ptb_fst_pynini,
    string_to_byte_strs,
    SEP,
)
from benchmark.data import load_wikitext, wikitext_detokenize


def fst_tokenize(fst, text):
    """Tokenize text using the FST."""
    # add_eos=True appends end-of-string marker for NLTK-compatible period handling
    byte_strs = string_to_byte_strs(text, add_eos=True)
    try:
        output = next(fst(byte_strs, None).language(tuple=True))
    except StopIteration:
        return None  # FST rejected input

    tokens = []
    current = []
    for sym in output:
        if sym == SEP:
            if current:
                tokens.append(bytes(int(b) for b in current).decode('utf-8', errors='replace'))
                current = []
        elif int(sym) < 256 and sym != '3':  # Skip EOS marker (byte 3)
            current.append(sym)
    if current:
        tokens.append(bytes(int(b) for b in current).decode('utf-8', errors='replace'))
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


def run_comparison(n_paragraphs=100, max_chars=500, verbose=True):
    """Run comparison between FST and NLTK on wikitext paragraphs."""
    print("Building PTB FST...")
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
