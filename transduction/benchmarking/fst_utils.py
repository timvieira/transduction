"""FST utility functions for the benchmarking package."""

from collections import deque

from transduction.benchmarking.fsts.ptb_pynini import (
    string_to_byte_strs,
    SEP,
)
from transduction.benchmarking.data import load_wikitext, wikitext_detokenize
from transduction.fsa import EPSILON
from transduction.fst import FST


def fst_output_language(fst, byte_strs):
    """Compose FST with input byte_strs and enumerate output language as tuples.

    Workaround: FST.__call__ and FSA.language don't support tuple-valued
    symbols, so we compose via FST.diag and do BFS with tuple accumulation.
    """
    input_fst = FST.from_string(byte_strs)
    composed = (input_fst @ fst).project(1)
    worklist = deque([(s, ()) for s in composed.start])
    while worklist:
        state, path = worklist.popleft()
        if state in composed.stop:
            yield path
        for a, j in composed.arcs(state):
            if a == EPSILON:
                worklist.append((j, path))
            else:
                worklist.append((j, path + (a,)))


def decode_symbol(sym):
    """Decode symbol to human-readable form."""
    if sym == SEP:
        return "|SEP|"
    if sym == "EOS":
        return "EOS"
    try:
        byte_val = int(sym)
        if 32 <= byte_val < 127:
            return f"'{chr(byte_val)}'"
        return f"[{byte_val}]"
    except ValueError:
        return str(sym)


def load_paragraphs(fst, n=10):
    """Load the first n wikitext paragraphs and compute their PTB outputs."""
    dataset = load_wikitext("test")
    paragraphs = []
    for item in dataset:
        text = item["text"].strip()
        if not text or text.startswith("="):
            continue
        detok = wikitext_detokenize(text)
        if len(detok) < 5:
            continue
        byte_strs = string_to_byte_strs(detok)
        try:
            output = next(fst_output_language(fst, byte_strs))
        except StopIteration:
            continue
        paragraphs.append({"text": detok, "output": output})
        if len(paragraphs) >= n:
            break
    return paragraphs
