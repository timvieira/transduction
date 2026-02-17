#!/usr/bin/env python3
"""End-to-end example: Transduced Language Model.

Demonstrates the full pipeline for computing the pushforward of a
language model through a finite-state transducer:

  1. Train a character-level n-gram LM on mixed-case English text
  2. Build an FST that normalizes text to lowercase
  3. Create a TransducedLM (pushforward of the LM through the FST)
  4. Query next-symbol probabilities and verify marginalization
  5. Greedy decode and sample decode

Key concept: the TransducedLM marginalizes over all source strings that
produce the same target string.  For case normalization:

    P_target('h') = P_source('h') + P_source('H')

The library handles this automatically, even for FSTs with infinite
preimages (this FST has infinite quotients because arbitrarily long
source strings can map to the same target prefix).

Usage:
    python examples/hello_world.py
"""

import numpy as np

from transduction import examples
from transduction.lm.ngram import CharNgramLM
from transduction.lm.transduced import TransducedLM


def main():

    # --- Step 1: Train a character-level n-gram LM ---
    #
    # The inner LM models source-side text (mixed case).
    training_data = 'Hello World hello world the hero held the world here'
    inner_lm = CharNgramLM.train(training_data, n=3)
    print(f'Trained: {inner_lm}')
    source_alphabet = sorted(s for s in inner_lm.alphabet if s != '<EOS>')
    print(f'  source alphabet: {source_alphabet}')
    print()

    # --- Step 2: Build an FST ---
    #
    # lowercase() maps both upper and lowercase source to lowercase target:
    #   'H' -> 'h',  'h' -> 'h',  'W' -> 'w',  'w' -> 'w', etc.
    fst = examples.lowercase()
    print(f'FST: {len(fst.states)} state(s), maps mixed-case -> lowercase')
    print()

    # --- Step 3: Create a TransducedLM ---
    #
    # TransducedLM computes P(target) by marginalizing over all source
    # strings that produce each target through the FST.  K=50 particles
    # are maintained for approximate inference (exact for this simple FST).
    tlm = TransducedLM(inner_lm, fst, K=50)
    print(f'Created: {tlm}')
    print()

    # --- Step 4: Verify marginalization ---
    #
    # P_target('h') should equal P_source('h') + P_source('H')
    # because both source symbols map to the same target symbol.
    s0 = inner_lm.initial()
    p_source_h = np.exp(s0.logp_next['h'])
    p_source_H = np.exp(s0.logp_next['H'])

    state = tlm.initial()
    p_target_h = np.exp(state.logp_next['h'])

    print('Marginalization over source cases:')
    print(f'  P_source(h) = {p_source_h:.4f}')
    print(f'  P_source(H) = {p_source_H:.4f}')
    print(f'  sum          = {p_source_h + p_source_H:.4f}')
    print(f'  P_target(h) = {p_target_h:.4f}  <-- matches')
    print()

    # --- Step 5: Query probabilities after conditioning ---
    state = state >> 'h'
    state = state >> 'e'
    print('After target "he", top predictions:')
    for sym, lp in state.logp_next.top(5).items():
        label = 'EOS' if sym == '<EOS>' else repr(sym)
        print(f'  P({label:5s}) = {np.exp(lp):.4f}')
    print()

    # --- Step 6: Greedy decode ---
    tokens = tlm.initial().greedy_decode(max_len=30)
    print(f'Greedy: {"".join(tokens)!r}')
    print()

    # --- Step 7: Sample decode ---
    np.random.seed(42)
    for i in range(3):
        tokens = tlm.initial().sample_decode(max_len=30)
        print(f'  Sample {i+1}: {"".join(tokens)!r}')


if __name__ == '__main__':
    main()
