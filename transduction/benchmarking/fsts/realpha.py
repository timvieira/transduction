from genlm.backend.tokenization.bytes import get_byte_vocab

from transduction.fst import FST
from transduction.fsa import EPSILON
from transformers import AutoTokenizer


def build_realpha(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    byte_vocab = get_byte_vocab(tokenizer)
    tokens = sorted(tokenizer.vocab.items(), key=lambda x: x[1])

    start_state = 0

    fst = FST(start=(start_state,), stop=(start_state,))

    for bytes_vals, (token_str, token) in zip(byte_vocab, tokens):
        if token_str in tokenizer.all_special_tokens:
            continue

        current_state = start_state
        current_input = token
        for idx, byte_val in enumerate(bytes_vals):
            if idx == len(bytes_vals) - 1:
                next_state = start_state
            else:
                next_state = len(fst.states)
            fst.add_arc(
                current_state,
                current_input if idx == 0 else EPSILON,
                byte_val,
                next_state
            )
            current_state = next_state
            current_input = EPSILON
    
    return fst


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("gpt2")
    print("Built tokenizer for gpt2")
    fst = build_realpha("gpt2")
    print("Built FST for realpha, states ", len(fst.states))
    print("Input vocab", len(fst.A))
    print("Output vocab", len(tok.vocab) - len(tok.all_special_tokens) - 1)
    # Subtract 1 for epsilon
    assert len(fst.A) - 1 == len(tok.vocab) - len(tok.all_special_tokens)
    assert len(fst.B) == 256
    print("Vocab looks healthy")