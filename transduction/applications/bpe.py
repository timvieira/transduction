from transduction.lazy import EPSILON
from transduction.fst import FST

from transduction.lm.statelm import decode_hf_tokenizer



def bpe_wfst(tokenizer, readable=False):
    """Build an FST that maps BPE token-ID sequences to byte sequences.

    For each non-special token in the vocabulary, the FST has a chain of
    epsilon-input arcs that spell out the token's bytes one at a time, then
    an arc that consumes the token ID (returning to the start state).  The
    start state ``()`` is also the sole accept state, so the FST loops over
    consecutive tokens.

    Args:
        tokenizer: A HuggingFace tokenizer instance.  Its vocabulary is
            decoded via ``decode_hf_tokenizer``; special tokens are skipped.
        readable: If ``True``, input-side labels are ``Token`` objects that
            display both the integer ID and byte string (useful for
            visualization).  If ``False`` (default), labels are raw ``int``
            token IDs.

    Returns:
        FST: Input alphabet = token IDs (or ``Token``), output alphabet =
        single bytes.
    """
    m = FST()
    m.add_start(())
    drop = {x.encode() for x in tokenizer.all_special_tokens}
    _, _, _decode, _ = decode_hf_tokenizer(tokenizer)
    for i, x in enumerate(_decode):
        if x in drop:
            continue
        _x = x
        x = tuple(x)
        for j in range(len(x)):
            m.add_arc(x[:j], EPSILON, bytes([x[j]]), x[:j+1])
        if readable:
            m.add_arc(x, Token(i, _x), EPSILON, ())
        else:
            m.add_arc(x, i, EPSILON, ())
    m.add_stop(())
    return m


class Token:
    def __init__(self, i, bytes):
        self.i = i
        self.bytes = bytes
    def __repr__(self):
        return f'{str(self.bytes)[2:-1]}/{self.i}'
    def __hash__(self):
        return hash(self.i)
    def __eq__(self, other):
        if isinstance(other, Token):
            return self.i == other.i
        else:
            return self.i == other
    def __radd__(self, other):
        if isinstance(other, int):
            return (other, self)
        else:
            return (*other, self)
    def __add__(self, other):
        if isinstance(other, (int, Token)):
            return (self, other)
        else:
            return (self, *other)
