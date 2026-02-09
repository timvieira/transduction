import numpy as np
import torch
import transformers
from transformers.cache_utils import DynamicCache

from functools import cached_property


# ---------------------------------------------------------------------------
# Vocabulary decoding (ported from tokenization.vocab.decode)
# ---------------------------------------------------------------------------

# GPT-2's byte↔unicode mapping (output of `_bytes_to_unicode`)
_encode_bytes_str = [
    'Ā', 'ā', 'Ă', 'ă', 'Ą', 'ą', 'Ć', 'ć', 'Ĉ', 'ĉ', 'Ċ', 'ċ', 'Č', 'č', 'Ď', 'ď',
    'Đ', 'đ', 'Ē', 'ē', 'Ĕ', 'ĕ', 'Ė', 'ė', 'Ę', 'ę', 'Ě', 'ě', 'Ĝ', 'ĝ', 'Ğ', 'ğ',
    'Ġ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
    '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'ġ',
    'Ģ', 'ģ', 'Ĥ', 'ĥ', 'Ħ', 'ħ', 'Ĩ', 'ĩ', 'Ī', 'ī', 'Ĭ', 'ĭ', 'Į', 'į', 'İ', 'ı',
    'Ĳ', 'ĳ', 'Ĵ', 'ĵ', 'Ķ', 'ķ', 'ĸ', 'Ĺ', 'ĺ', 'Ļ', 'ļ', 'Ľ', 'ľ', 'Ŀ', 'ŀ', 'Ł',
    'ł', '¡', '¢', '£', '¤', '¥', '¦', '§', '¨', '©', 'ª', '«', '¬', 'Ń', '®', '¯',
    '°', '±', '²', '³', '´', 'µ', '¶', '·', '¸', '¹', 'º', '»', '¼', '½', '¾', '¿',
    'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï',
    'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß',
    'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï',
    'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ',
]

_default_byte_decoder = {s: i for i, s in enumerate(_encode_bytes_str)}


# TODO: Add some documentation here about what's going on and what assumptions
# are being made.  This decoding method might not work for all hugging face
# tokenizers.
def decode_hf_tokenizer(tokenizer):
    "Extract what we need from a HuggingFace tokenizer."
    _merges = []
    V = tokenizer.get_vocab()
    if hasattr(tokenizer, 'bpe_ranks'):
        for (u, v) in tokenizer.bpe_ranks:
            _merges.append((V[u], V[v], V[u + v]))
    else:
        import json
        subtokenizer_dict = json.loads(tokenizer._tokenizer.to_str())
        for (u, v) in subtokenizer_dict["model"]["merges"]:
            _merges.append((V[u], V[v], V[u + v]))

    if hasattr(tokenizer, 'byte_decoder'):
        byte_decoder = tokenizer.byte_decoder
    else:
        byte_decoder = _default_byte_decoder

    _encode = {}
    _decode = [None] * len(V)
    for bs, token_id in V.items():
        b = bytes([byte_decoder[b] for b in bs])
        _encode[b] = token_id
        _decode[token_id] = b

    _encode_byte = [None] * 256
    for i in range(256):
        _encode_byte[i] = _encode[bytes([i])]

    return (_merges, _encode, _decode, _encode_byte)


# ---------------------------------------------------------------------------
# Utility classes (ported from tokenization.util)
# ---------------------------------------------------------------------------

def flatten(xs):
    if len(xs) == 0:
        return ()
    else:
        ys, y = xs
        return flatten(ys) + (y,)


def unflatten(ys):
    xs = ()
    for y in ys:
        xs = (xs, y)
    return xs


class LazyProb:
    """Efficiently maps token bytes/ids to their probabilities in an LLM's
    next-token distribution without materializing a full dictionary."""

    def __init__(self, _p, encode, decode):
        self._p = _p
        self._encode = encode
        self._decode = decode

    def keys(self):
        return self._decode

    def values(self):
        return self._p

    def items(self):
        return zip(self._decode, self._p)

    def __contains__(self, token):
        if isinstance(token, int):
            return 0 <= token < len(self._decode)
        return token in self._encode

    def __getitem__(self, token):
        if isinstance(token, int):
            i = token
        else:
            i = self._encode.get(token)
        return self._p[i] if i is not None else 0

    def materialize(self, top=None):
        _p = self._p
        _decode = self._decode
        top_p = _p.argsort() if top is None else _p.argsort()[-int(top):]
        pp = {}
        for i in reversed(top_p):
            pp[_decode[i]] = _p[i]
        return pp

    def top(self, K):
        return self.materialize(top=K)

    def __repr__(self):
        return repr(self.materialize())

    def argmax(self):
        """Return the token with the highest log-probability."""
        return self._decode[self._p.argmax()]

    def sample(self):
        """Sample a token from the distribution."""
        logps = self._p.copy().astype(np.float64)
        logps -= logps.max()
        probs = np.exp(logps)
        probs /= probs.sum()
        return self._decode[np.random.choice(len(probs), p=probs)]

    def apply(self, f):
        return LazyProb(
            _p=f(self._p),
            encode=self._encode,
            decode=self._decode,
        )

    def copy(self):
        return self.apply(lambda x: x.copy())


# ---------------------------------------------------------------------------
# LM classes
# ---------------------------------------------------------------------------

# TODO: This class will encounter issues when its token vocabulary has multiple
# token_ids that map to the same string of bytes.
class TokenizedLLM:

    def __init__(self, tokenizer, model, byte_level=True):
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

        assert byte_level, "Only byte_level=True is supported"
        (_, self._encode, self._decode, _) = decode_hf_tokenizer(tokenizer)
        self.eos = self.tokenizer.eos_token.encode()

        self._calls = 0
        self.V = set(self._decode)

    def initial(self):
        return StateLM.initial(self)

    def encode_prompt(self, prompt):
        "Encode ``prompt`` as a tuple of tokens (each a bytes object)."
        return unflatten(tuple(self._decode[i] for i in self.tokenizer.encode(prompt)))


def load_model_by_name(model_name, device=None, **kwargs):
    """Load an LLM from HuggingFace into a ``TokenizedLLM``."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return TokenizedLLM(
        tokenizer=transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False),
        model=(
            transformers.AutoModelForCausalLM.from_pretrained(model_name)
            .to(device)
            .eval()
        ),
        **kwargs,
    )


from transduction.lm.base import LMState


class StateLM(LMState):
    """Immutable LM state for incremental decoding with KV cache sharing.

    ``state >> token`` returns a new state; the parent's cache is reused.
    Multiple children can branch from the same parent (e.g., in
    ``prioritized_enumeration``).  Linear chains (``importance_sampling``)
    create one child per state.

    KV Cache Safety
    ---------------
    Modern transformers (>=4.40) returns ``DynamicCache`` objects that
    ``model.forward()`` **mutates in-place** via ``cache.update()``.  If two
    children share a parent's cache directly, the first child's forward pass
    corrupts it for the second.

    Rejected alternatives:
    - ``copy.deepcopy``: slow, opaque, bad with CUDA tensors.
    - Recompute from scratch: defeats the purpose of incremental decoding.
    - Refcount / lazy clone: fragile — ``>>`` can be called *after* an earlier
      child already evaluated ``.out``, so the refcount at creation time doesn't
      predict future children.

    Chosen fix (TODO — not yet implemented):
    Store each state's KV tensors as **immutable tuples**.  Before each forward
    call, wrap the parent's tuples in a **fresh ``DynamicCache``** (just
    ``list.append`` of references — no tensor cloning).  The model's
    ``torch.cat`` inside ``cache.update()`` creates new tensors, leaving the
    parent's originals untouched.  Zero tensor clones, negligible overhead
    vs. forward pass, safe regardless of ``>>`` ordering.
    """

    def __init__(self, lm, logp, context, parent):
        self.lm = lm
        self.eos = lm.eos
        self.logp = logp
        self.context = context
        self.parent = parent

    def __rshift__(self, x):
        if x not in self.logp_next or x == self.eos:
            raise ValueError(f"Out of vocabulary: {x!r}")
        return StateLM(
            lm=self.lm,
            logp=self.logp + self.logp_next[x],
            context=(self.context, x),
            parent=self,
        )

    @cached_property
    def logp_next(self):
        with torch.no_grad():
            return LazyProb(
                torch.nn.functional.log_softmax(self.out.logits, dim=-1).squeeze().detach().numpy(),
                self.lm._encode, self.lm._decode,
            )

    @cached_property
    def out(self):
        self.lm._calls += 1
        lm = self.lm
        with torch.no_grad():
            if self.context == ():
                input_ids = torch.LongTensor([[lm.tokenizer.bos_token_id]], device=lm.device)
                return lm.model(input_ids=input_ids, past_key_values=None, use_cache=True)
            else:
                (_, x) = self.context
                i = lm._encode[x]
                input_ids = torch.LongTensor([[i]], device=lm.device)
                past_kv = self.parent.out.past_key_values
                if isinstance(past_kv, DynamicCache):
                    raise TypeError(
                        "StateLM does not support DynamicCache (mutates in-place, "
                        "breaks tree-structured branching). See: "
                        "https://github.com/timvieira/transduction/issues/1"
                    )
                return self.lm.model(
                    input_ids=input_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )

    @cached_property
    def p_next(self):
        return self.logp_next.apply(np.exp)

    def token_ids(self):
        return [self.lm._encode[x] for x in flatten(self.context)]

    def __repr__(self):
        return f'{self.__class__.__name__}({[x for x in flatten(self.context)]})'

    @classmethod
    def initial(cls, lm):
        if isinstance(lm, str):
            lm = load_model_by_name(lm, byte_level=True)
        return cls(
            lm=lm,
            logp=0,
            context=(),
            parent=None,
        )
