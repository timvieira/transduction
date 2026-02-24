from __future__ import annotations

import numpy as np
import torch
import transformers
from transformers.cache_utils import DynamicCache

from functools import cached_property
from typing import Any

from transduction.lm.base import LM, LMState


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


class HfTokenizerVocab:
    """Byte-level encode/decode tables extracted from a HuggingFace tokenizer.

    Assumptions:
      - The tokenizer is a byte-level BPE tokenizer (GPT-2 style).  Each
        token's string representation is a sequence of Unicode characters that
        map 1:1 to bytes via the GPT-2 ``bytes_to_unicode()`` table (the
        ``_encode_bytes_str`` array above).
      - Merge rules are extracted from either ``tokenizer.bpe_ranks`` (older
        GPT2TokenizerFast) or the ``tokenizer._tokenizer`` JSON backend
        (newer PreTrainedTokenizerFast).  The ``_tokenizer`` path accesses a
        private attribute and may break on future transformers versions.
      - The ``byte_decoder`` attribute (mapping Unicode chars → byte values)
        is read from ``tokenizer.byte_decoder`` if present, otherwise the
        default GPT-2 mapping is used.
      - Every byte 0–255 must appear as a single-byte token in the vocabulary.
        Tokenizers that lack byte-level coverage will raise an error.

    Attributes:
      merges: list of (left_id, right_id, merged_id) triples
      encode: dict mapping bytes → token_id
      decode: list mapping token_id → bytes
      encode_byte: list mapping byte_value (0–255) → single-byte token_id
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

        self.merges: list[tuple[int, int, int]] = []
        V = tokenizer.get_vocab()
        if hasattr(tokenizer, 'bpe_ranks'):
            for (u, v) in tokenizer.bpe_ranks:
                self.merges.append((V[u], V[v], V[u + v]))
        else:
            import json
            subtokenizer_dict = json.loads(tokenizer._tokenizer.to_str())
            for (u, v) in subtokenizer_dict["model"]["merges"]:
                self.merges.append((V[u], V[v], V[u + v]))

        if hasattr(tokenizer, 'byte_decoder'):
            byte_decoder = tokenizer.byte_decoder
        else:
            byte_decoder = _default_byte_decoder

        self.encode: dict[bytes, int] = {}
        self.decode: list[bytes | None] = [None] * len(V)
        for bs, token_id in V.items():
            b = bytes([byte_decoder[b] for b in bs])
            self.encode[b] = token_id
            self.decode[token_id] = b

        self.encode_byte: list[int | None] = [None] * 256
        for i in range(256):
            self.encode_byte[i] = self.encode[bytes([i])]


# ---------------------------------------------------------------------------
# Utility classes (ported from tokenization.util)
# ---------------------------------------------------------------------------

def flatten(xs: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(xs) == 0:
        return ()
    else:
        ys, y = xs
        return flatten(ys) + (y,)


def unflatten(ys: tuple[Any, ...]) -> tuple[Any, ...]:
    xs: tuple[Any, ...] = ()
    for y in ys:
        xs = (xs, y)
    return xs


# ---------------------------------------------------------------------------
# LM classes
# ---------------------------------------------------------------------------

class TokenLogProbs:
    """Log-probability distribution over token IDs (backed by a numpy array).

    Supports ``logp_next[token_id]``, ``token_id in logp_next``,
    ``argmax()``, ``sample()``, ``top(K)``, and ``relabel(keys)``
    to produce a dict with arbitrary key types.
    """

    def __init__(self, _p: np.ndarray, decode: list[bytes | None]) -> None:
        self._p = _p
        self._decode = decode

    def __getitem__(self, token_id: int) -> float:
        return self._p[token_id]

    def __contains__(self, token_id: int) -> bool:
        return 0 <= token_id < len(self._p) and self._decode[token_id] is not None

    def keys(self) -> range:
        return range(len(self._p))

    def values(self) -> np.ndarray:
        return self._p

    def items(self) -> enumerate:
        return enumerate(self._p)

    def argmax(self) -> int:
        return int(self._p.argmax())

    def sample(self) -> int:
        logps = self._p.copy().astype(np.float64)
        logps -= logps.max()
        probs = np.exp(logps)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def top(self, K: int) -> dict[int, Any]:
        top_idx = self._p.argsort()[-K:]
        return {int(i): self._p[i] for i in reversed(top_idx)}

    def relabel(self, keys: list) -> dict:
        """Return a dict mapping ``keys[i] → logp[i]``, skipping None keys.

        Example: ``state.logp_next.relabel(lm._decode)`` returns a
        ``dict[bytes, float]`` keyed on token byte strings.
        """
        top_idx = self._p.argsort()
        return {keys[i]: self._p[i] for i in reversed(top_idx) if keys[i] is not None}

    def __repr__(self) -> str:
        return repr(self.top(10))


# TODO: This class will encounter issues when its token vocabulary has multiple
# token_ids that map to the same string of bytes.
class HuggingFaceLM(LM[int]):
    """LM[int] wrapping a HuggingFace causal LM.

    Owns the model, tokenizer, and byte-level vocabulary tables.
    ``initial()`` returns a ``TokenIDState`` keyed on int token IDs.

    Directly compatible with callers that work with integer source symbols
    (e.g., ``CharacterBeam``, ``GeneralizedBeam``, ``FusedTransducedLM``).
    """

    def __init__(self, tokenizer: Any, model: Any, byte_level: bool = True) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

        assert byte_level, "Only byte_level=True is supported"
        vocab = HfTokenizerVocab(tokenizer)
        self._encode = vocab.encode
        self._decode = vocab.decode

        self._calls: int = 0
        self.V: set[bytes | None] = set(self._decode)
        self.eos: int = self._encode[self.tokenizer.eos_token.encode()]

    def initial(self) -> TokenIDState:
        return TokenIDState(
            lm=self,
            logp=0,
            context=(),
            parent=None,
        )

    def encode_prompt(self, prompt: str) -> tuple[Any, ...]:
        "Encode ``prompt`` as a tuple of tokens (each a bytes object)."
        return unflatten(tuple(self._decode[i] for i in self.tokenizer.encode(prompt)))

    @classmethod
    def from_name(cls, model_name: str, **kw: Any) -> HuggingFaceLM:
        return load_model_by_name(model_name, **kw)


def load_model_by_name(model_name: str, device: torch.device | None = None,
                       **kwargs: Any) -> HuggingFaceLM:
    """Load a HuggingFace causal LM into a ``HuggingFaceLM``."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return HuggingFaceLM(
        tokenizer=transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False),
        model=(
            transformers.AutoModelForCausalLM.from_pretrained(model_name)
            .to(device)
            .eval()
        ),
        **kwargs,
    )


class TokenIDState(LMState[int]):
    """Immutable LM state keyed on integer token IDs.

    Stores int token IDs in ``context``; the model forward pass uses
    them directly with no encode/decode round-trip.

    KV Cache Safety
    ---------------
    Modern transformers (>=4.40) returns ``DynamicCache`` objects that
    ``model.forward()`` **mutates in-place** via ``cache.update()``.  If two
    children share a parent's cache directly, the first child's forward pass
    corrupts it for the second.  We detect ``DynamicCache`` and raise rather
    than silently produce wrong results.
    """

    def __init__(self, lm: HuggingFaceLM, logp: float,
                 context: tuple[Any, ...], parent: TokenIDState | None) -> None:
        self.lm = lm
        self.eos = lm.eos
        self.logp = logp
        self.context = context
        self.parent = parent

    def __rshift__(self, token_id: int) -> TokenIDState:
        if token_id not in self.logp_next or token_id == self.eos:
            raise ValueError(f"Out of vocabulary: {token_id!r}")
        return TokenIDState(
            lm=self.lm,
            logp=self.logp + self.logp_next[token_id],
            context=(self.context, token_id),
            parent=self,
        )

    @cached_property
    def logp_next(self) -> TokenLogProbs:  # type: ignore[override]
        with torch.no_grad():
            return TokenLogProbs(
                torch.nn.functional.log_softmax(self.out.logits, dim=-1).squeeze().detach().numpy(),
                self.lm._decode,
            )

    @cached_property
    def out(self) -> Any:
        self.lm._calls += 1
        with torch.no_grad():
            if self.context == ():
                input_ids = torch.LongTensor([[self.lm.tokenizer.bos_token_id]], device=self.lm.device)
                return self.lm.model(input_ids=input_ids, past_key_values=None, use_cache=True)
            else:
                (_, token_id) = self.context
                input_ids = torch.LongTensor([[token_id]], device=self.lm.device)
                past_kv = self.parent.out.past_key_values  # type: ignore[union-attr]
                if isinstance(past_kv, DynamicCache):
                    raise TypeError(
                        "TokenIDState does not support DynamicCache (mutates in-place, "
                        "breaks tree-structured branching). See: "
                        "https://github.com/timvieira/transduction/issues/1"
                    )
                return self.lm.model(
                    input_ids=input_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )

    def token_ids(self) -> list[int]:
        return list(flatten(self.context))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.token_ids()})'
