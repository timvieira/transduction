from __future__ import annotations

import torch
import transformers
from transformers.cache_utils import DynamicCache, DynamicLayer

from functools import cached_property
from typing import Any

from transduction.lm.base import LM, LMState
from transduction.util import LogDistr


def _clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    """Deep-clone a DynamicCache so tree-structured branching works correctly."""
    new_cache = DynamicCache()
    for layer in cache.layers:
        new_layer = DynamicLayer()
        new_layer.keys = layer.keys.clone()
        new_layer.values = layer.values.clone()
        new_layer.dtype = layer.dtype
        new_layer.device = layer.device
        new_layer.is_initialized = True
        new_cache.layers.append(new_layer)
    return new_cache


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
    """Log-probability distribution over token IDs (backed by a torch tensor).

    Supports ``logp_next[token_id]``, ``token_id in logp_next``,
    ``argmax()``, ``sample()``, ``top(K)``, and ``relabel(keys)``
    to produce a dict with arbitrary key types.
    """

    def __init__(self, _p: torch.Tensor, decode: list[bytes | None]) -> None:
        self._p = _p
        self._decode = decode

    def __getitem__(self, token_id: int | torch.Tensor) -> float:
        return float(self._p[token_id])

    def __contains__(self, token_id: int) -> bool:
        return 0 <= token_id < len(self._p) and self._decode[token_id] is not None

    def keys(self) -> range:
        return range(len(self._p))

    def values(self) -> torch.Tensor:
        return self._p

    def items(self) -> enumerate:
        return enumerate(self._p)

    def argmax(self) -> int:
        return int(self._p.argmax())

    def sample(self) -> int:
        logps = self._p.clone().double()
        logps -= logps.max()
        probs = torch.exp(logps)
        probs /= probs.sum()
        return int(torch.multinomial(probs, 1).item())

    def top(self, K: int) -> LogDistr[bytes]:
        top_idx = self._p.argsort()[-K:]
        return LogDistr({
            self._decode[i]: float(self._p[i])
            for i in reversed(top_idx.tolist())
            if self._decode[i] is not None
        })

    def top_ids(self, K: int) -> dict[int, Any]:
        top_idx = self._p.argsort()[-K:]
        return {int(i): float(self._p[i]) for i in reversed(top_idx.tolist())}

    def relabel(self, keys: list) -> dict:
        """Return a dict mapping ``keys[i] → logp[i]``, skipping None keys.

        Example: ``state.logp_next.relabel(lm._decode)`` returns a
        ``dict[bytes, float]`` keyed on token byte strings.
        """
        top_idx = self._p.argsort()
        return {keys[i]: float(self._p[i]) for i in reversed(top_idx.tolist()) if keys[i] is not None}

    def __repr__(self) -> str:
        return repr(self.top_ids(10))


class _BatchedOutput:
    """Lightweight container matching the interface of model forward output.

    Holds ``.logits`` and ``.past_key_values`` for a single sequence,
    allowing ``TokenIDState.logp_next`` to consume it the same way it
    consumes a real model output.
    """
    __slots__ = ('logits', 'past_key_values')

    def __init__(self, logits: torch.Tensor, past_key_values: Any) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class _BatchWorkspace:
    """Pre-allocated arena for cross-parent batched forward passes.

    Lazily allocates on first use (needs model config for shapes).
    Grows if sequence length exceeds current buffer size.
    Between calls, buffers are overwritten in-place.
    """
    __slots__ = ('_capacity', '_device', '_buf_seq', '_num_layers', '_dtype',
                 '_keys_buf', '_vals_buf', '_attn_mask', '_pos_ids', '_input_ids')

    def __init__(self, capacity: int, device: torch.device) -> None:
        self._capacity = capacity
        self._device = device
        self._buf_seq = 0
        self._num_layers = 0
        self._dtype: torch.dtype | None = None
        self._keys_buf: list[torch.Tensor] = []
        self._vals_buf: list[torch.Tensor] = []
        self._attn_mask: torch.Tensor | None = None
        self._pos_ids: torch.Tensor | None = None
        self._input_ids: torch.Tensor | None = None

    def ensure(self, num_layers: int, num_heads: int, head_dim: int,
               max_seq: int, dtype: torch.dtype) -> None:
        """Ensure buffers can accommodate the given dimensions. Lazy-init or grow."""
        if (self._attn_mask is not None
                and self._num_layers == num_layers
                and self._buf_seq >= max_seq
                and self._dtype == dtype):
            return
        self._num_layers = num_layers
        self._buf_seq = max(max_seq, self._buf_seq)
        self._dtype = dtype
        self._keys_buf = [
            torch.zeros(self._capacity, num_heads, self._buf_seq, head_dim,
                        dtype=dtype, device=self._device)
            for _ in range(num_layers)
        ]
        self._vals_buf = [
            torch.zeros(self._capacity, num_heads, self._buf_seq, head_dim,
                        dtype=dtype, device=self._device)
            for _ in range(num_layers)
        ]
        self._attn_mask = torch.zeros(self._capacity, self._buf_seq + 1,
                                       device=self._device)
        self._pos_ids = torch.zeros(self._capacity, 1, dtype=torch.long,
                                     device=self._device)
        self._input_ids = torch.zeros(self._capacity, 1, dtype=torch.long,
                                       device=self._device)


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
        self._workspace = _BatchWorkspace(self._MAX_BATCH, self.device)

    def initial(self) -> TokenIDState:
        return TokenIDState(
            lm=self,
            logprefix=0,
            context=(),
            _kv_source=None,
        )

    def encode_prompt(self, prompt: str) -> tuple[Any, ...]:
        "Encode ``prompt`` as a tuple of tokens (each a bytes object)."
        return unflatten(tuple(self._decode[i] for i in self.tokenizer.encode(prompt)))

    # Maximum siblings to batch in a single forward pass.  Keeps peak
    # memory bounded when an FST node has very high fan-out (e.g. full
    # vocab transitions from one hub).
    _MAX_BATCH: int = 64

    def prefetch(self, states: list) -> None:
        """Pre-compute logp_next for multiple TokenIDStates in batched forward passes.

        Groups pending states (those without a cached ``out``) by parent.
        Same-parent groups with >1 sibling use ``_batch_siblings`` (no padding).
        Remaining singletons from different parents are batched together via
        ``_batch_cross_parents`` (left-padded KV + attention mask).
        Large groups are chunked to stay within ``_MAX_BATCH``.
        """
        # Filter to pending TokenIDStates that haven't computed ``out`` yet.
        pending = [s for s in states
                    if isinstance(s, TokenIDState) and 'out' not in s.__dict__
                    and s.context != () and s._kv_source is not None]

        # Group by parent identity (siblings share a parent KV cache).
        groups: dict[int, list[TokenIDState]] = {}
        for s in pending:
            key = id(s._kv_source)
            groups.setdefault(key, []).append(s)

        # Same-parent groups with >1 sibling → _batch_siblings.
        # Singletons are collected for cross-parent batching.
        singletons: list[TokenIDState] = []
        for siblings in groups.values():
            if len(siblings) > 1:
                for i in range(0, len(siblings), self._MAX_BATCH):
                    self._batch_siblings(siblings[i:i + self._MAX_BATCH])
            else:
                singletons.append(siblings[0])

        # Cross-parent batch: batch singletons from different parents.
        if len(singletons) >= 2:
            for i in range(0, len(singletons), self._MAX_BATCH):
                self._batch_cross_parents(singletons[i:i + self._MAX_BATCH])

    def _batch_siblings(self, siblings: list[TokenIDState]) -> None:
        """Run a single batched forward pass for sibling TokenIDStates.

        All siblings must share the same parent (same KV cache).
        Injects results into each state's ``__dict__['out']`` so that
        the ``cached_property`` is pre-populated.
        """
        parent = siblings[0]._kv_source
        assert parent is not None

        # Stack token IDs: each sibling consumed one token from the parent.
        token_ids = [s.context[1] for s in siblings]
        input_ids = torch.tensor([[tid] for tid in token_ids], dtype=torch.long, device=self.device)

        # Expand parent's KV cache along batch dimension.
        parent_kv = parent.out.past_key_values
        batch_size = len(siblings)

        batch_kv = DynamicCache()
        for layer in parent_kv.layers:
            new_layer = DynamicLayer()
            new_layer.keys = layer.keys.expand(batch_size, -1, -1, -1).contiguous()
            new_layer.values = layer.values.expand(batch_size, -1, -1, -1).contiguous()
            new_layer.dtype = layer.dtype
            new_layer.device = layer.device
            new_layer.is_initialized = True
            batch_kv.layers.append(new_layer)

        # Single batched forward pass.
        self._calls += 1
        with torch.no_grad():
            result = self.model(
                input_ids=input_ids,
                past_key_values=batch_kv,
                use_cache=True,
            )

        # Slice output: extract per-child logits + KV cache.
        out_kv = result.past_key_values
        for idx, state in enumerate(siblings):
            child_kv = DynamicCache()
            for layer in out_kv.layers:
                new_layer = DynamicLayer()
                new_layer.keys = layer.keys[idx:idx+1].clone()
                new_layer.values = layer.values[idx:idx+1].clone()
                new_layer.dtype = layer.dtype
                new_layer.device = layer.device
                new_layer.is_initialized = True
                child_kv.layers.append(new_layer)
            child_logits = result.logits[idx:idx+1].clone()
            # Inject into cached_property slot.
            state.__dict__['out'] = _BatchedOutput(child_logits, child_kv)
            state._kv_source = None

    def _batch_cross_parents(self, children: list[TokenIDState]) -> None:
        """Batch children from different parents in a single forward pass.

        Uses left-padded KV caches with attention masks and explicit position
        IDs so children with different-length parents can be batched together.
        Uses the pre-allocated ``_workspace`` arena to avoid per-call allocation.
        """
        N = len(children)

        # Step 1: Collect metadata from each child's parent.
        parents: list[TokenIDState] = []
        parent_lens: list[int] = []
        token_ids: list[int] = []
        for child in children:
            parent = child._kv_source
            assert parent is not None
            parent_kv = parent.out.past_key_values
            parent_len = parent_kv.layers[0].keys.shape[2]
            parents.append(parent)
            parent_lens.append(parent_len)
            token_ids.append(child.context[1])

        max_seq = max(parent_lens)

        # Get model config from first parent's KV cache.
        first_kv = parents[0].out.past_key_values
        num_layers = len(first_kv.layers)
        num_heads = first_kv.layers[0].keys.shape[1]
        head_dim = first_kv.layers[0].keys.shape[3]
        kv_dtype = first_kv.layers[0].keys.dtype

        # Step 2: Fill workspace buffers (no allocation — write into arena).
        ws = self._workspace
        ws.ensure(num_layers, num_heads, head_dim, max_seq, kv_dtype)

        # Zero only the region that will be read.
        for layer_idx in range(num_layers):
            ws._keys_buf[layer_idx][:N, :, :max_seq, :].zero_()
            ws._vals_buf[layer_idx][:N, :, :max_seq, :].zero_()
        ws._attn_mask[:N, :max_seq + 1].zero_()

        for i, (child, parent, parent_len) in enumerate(
            zip(children, parents, parent_lens)
        ):
            pad = max_seq - parent_len
            parent_kv = parent.out.past_key_values

            for layer_idx in range(num_layers):
                layer = parent_kv.layers[layer_idx]
                # Left-pad: write parent KV into positions [pad, max_seq).
                ws._keys_buf[layer_idx][i, :, pad:max_seq, :] = layer.keys[0]
                ws._vals_buf[layer_idx][i, :, pad:max_seq, :] = layer.values[0]

            # Attention mask: 1 for real positions + new token, 0 for padding.
            ws._attn_mask[i, pad:max_seq + 1] = 1.0
            # Position of the new token (0-indexed).
            ws._pos_ids[i, 0] = parent_len
            # Input token ID.
            ws._input_ids[i, 0] = token_ids[i]

        # Step 3: Forward pass with workspace buffer views (no copy).
        batch_kv = DynamicCache()
        for layer_idx in range(num_layers):
            new_layer = DynamicLayer()
            new_layer.keys = ws._keys_buf[layer_idx][:N, :, :max_seq, :]
            new_layer.values = ws._vals_buf[layer_idx][:N, :, :max_seq, :]
            new_layer.dtype = first_kv.layers[0].dtype
            new_layer.device = first_kv.layers[0].device
            new_layer.is_initialized = True
            batch_kv.layers.append(new_layer)

        self._calls += 1
        with torch.no_grad():
            result = self.model(
                input_ids=ws._input_ids[:N],
                past_key_values=batch_kv,
                attention_mask=ws._attn_mask[:N, :max_seq + 1],
                position_ids=ws._pos_ids[:N],
                use_cache=True,
            )

        # Step 4: Extract per-child results, stripping left-padding.
        out_kv = result.past_key_values
        for i, (child, parent_len) in enumerate(zip(children, parent_lens)):
            pad = max_seq - parent_len
            child_kv = DynamicCache()
            for layer_idx in range(num_layers):
                layer = out_kv.layers[layer_idx]
                new_layer = DynamicLayer()
                # Strip left-padding: keep [pad:] = parent_len + 1 real positions.
                new_layer.keys = layer.keys[i:i+1, :, pad:, :].clone()
                new_layer.values = layer.values[i:i+1, :, pad:, :].clone()
                new_layer.dtype = layer.dtype
                new_layer.device = layer.device
                new_layer.is_initialized = True
                child_kv.layers.append(new_layer)
            child_logits = result.logits[i:i+1].clone()
            # Inject into cached_property slot.
            child.__dict__['out'] = _BatchedOutput(child_logits, child_kv)
            child._kv_source = None

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

    Memory Management
    -----------------
    ``_kv_source`` holds a reference to the parent state, used solely to
    retrieve its KV cache during the forward pass.  Once ``out`` is computed,
    ``_kv_source`` is set to ``None`` so that ancestor states (and their KV
    caches) can be garbage-collected when no longer reachable.
    """

    def __init__(self, lm: HuggingFaceLM, logprefix: float,
                 context: tuple[Any, ...], _kv_source: TokenIDState | None) -> None:
        self.lm = lm
        self.eos = lm.eos
        self.logprefix = logprefix
        self.context = context
        self._kv_source = _kv_source

    def __rshift__(self, token_id: int) -> TokenIDState:
        if token_id not in self.logp_next or token_id == self.eos:
            raise ValueError(f"Out of vocabulary: {token_id!r}")
        return TokenIDState(
            lm=self.lm,
            logprefix=self.logprefix + self.logp_next[token_id],
            context=(self.context, token_id),
            _kv_source=self,
        )

    @cached_property
    def logp_next(self) -> TokenLogProbs:  # type: ignore[override]
        with torch.no_grad():
            return TokenLogProbs(
                torch.nn.functional.log_softmax(self.out.logits, dim=-1).squeeze().detach().cpu(),
                self.lm._decode,
            )

    @cached_property
    def out(self) -> Any:
        self.lm._calls += 1
        with torch.no_grad():
            if self.context == ():
                input_ids = torch.tensor([[self.lm.tokenizer.bos_token_id]], dtype=torch.long, device=self.lm.device)
                result = self.lm.model(input_ids=input_ids, past_key_values=None, use_cache=True)
            else:
                (_, token_id) = self.context
                input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.lm.device)
                # Clone the parent's KV cache so the model's in-place mutation
                # doesn't corrupt it for sibling children.
                past_kv = self._kv_source.out.past_key_values  # type: ignore[union-attr]
                if isinstance(past_kv, DynamicCache):
                    past_kv = _clone_dynamic_cache(past_kv)
                self._kv_source = None
                result = self.lm.model(
                    input_ids=input_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            return result

    def token_ids(self) -> list[int]:
        return list(flatten(self.context))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.token_ids()})'
