from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt

from transduction.lm.base import LM, LMState
import torch
from transduction.lm.huggingface_lm import TokenLogProbs, flatten


@dataclass
class _StrippedState:
    """Lightweight KV cache snapshot with scores stripped.

    Full ``LlamaState.scores`` would be ``(n_tokens, n_vocab) * 4`` bytes per
    state snapshot — stripping it reduces memory by orders of magnitude.  The
    ``last_logits`` row (~128 KB for a 32k vocab) is kept so that the caller
    can compute ``logp_next`` without a redundant ``eval()`` call.
    """
    last_logits: npt.NDArray[np.single]   # shape (n_vocab,)
    llama_state: bytes                     # KV cache bytes
    llama_state_size: int
    input_ids: npt.NDArray[np.intc]
    n_tokens: int
    seed: int


class LlamaCppLM(LM[int]):
    """LM[int] wrapping a llama-cpp-python ``Llama`` instance.

    Provides the same ``LM[int]`` / ``LMState[int]`` interface as
    ``HuggingFaceLM``, using llama-cpp-python for CPU-based inference
    with GGUF models.

    The model must be created with ``logits_all=True`` so that
    ``save_state`` / ``load_state`` preserve logits properly.
    """

    def __init__(self, model: Any) -> None:
        import llama_cpp as _llama_cpp
        self._llama_cpp = _llama_cpp

        self._model = model
        if not model._logits_all:
            raise ValueError(
                "LlamaCppLM requires logits_all=True on the Llama instance"
            )

        n_vocab = model.n_vocab()
        self._n_vocab = n_vocab

        # Build byte-level encode / decode vocab tables.
        self._decode: list[bytes | None] = [None] * n_vocab
        self._encode: dict[bytes, int] = {}
        for tid in range(n_vocab):
            b = model.detokenize([tid])
            if b:
                self._decode[tid] = b
                self._encode[b] = tid

        self.eos: int = model.token_eos()
        self._calls: int = 0

        # Cache the initial BOS state (one-time cost).
        self._initial_stripped = self._make_initial_state()

    def _make_initial_state(self) -> _StrippedState:
        """Eval BOS token and return the stripped initial state."""
        self._model.reset()
        bos = self._model.token_bos()
        self._model.eval([bos])
        return self._save_stripped()

    def _save_stripped(self) -> _StrippedState:
        """Save current model state with scores stripped."""
        n_tokens = self._model.n_tokens
        last_logits = self._model.scores[n_tokens - 1, :].copy()
        # Save KV cache via llama.cpp C API
        state_size = self._llama_cpp.llama_get_state_size(self._model._ctx.ctx)
        buf = (ctypes.c_uint8 * int(state_size))()
        n_bytes = self._llama_cpp.llama_copy_state_data(self._model._ctx.ctx, buf)
        if int(n_bytes) > int(state_size):
            raise RuntimeError("Failed to copy llama state data")
        compact = (ctypes.c_uint8 * int(n_bytes))()
        ctypes.memmove(compact, buf, int(n_bytes))
        return _StrippedState(
            last_logits=last_logits,
            llama_state=bytes(compact),
            llama_state_size=int(n_bytes),
            input_ids=self._model.input_ids[:n_tokens].copy(),
            n_tokens=n_tokens,
            seed=self._model._seed,
        )

    def _load_stripped(self, stripped: _StrippedState) -> None:
        """Restore a stripped state into the model.

        Skips the scores copy entirely (saves ``n_tokens * n_vocab * 4``
        bytes of allocation + memcpy per load).  The scores will be
        overwritten by the next ``eval()`` call.
        """
        self._model.input_ids[:stripped.n_tokens] = stripped.input_ids[:stripped.n_tokens]
        self._model.n_tokens = stripped.n_tokens
        self._model._seed = stripped.seed
        buf = (ctypes.c_uint8 * stripped.llama_state_size).from_buffer_copy(
            stripped.llama_state
        )
        n_set = self._llama_cpp.llama_set_state_data(self._model._ctx.ctx, buf)
        if n_set != stripped.llama_state_size:
            raise RuntimeError("Failed to set llama state data")

    def initial(self) -> LlamaCppState:
        return LlamaCppState(
            lm=self,
            logp=0,
            context=(),
            parent=None,
        )

    def prefetch(self, states: list) -> None:
        # Single shared KV cache, no batching — no-op.
        pass

    @classmethod
    def from_file(cls, path: str, n_ctx: int = 2048, **kwargs: Any) -> LlamaCppLM:
        """Load a GGUF model from ``path``."""
        from llama_cpp import Llama
        model = Llama(model_path=path, n_ctx=n_ctx, logits_all=True, verbose=False, **kwargs)
        return cls(model)


class LlamaCppState(LMState[int]):
    """Immutable decode position for a ``LlamaCppLM``.

    Uses the same cons-cell context tuple as ``TokenIDState``:
    ``(parent_context, token_id)``.  All computation is lazy — the
    forward pass only runs when ``logp_next`` is first accessed.
    """

    def __init__(self, lm: LlamaCppLM, logp: float,
                 context: tuple[Any, ...], parent: LlamaCppState | None) -> None:
        self.lm = lm
        self.eos = lm.eos
        self.logp = logp
        self.context = context
        self.parent = parent

    def __rshift__(self, token_id: int) -> LlamaCppState:
        if token_id not in self.logp_next or token_id == self.eos:
            raise ValueError(f"Out of vocabulary: {token_id!r}")
        return LlamaCppState(
            lm=self.lm,
            logp=self.logp + self.logp_next[token_id],
            context=(self.context, token_id),
            parent=self,
        )

    @cached_property
    def _forward_result(self) -> _StrippedState:
        """Load parent KV state, eval one token, return stripped state."""
        lm = self.lm
        if self.context == ():
            # Initial state — return the cached BOS result.
            return lm._initial_stripped
        lm._calls += 1
        (_, token_id) = self.context
        assert self.parent is not None
        # Load parent's KV cache into the model.
        parent_stripped = self.parent._forward_result
        lm._load_stripped(parent_stripped)
        # Eval the new token.
        lm._model.eval([token_id])
        return lm._save_stripped()

    @cached_property
    def logp_next(self) -> TokenLogProbs:  # type: ignore[override]
        logits = self._forward_result.last_logits.astype(np.float64)
        logits -= logits.max()
        log_probs = logits - np.log(np.exp(logits).sum())
        return TokenLogProbs(torch.from_numpy(log_probs.astype(np.float32)), self.lm._decode)

    def token_ids(self) -> list[int]:
        return list(flatten(self.context))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.token_ids()})'
