import gc
import torch
import warnings
from collections import OrderedDict

from tokenization.util import unflatten
from tokenization.lm import LM, LazyProb
from tokenization.backend.vocab_hack import decode_tokenizer_vocab

try:
    from vllm.utils import Counter
    from vllm.inputs import TokensPrompt
    from vllm import SamplingParams, LLMEngine, EngineArgs
    from vllm.model_executor.layers.sampler import SamplerOutput#, Sampler
    from vllm.sequence import SequenceOutput, CompletionSequenceGroupOutput, Logprob
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    VLLM_AVAILABLE = True
except ImportError:
    warnings.warn("vllm is not installed. Some functionality will not be available.")
    VLLM_AVAILABLE = False

if VLLM_AVAILABLE:
    def load_model_by_name(model_name, engine_opts=None, **kwargs):
        """ Load an LLM into a `VirtualLLM`. """
        if engine_opts is None:
            engine_opts = {}
        return VirtualLLM.from_name(
            model_name, engine_opts=engine_opts, **kwargs
        )

    class VirtualLLM(LM):
        def __init__(self, llm_engine, cache_size=None):
            self.llm_engine = llm_engine
            # Replace default sampler with custom implementation optimized for
            # next token logprob calculations.
            self.llm_engine.model_executor.driver_worker.model_runner.model.sampler = (
                DeferredSampler()
            )
            # VLLM processes the outputs of a decoding step asynchronously
            # during the forward pass of the next decoding step. This leads to an extra
            # forward pass at each call to `batch_next_token_logprobs`, which essentially
            # doubles the amount of time we spend on the GPU. The following turns off that
            # functionality.
            self.llm_engine.scheduler[0]._allow_async_output_proc = lambda x: False
            self.llm_engine.log_stats = False
            self.request_counter = Counter()

            self.tokenizer = self.llm_engine.get_tokenizer()
            self._decode = decode_tokenizer_vocab(self.tokenizer)
            self._encode = {x: i for i, x in enumerate(self._decode)}
            self.max_context_len = self.llm_engine.model_config.max_model_len
            self.warn_truncation = True

            self._cache = OrderedDict() if cache_size is not None else None
            self._cache_size = cache_size

            super().__init__(V=set(self._decode), eos=self.tokenizer.eos_token)

        @classmethod
        def from_name(cls, model_name, max_context_len=None, engine_opts=None, **kwargs):
            if engine_opts is None:
                engine_opts = {}
            if max_context_len is not None:
                engine_opts['max_model_len'] = max_context_len

            return cls(
                LLMEngine.from_engine_args(
                    EngineArgs(
                        model=model_name,
                        tokenizer=model_name,
                        enable_prefix_caching=True,
                        **engine_opts
                    )
                ),
                **kwargs,
            )

        def encode_prompt(self, prompt):
            "Encode `prompt` as a tuple of tokens (each a string)."
            enc = self.tokenizer.encode(prompt)
            # bos will be added in encode_context
            if enc[0] == self.tokenizer.bos_token_id:
                enc = enc[1:]
            return unflatten(tuple(self._decode[i] for i in enc))

        def clear_cache(self):
            if self._cache is not None:
                self._cache.clear()

        def get_states(self, contexts, stats=None):
            """Batched version of get_state that processes multiple contexts at once"""
            values = [None] * len(contexts)

            if self._cache is None:
                cache_misses = list(enumerate(contexts))
            else:
                cache_misses = []
                for i, context in enumerate(contexts):
                    assert (
                        isinstance(context, tuple) and
                        (len(context) == 0 or len(context) == 2)
                    )
                    value = self._cache.get(context, None)
                    if value is not None:
                        self._cache.move_to_end(context)
                        values[i] = value
                    else:
                        cache_misses.append((i, context))

            if cache_misses:
                miss_contexts = [c for _, c in cache_misses]
                miss_values = self.batch_next_token_logprobs(
                    miss_contexts, stats=stats
                )

                for (i, context), value in zip(cache_misses, miss_values):
                    values[i] = value
                    if self._cache is not None:
                        self._cache[context] = value
                        if len(self._cache) > self._cache_size:
                            self._cache.popitem(last=False)

            assert len(values) == len(contexts)

            return torch.stack(values)

        def encode_context(self, context):
            depth = 0
            current = context
            while current != ():
                depth += 1
                current = current[0]

            usable_depth = min(depth, self.max_context_len - 1)

            if usable_depth < depth and self.warn_truncation:
                warnings.warn(
                    f'Context of length {depth} exceeds model max context length {self.max_context_len}. '
                    f'Truncating to length {usable_depth}. (Subsequent warnings will be surpressed.)'
                )
                self.warn_truncation = False

            token_ids = [None] * (usable_depth + 1)
            token_ids[0] = self.tokenizer.bos_token_id

            current = context
            i = usable_depth
            while i >= 1:
                token_ids[i] = self._encode[current[1]]
                current = current[0]
                i -= 1

            return token_ids

        def add_request(self, context):
            token_ids = self.encode_context(context)
            request_id = str(next(self.request_counter))
            self.llm_engine.add_request(
                request_id=request_id,
                prompt=TokensPrompt(prompt_token_ids=token_ids),
                params=DEFAULT_SAMPLING_PARAMS,
                lora_request=None,
                prompt_adapter_request=None,
            )
            return request_id, len(token_ids) # for logging

        def batch_next_token_logprobs(self, contexts, stats=None):
            if self.llm_engine.has_unfinished_requests():
                self.abort_all_requests()

            n_tokens = 0
            request_ids = []
            for context in contexts:
                request_id, n = self.add_request(context)
                request_ids.append(request_id)
                n_tokens += 0

            outputs = []
            n_forward_passes = 0
            while self.llm_engine.has_unfinished_requests():
                n_forward_passes += 1 # proxy
                step_outputs = self.llm_engine.step()
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)

            if stats is not None:
                stats.add_forward_passes(n_forward_passes)
                stats.add_tokens_processed_by_lm(n_tokens)

            req_id2logprobs = {
                output.request_id : self._validate_output(output)
                for output in outputs
            }

            return [req_id2logprobs[req_id] for req_id in request_ids]

        def _validate_output(self, output):
            assert len(output.outputs) == 1, f"Expected 1 output but got {len(output.outputs)}"
            assert len(output.outputs[0].logprobs) == 1, f"Expected 1 logprob but got {len(output.outputs[0].logprobs)}"
            return output.outputs[0].logprobs[0].logprobs

        def p_next(self, context, return_logp=False, keep_on_gpu=False):
            assert isinstance(context, tuple) and len(context) == 0 or len(context) == 2, context

            _p = self.get_states([context])[0]

            if not return_logp:
                _p = torch.exp(_p)

            if keep_on_gpu:
                return _p
            else:
                return LazyProb(_p.cpu().numpy(), self._encode, self._decode)

        def logp_next(self, context, keep_on_gpu=False):
            return self.p_next(context, return_logp=True, keep_on_gpu=keep_on_gpu)

        def batch_p_next(self, contexts, return_logp=False, keep_on_gpu=False, stats=None):
            assert all(isinstance(c, tuple) and (len(c) == 0 or len(c) == 2) for c in contexts)

            _ps = self.get_states(contexts, stats=stats)

            if not return_logp:
                _ps = torch.exp(_ps)

            if keep_on_gpu:
                return _ps
            else:
                return [LazyProb(p, self._encode, self._decode) for p in _ps.cpu().numpy()]

        def batch_logp_next(self, contexts, keep_on_gpu=False, stats=None):
            return self.batch_p_next(
                contexts, return_logp=True, keep_on_gpu=keep_on_gpu, stats=stats
            )

        def abort_all_requests(self):
            for request_id in range(self.request_counter.counter):
                self.llm_engine.abort_request(str(request_id))

        def __del__(self):
            destroy_model_parallel()
            destroy_distributed_environment()

            del self.llm_engine.model_executor
            gc.collect()
            torch.cuda.empty_cache()

    DEFAULT_SAMPLING_PARAMS = SamplingParams(
        max_tokens=1,     # Only run a single step of decoding.
        n=1,              # Only create a single sequence for each prompt (sequence group).
        logprobs=1,       # logprobs must be > 1 to return logprob vector using the custom sampler.
        detokenize=False, # Detokenization causes significant overhead, avoid it.
        stop=None,
        ignore_eos=True,
    )

    class LazyLogprobDict:
        # vLLM stores logprobs in a dictionary which maps token ids to `Logprob` objects.
        # Constructing the dictionary is extremely slow for a large number of logprobs.
        # This class provides a dictionary-like interface that maps token IDs to Logprob objects,
        # but internally stores the raw logprob values as an np.array (indexed by token id) and
        # lazily creates Logprob objects.
        def __init__(self, logprobs):
            self.logprobs = logprobs

        def __getitem__(self, key):
            if 0 <= key < len(self.logprobs):
                return Logprob(self.logprobs[key])
            raise KeyError(key)

        def __contains__(self, key):
            return 0 <= key < len(self.logprobs)

        def __len__(self):
            return len(self.logprobs)

        def items(self):
            return ((i, Logprob(prob)) for i, prob in enumerate(self.logprobs))

        def keys(self):
            return range(len(self.logprobs))

        def values(self):
            return iter(map(Logprob, self.logprobs))

        def get(self, key, default=None):
            try:
                return self[key]
            except KeyError:
                return default

    class DeferredSampler(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, logits, sampling_metadata):
            """
            Args:
                logits: (num_tokens, vocab_size).
                sampling_metadata: Metadata for sampling.
            """
            assert logits is not None

            logprobs = logits.log_softmax(dim=-1, dtype=torch.float)#.cpu().numpy()

            sample_idx = 0
            sampler_output = []
            for seq_group in sampling_metadata.seq_groups:
                seq_ids = seq_group.seq_ids
                num_parent_seqs = len(seq_ids)
                logprobs_by_seq = logprobs[sample_idx : sample_idx + num_parent_seqs]

                assert len(logprobs_by_seq) == len(seq_ids)

                seq_outputs = []
                for (seq_id, seq_logprobs) in zip(seq_ids, logprobs_by_seq):
                    seq_outputs.append(
                        SequenceOutput(
                            # We do not actually sample a token and construct a lazy logprob dict
                            # instead of instantiating the full dictionnary of logprobs.
                            # We pass in a token_id of 0 as the sampled next token, but this
                            # will have no effect on the vllm scheduler since the sequence will
                            # immediately be removed from the scheduler.
                            seq_id, 0, LazyLogprobDict(seq_logprobs)
                        )
                    )

                sampler_output.append(
                    CompletionSequenceGroupOutput(samples=seq_outputs, prompt_logprobs=[])
                )

                sample_idx += 1

            sampler_outputs = SamplerOutput(
                outputs=sampler_output,
                sampled_token_probs=None,
                sampled_token_ids=None,
                logprobs=None,
                deferred_sample_results_args=None
            )

            return sampler_outputs
