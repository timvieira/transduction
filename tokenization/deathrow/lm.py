"""
Language models go here
"""

import numpy as np
import torch
import transformers
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from collections import OrderedDict
from arsenal.maths import sample_dict

from tokenization.vocab.decode import decode_hf_tokenizer
from tokenization.util import  flatten, unflatten, Chart, LazyProb


def concat(xs, ys):
    return unflatten(flatten(xs) + flatten(ys))


class LM:
    r"""We say that $p\colon V^* \to [0,1]$ is a language model if $p$ is a probability
    distribution over strings from some alphabet $V$ of tokens.

    Every language model admits a left-to-right factorization:

    $$
    p(x_1 x_2 \cdots x_T) = p(x_1 \mid \varepsilon) p(x_2 \mid x_1) \cdots p(x_T \mid x_1 \cdots x_{T-1}) p(\mathrm{EOS} \mid x_1 \cdots x_T)
    $$

    Arguments:

      - `V`: a vocabulary of symbols

      - `eos`: a distinguished end of sequence symbol

      - `p_next(xs)`: $p(\cdot \mid x_1 \cdots x_T)$ is provided by subclasses.

    """

    def __init__(self, V, eos):
        self.eos = eos
        self.V = V
        self.concat = concat
        self.empty = ()

    def batch_p_next(self, contexts, keep_on_gpu=True):
        return torch.Tensor([self.p_next(context)._p for context in contexts])

    def logp_next(self, context):
        "Compute the log conditional distribution over the next token given the `prefix`."
        raise NotImplementedError()

    def logprefix(self, context):
        assert isinstance(context, tuple) and len(context) == 0 or len(context) == 2, context
        if len(context) == 0:
            return 0.0
        else:
            context, y = context
            return self.logprefix(context) + self.logp_next(context)[y]

    def logp(self, context):
        "Compute the log-probability of a complete string."
        return self.logprefix(context) + self.logp_next(context)[self.eos]

    def logp_next_seq(self, context, extension):
        """
        Compute `p(extension | context)` where `extension` is a sequence with |extension| > 1.
        """
        return self.logprefix(self.concat(context, extension)) - self.logprefix(context)

    def clear_cache(self):  # pragma: no cover
        pass

    def sample(
        self,
        ys=(),
        draw=sample_dict,
        prob=True,
        verbose=0,
        max_tokens=np.inf,
    ):
        "Draw a sample from this distribution."
        assert isinstance(ys, tuple) and len(ys) in {0, 2}, ys
        logP = 0
        t = 0
        while True:
            logp = self.logp_next(ys)
            p = logp.apply(np.exp)
            y = draw(p) if t < max_tokens else self.eos
            logP += logp[y]
            t += 1
            if verbose:
                if y == self.eos:
                    print()
                else:
                    print(y, end='')
            if y == self.eos:
                return [ys, logP] if prob else ys
            ys = (ys, y)

    def greedy(self, ys, **kwargs):
        return self.sample(ys=ys, draw=lambda p: p.materialize(top=1).argmax(), **kwargs)


class TokenizedLLM(LM):
    """
    This is a simple class which wraps a token LLM with a tokenizer.
    """

    def __init__(self, tokenizer, model, cache_size=128, byte_level=False):
        self.tokenizer = tokenizer

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)    # send the model to gpu, if one is available
        model.eval()             # Set the model in evaluation mode; avoids gradient overhead

        self.byte_level = byte_level
        if byte_level:
            (_, self._encode, self._decode, _) = decode_hf_tokenizer(tokenizer)
            eos = self.tokenizer.eos_token.encode()
        else:
            from tokenization.vocab import hacky
            self._decode = hacky.decode_tokenizer_vocab(self.tokenizer)
            self._encode = {x: i for i, x in enumerate(self._decode)}
            eos = self.tokenizer.eos_token

        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._eos_token_id = self._encode[eos]
        self._calls = 0

        super().__init__(V=set(self._decode), eos=eos)

    def encode_prompt(self, prompt):
        "Encode `prompt` as a tuple of tokens (each a string)."
        return unflatten(tuple(self._decode[i] for i in self.tokenizer.encode(prompt)))

#    def __call__(self, context):
#        return self._model([self._encode[x] for x in context])

#    def __call__(self, context):
#        return np.exp(self.logp(context))

#    def logp(self, context):
#        input_ids = context
#        if isinstance(input_ids, list):
#            input_ids = torch.LongTensor([input_ids]).squeeze()
#        if input_ids[0] != self.model.config.bos_token_id:
#            input_ids = torch.cat(
#                [torch.LongTensor([self.model.config.bos_token_id]), input_ids]
#            )
#        with torch.no_grad():
#            input_ids = input_ids.to(self.device)
#            outputs = self.model(input_ids=input_ids, labels=input_ids)
#            lprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
#        token_lprobs = torch.gather(lprobs, 1, input_ids[1:].unsqueeze(-1)).squeeze(-1)
#        return torch.sum(token_lprobs, dim=-1).item()

    def clear_cache(self):
        self._cache.clear()

    def get_state(self, context):
        assert isinstance(context, tuple) and (len(context) == 0 or len(context) == 2), context

        value = self._cache.get(context, None)
        if value is not None:
            self._cache.move_to_end(context)   # Move the key to the end to show it was recently used
            return value

        self._calls += 1

        if len(context) == 0:
            # Note: initial state uses BOS padding.
            input_ids = torch.LongTensor([self.tokenizer.bos_token_id], device=self.device)
            value = self.model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
            )

        else:
            (xs, x) = context
            x = self._encode[x]
            prev_state = self.get_state(xs)
            input_ids = torch.LongTensor([x], device=self.device)
            value = self.model(
                input_ids=input_ids,
                past_key_values=prev_state.past_key_values,
                use_cache=True,
            )

        self._cache[context] = value
        if len(self._cache) > self._cache_size:
            # Pop the first item, as it is least recently used
            self._cache.popitem(last=False)

        return value

    def logp_next(self, context):
        with torch.no_grad():
            outputs = self.get_state(context)
            lprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        _logp = lprobs[0, :]
        if hasattr(_logp, 'cpu'):
            _logp = _logp.cpu().numpy()
        return LazyProb(_logp, self._encode, self._decode)

    # TODO: needs test cases!
    def _compute_logp_batch(self, sequences, batch_size=64):
        pad_token_id = self._encode[self.eos]

        # Function to compute log probabilities for a batch
        def compute_batch_log_probabilities(batch_token_ids):
            # Pad the sequences to the same length
            input_ids = pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in batch_token_ids],
                batch_first=True,
                padding_value=pad_token_id
            ).to(self.device)

            # Create attention mask
            attention_mask = (input_ids != pad_token_id).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                # Compute log-likelihoods for each sequence
                losses = outputs.loss * attention_mask.sum(dim=1)  # Scale loss by sequence lengths
                log_likelihoods = (-losses.cpu()).tolist()
            return log_likelihoods

        # Batch processing
        log_probs = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring sequences in batches"):
            batch = sequences[i:i + batch_size]
            log_probs.extend(compute_batch_log_probabilities(batch))
        return log_probs

    def initial(self):
        from tokenization.statelm import StateLM
        return StateLM.initial(self)

    def __matmul__(self, fst, *args, **kwargs):
        return self.transduce(fst, *args, **kwargs)

    def transduce(self, fst, *args, **kwargs):
        from tokenization.transduction import TransducedLM
        return TransducedLM(self, fst, *args, **kwargs)



def load_model_by_name(model_name, **kwargs):
    """
    Load an LLM from 🤗 into a `TokenizedLLM`.
    """
    return TokenizedLLM(
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False),
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name),
        **kwargs,
    )


class PCFG:
    "Probabilistic context-free grammar"

    def __init__(self, cfg):
        self.cfg = cfg
        self.V = cfg.V
        self._decode = list(self.V)
        self._encode = dict(zip(self._decode, range(len(self._decode))))
        self.eos = cfg.eos

    def logp_next(self, context):
        p = self.cfg.p_next(flatten(context))
        logp = np.full(len(self.V), -np.inf)
        for i, x in enumerate(self._decode):
            if p[x] > 0:
                logp[i] = np.log(p[x])
        return LazyProb(logp, self._encode, self._decode)

    @classmethod
    def from_string(cls, grammar):
        import genparse
        return cls(genparse.EarleyLM.from_string(grammar))
