import numpy as np
import torch
import transformers

from arsenal.maths import sample_dict
from functools import cached_property

from tokenization.vocab.decode import decode_hf_tokenizer
from tokenization.util import  unflatten, flatten, LazyProb


class TokenizedLLM:

    def __init__(self, tokenizer, model, byte_level=False):
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

        self.byte_level = byte_level
        if byte_level:
            (_, self._encode, self._decode, _) = decode_hf_tokenizer(tokenizer)
            self.eos = self.tokenizer.eos_token.encode()
        else:
            from tokenization.vocab import hacky
            import warnings
            warnings.warn('`byte_level=False` will be deprecated soon.')
            self._decode = hacky.decode_tokenizer_vocab(self.tokenizer)
            self._encode = {x: i for i, x in enumerate(self._decode)}
            self.eos = self.tokenizer.eos_token

        self._calls = 0
        self.V = set(self._decode)

    @cached_property
    def bpe(self):
        from tokenization.bpe import BPE
        return BPE.from_huggingface(self.tokenizer)

    def encode_prompt(self, prompt):
        "Encode `prompt` as a tuple of tokens (each a string)."
        return unflatten(tuple(self._decode[i] for i in self.tokenizer.encode(prompt)))

    def initial(self):
        return StateLM.initial(self)

    def state_tokenized(self, context):
        return self.initial().advance(flatten(self.encode_prompt(context)))

    def states_beam(self, context, **kwargs):
        from tokenization.character_beam_trie import CharacterBeam
        return CharacterBeam(self, **kwargs).candidates(context)

    def states_token_healing(self, context, **kwargs):
        from tokenization.character_beam_trie import TokenHealingHeuristic
        return TokenHealingHeuristic(self, **kwargs).candidates(context)


def load_model_by_name(model_name, device=None, **kwargs):
    """
    Load an LLM from 🤗 into a `TokenizedLLM`.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return TokenizedLLM(
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False),
        model = (
            transformers.AutoModelForCausalLM.from_pretrained(model_name)
            .to(device)
            .eval()             # Set the model in evaluation mode; avoids gradient overhead,
        ),
        **kwargs,
    )


class StateLM:
    def __init__(self, lm, logp, context, parent):
        self.lm = lm
        self.logp = logp
        self.context = context
        self.parent = parent

    def __lshift__(self, x):
        return StateLM(
            lm = self.lm,
            logp = self.logp + self.logp_next[x],
            context = (self.context, x),
            parent = self,
        )

    @cached_property
    def logp_next(self):
        with torch.no_grad():
            return LazyProb(
                torch.nn.functional.log_softmax(self.out.logits, dim=-1).squeeze().detach().numpy(),
                self.lm._encode, self.lm._decode
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
                i = lm._encode[x]   # look up token id
                input_ids = torch.LongTensor([[i]], device=lm.device)
                return self.lm.model(
                    input_ids=input_ids,
                    past_key_values=self.parent.out.past_key_values,
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
        if isinstance(lm, str): lm = load_model_by_name(lm, byte_level=True)
        # Note: initial state uses BOS padding.
        return cls(
            lm = lm,
            logp = 0,
            context = (),
            parent = None,
        )

    def sample_next_token(self):
        return sample_dict(self.p_next)

    def sample(self):
        return self << self.sample_next_token()

    def advance(self, xs):
        assert isinstance(xs, tuple) and all(isinstance(x, (bytes, str)) for x in xs), xs
        s = self
        for x in xs:
            s <<= x
        return s
