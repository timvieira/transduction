import math
import torch
from evo2 import Evo2

from vortex.utils.inference import InferenceParams


class Evo2Scorer:
    def __init__(self, model_name="evo2_1b_base", device=None, use_fp8=False):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model {model_name} on {self.device}...")
        
        self.wrapper = Evo2(model_name)
        self.model = self.wrapper.model
        self.tokenizer = self.wrapper.tokenizer
        
        if not use_fp8:
            print("Patching model to disable FP8 projections...")
            count = 0
            for name, module in self.model.named_modules():
                if hasattr(module, 'use_fp8_input_projections'):
                    module.use_fp8_input_projections = False
                    count += 1
            print(f"Disabled FP8 on {count} layers.")
            
        self.model.to(dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.eval()

        self.vocab_map = {}
        # We assume standard DNA upper case. 
        for char in ['A', 'C', 'G', 'T']:
            ids = self.tokenizer.tokenize(char)
            self.vocab_map[char] = ids[-1]
            
        print(f"Vocab Map: {self.vocab_map}")
        self.cache = {}

    def __call__(self, seq):
        if seq in self.cache:
            return self.cache[seq]

        input_ids = self._prepare_input(seq)

        with torch.no_grad():
            logits, _ = self.model(input_ids)
        
        dist, _ = self._process_logits(logits, None)

        self.cache[seq] = dist
        return dist

    def get_initial_state(self):
        input_ids = torch.tensor([[self.tokenizer.eod_id]]).to(self.device)
        
        # Initialize Hyena Inference Params (The "KV Cache" equivalent)
        inference_params = InferenceParams(
            max_seqlen=8192, 
            max_batch_size=1,
            seqlen_offset=0
        )

        with torch.no_grad():
            logits, _ = self.model(
                input_ids, 
                inference_params_dict=inference_params
            )
            
        inference_params.seqlen_offset += input_ids.shape[1]
        
        return self._process_logits(logits, inference_params)

    def step(self, state, new_char):
        inference_params = state
        
        token_id = self.vocab_map.get(new_char)
        if token_id is None:
            raise ValueError(f"Invalid character: {new_char}")

        input_ids = torch.tensor([[token_id]]).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(
                input_ids, 
                inference_params_dict=inference_params
            )
        
        inference_params.seqlen_offset += 1
        
        return self._process_logits(logits, inference_params)

    def _prepare_input(self, seq):
        if not seq:
            return torch.tensor([[self.tokenizer.eod_id]]).to(self.device)
        
        token_ids = self.tokenizer.tokenize(seq)
        return torch.tensor([token_ids], dtype=torch.long).to(self.device)

    def _process_logits(self, logits, state):
        # Logits shape: (batch, seq_len, vocab_size)
        # We take the last token
        next_token_logits = logits[0, -1, :]
        next_token_log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        dist = {}
        for char, token_id in self.vocab_map.items():
            dist[char] = next_token_log_probs[token_id].item()
        
        return dist, state