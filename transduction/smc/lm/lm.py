import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFLM:
    def __init__(self, model_name="vesteinn/gpt2-dna", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Optimization: Pre-calculate token IDs for nucleotides
        self.vocab_map = {}
        for token, idx in self.tokenizer.vocab.items():
            if token not in self.tokenizer.all_special_tokens:
                self.vocab_map[token] = idx

        # Optimization: Internal Cache for the __call__ method
        self.cache = {}

    def __call__(self, seq):
        """
        Implements the prior interface: get_source_lm_probs(seq).
        Calculates P(next_char | seq).
        Includes caching to prevent re-computation of the same string.
        """
        # 1. Check Cache
        if seq in self.cache:
            return self.cache[seq]

        # 2. Prepare Input
        if not seq:
            # Handle start of sequence (BOS)
            if self.tokenizer.bos_token_id is not None:
                input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
            else:
                return {k: math.log(1/len(self.vocab_map)) for k in self.vocab_map}
        else:
            # Tokenize full sequence
            inputs = self.tokenizer(seq, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)

        # 3. Run Model (Non-incremental pass)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # 4. Process Output
        # Helper extracts valid nucleotide probs from the LAST position
        dist, _ = self._process_logits(logits, None)

        # 5. Update Cache and Return
        self.cache[seq] = dist
        return dist

    def get_initial_state(self):
        """
        Starts a new incremental sequence. 
        Returns: (log_probs_dict, past_key_values)
        """
        if self.tokenizer.bos_token_id is not None:
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
        else:
            return {k: math.log(0.25) for k in self.vocab_map}, None

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        
        return self._process_logits(outputs.logits, outputs.past_key_values)

    def step(self, past_key_values, new_char):
        """
        Updates the state with one new character.
        Returns: (log_probs_dict, new_past_key_values)
        """
        token_id = self.vocab_map.get(new_char)
        if token_id is None:
            raise ValueError(f"Invalid character: {new_char}")

        input_ids = torch.tensor([[token_id]]).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
        
        return self._process_logits(outputs.logits, outputs.past_key_values)

    def _process_logits(self, logits, past_key_values):
        """Internal helper to format logits into a dict."""
        next_token_logits = logits[0, -1, :]
        next_token_log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        dist = {}
        for char, token_id in self.vocab_map.items():
            if token_id is not None:
                dist[char] = next_token_log_probs[token_id].item()
            else:
                dist[char] = float('-inf')
        
        return dist, past_key_values