from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

class HFModel:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.vocab_map = {}
        for token, idx in self.tokenizer.get_vocab().items():
            if token not in self.tokenizer.all_special_tokens:
                self.vocab_map[token] = idx

        self.last_input_ids = None
        self.past_key_values = None

    def logp_next(self, seq):
        if not seq:
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
        else:
            inputs = self.tokenizer(seq, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)

        use_cache = False
        new_input_ids = input_ids

        if self.past_key_values is not None and self.last_input_ids is not None:
            prev_len = self.last_input_ids.shape[1]
            curr_len = input_ids.shape[1]
            
            if curr_len > prev_len and torch.equal(input_ids[:, :prev_len], self.last_input_ids):
                use_cache = True
                new_input_ids = input_ids[:, prev_len:]

        with torch.no_grad():
            if use_cache:
                outputs = self.model(
                    new_input_ids, 
                    past_key_values=self.past_key_values,
                    use_cache=True
                )
            else:
                outputs = self.model(
                    new_input_ids, 
                    use_cache=True
                )

        self.past_key_values = outputs.past_key_values
        self.last_input_ids = input_ids

        next_token_logits = outputs.logits[0, -1, :]
        return self._process_logits(next_token_logits)
    
    def _process_logits(self, next_token_logits):
        next_token_log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        dist = {}
        log_probs_list = next_token_log_probs.tolist()
        
        for token_str, token_id in self.vocab_map.items():
            dist[token_str] = log_probs_list[token_id]
        
        return dist


if __name__ == "__main__":
    import time
    import torch

    print("Loading model...")
    lm = HFModel("gpt2")
    
    lm.logp_next("Warmup")
    lm.past_key_values = None
    lm.last_input_ids = None

    base_text = "The quick brown fox jumps over the lazy dog. " * 75 
    print(f"\nSequence length: ~{len(base_text.split())} words")

    t0 = time.perf_counter()
    dist = lm.logp_next(base_text)
    t1 = time.perf_counter()
    print(f"1. Initial (scratch): {(t1 - t0)*1000:.2f} ms")

    text_new = base_text + " And"
    
    inputs = lm.tokenizer(text_new, return_tensors="pt")
    input_ids = inputs["input_ids"].to(lm.device)
    
    new_token_id = input_ids[:, -1:] 
    
    print(f"2. Extension (With Cache)...")
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = lm.model(
            new_token_id, 
            past_key_values=lm.past_key_values,
            use_cache=True
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"   Time: {(t1 - t0)*1000:.2f} ms")

    print(f"3. Extension (Cache Cleared)...")
    lm.past_key_values = None
    
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = lm.model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"   Time: {(t1 - t0)*1000:.2f} ms")