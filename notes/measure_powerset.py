"""Measure powerset state sizes in the determinized PrecoverNFA for BPE FSTs."""
import time
from transformers import AutoTokenizer
from transduction.applications.bpe import bpe_wfst
from transduction.fst import FST
from transduction.fsa import EPSILON
from transduction.lm.statelm import decode_hf_tokenizer
from transduction.util import set_memory_limit
set_memory_limit(8)

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
_, _, _decode, _ = decode_hf_tokenizer(tokenizer)
drop = {x.encode() for x in tokenizer.all_special_tokens}

all_token_ids = sorted(i for i in range(len(_decode)) if _decode[i] not in drop)

def subsampled_bpe_fst(decode, token_ids, drop=frozenset()):
    m = FST()
    m.add_start(())
    for i in token_ids:
        x = decode[i]
        if x in drop:
            continue
        bx = tuple(x)
        for j in range(len(bx)):
            m.add_arc(bx[:j], EPSILON, bx[j], bx[:j+1])
        m.add_arc(bx, i, EPSILON, ())
    m.add_stop(())
    return m.renumber()

target_text = "The quick brown fox"
target_bytes = list(target_text.encode())
target_seq = target_bytes

for vocab_size in [43, 100, 500, 1000, 5000, 10000, 50000]:
    used_ids = all_token_ids[:vocab_size]
    fst = subsampled_bpe_fst(_decode, used_ids, drop)

    # Use the Rust backend to build the decomposition DFA
    from transduction.rust_bridge import RustPeekabooState
    try:
        t0 = time.perf_counter()
        ps = RustPeekabooState(fst, target=(), parent=None, univ=None)
        # Force BFS computation
        _ = ps.dfa
        init_time = time.perf_counter() - t0

        # Decode the start DFA state to see its NFA state set
        start_id = ps.dfa._start_id
        nfa_states = ps.decode_dfa_state(start_id)
        print(f'VOCAB={vocab_size:6d}  '
              f'FST states={len(fst.states):6d}  '
              f'init={init_time*1000:.1f}ms  '
              f'start DFA state has {len(nfa_states)} NFA states')

        # Now extend by a few target bytes and check sizes
        ps1 = ps
        for step in range(min(3, len(target_seq))):
            y = target_seq[step]
            t0 = time.perf_counter()
            ps1 = ps1 >> y
            _ = ps1.dfa
            step_time = time.perf_counter() - t0

            start_id = ps1.dfa._start_id
            nfa_states = ps1.decode_dfa_state(start_id)
            print(f'  step {step+1} (byte {y!r}): '
                  f'{step_time*1000:.1f}ms  '
                  f'start DFA state has {len(nfa_states)} NFA states')

    except Exception as e:
        print(f'VOCAB={vocab_size:6d}  ERROR: {type(e).__name__}: {e}')
    print()
t
