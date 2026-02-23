import torch
import numpy as np
from time import time
from arsenal import timeit
from tokenization.util import prefixes
try:
    from tokenization.vllm import load_model_by_name
except ImportError:
    load_model_by_name = None
    from tokenization.lm import load_model_by_name
    import warnings
    warnings.warn('vllm not available, so batched CharacterBeam is not available either.')
from tokenization.lm import Chart
from tokenization.charlm import CharLM
from tokenization.batch_trie import BatchTokenCharacterTrie
from tokenization.util import format_table
from arsenal.maths import logsumexp
from contextlib import contextmanager, nullcontext


class Bundle:
    def __init__(self, *, key, wpath, mass, node, trie, n_tokens):
        self.key = key
        self.wpath = wpath
        self.node = node
        self.mass = mass
        self.weight = wpath + np.log(mass[node])
        self.trie = trie
        self.n_tokens = n_tokens # used for logging
        self._extend = None

    def filter(self, curr_char):
        if curr_char not in self.trie.children[self.node]:
            return
        return Bundle(
            key = self.key,
            mass = self.mass,
            node = self.trie.children[self.node][curr_char],
            trie = self.trie,
            n_tokens = self.n_tokens,
            wpath = self.wpath,
        )

    def extend(self, new_mass):
        if self._extend is not None:
            return self._extend
        node = self.trie.children[self.node].get(None)
        if node is None:
            return
        token_string = self.trie.leaf2word[node]
        new_key = (self.key, token_string)
        value = Bundle(
            key = new_key,
            node = self.trie.root,
            mass = new_mass,
            trie = self.trie,
            n_tokens = self.n_tokens + 1,
            wpath = self.wpath + np.log(self.mass[node])
        )
        self._extend = value
        return value

    def get_curr_leaf(self):
        node = self.trie.children[self.node].get(None)
        if node is None: # not at leaf
            return
        return self.trie.leaf2word[node]

    def p_next(self):
        return Chart(0, {char: self.mass[i] for char, i in self.trie.children[self.node].items()})

    def items(self):
        "Method for inspecting a bundle"
        agenda = [self.node]
        while agenda:
            i = agenda.pop()
            for symbol, j in self.trie.children[i].items():
                if symbol is None:
                    yield (j, self.mass[j])
                else:
                    agenda.append(j)

    def about_str(self):
        lines = [f"Current prefix: {repr(self.trie.node2prefix[self.node])}"]
        lines.append(f"Weight: {self.weight:.2f}")
        lines.append("Continuations:")

        children = self.trie.children[self.node].items()
        children = sorted(children, key=lambda x: (x[0] is not None, x[0] or ''))

        for i, (char, node_id) in enumerate(children):
            is_last = i == len(children) - 1
            prefix = '└── ' if is_last else '├── '
            char_display = 'EOT' if char is None else repr(char)[1:-1]
            mass = self.mass[node_id]
            lines.append(f"{prefix}{char_display:<4} → {node_id} (mass={mass:.6f})")

        return '\n'.join(lines)

class BeamStats:
    def __init__(self):
        self.forward_passes_per_step = []
        self.beam_size_per_step = []
        self.batch_size_per_step = []
        self.times = {}
        self.n_tokens_per_step = []
        self.current_step = 0
        self.tokens_processed_by_lm_per_step = []

    def next_step(self):
        # Called at the start of processing each new character
        self.current_step += 1
        self.forward_passes_per_step.append(0)
        self.beam_size_per_step.append(0)
        self.batch_size_per_step.append(0)
        self.n_tokens_per_step.append(0)
        self.tokens_processed_by_lm_per_step.append(0)

    def add_forward_passes(self, count):
        if self.forward_passes_per_step:
            self.forward_passes_per_step[-1] = count

    def add_beam_size(self, size):
        if self.beam_size_per_step:
            self.beam_size_per_step[-1] = size

    def add_batch_size(self, size):
        if self.batch_size_per_step:
            self.batch_size_per_step[-1] = size

    def add_step_time(self, t, name):
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(t)

    def add_n_tokens(self, n_tokens):
        if self.n_tokens_per_step:
            self.n_tokens_per_step[-1] = n_tokens

    def add_tokens_processed_by_lm(self, n_tokens):
        if self.tokens_processed_by_lm_per_step:
            self.tokens_processed_by_lm_per_step[-1] = n_tokens

    def get_stats(self):
        return {
            'forward_passes_per_step': self.forward_passes_per_step,
            'beam_size_per_step': self.beam_size_per_step,
            'batch_sizes_per_step': self.batch_size_per_step,
            'n_tokens_per_step': self.n_tokens_per_step,
            'total_steps': self.current_step,
            'tokens_processed_by_lm_per_step' : self.tokens_processed_by_lm_per_step
        }

    @contextmanager
    def timer(self, name):
        start_time = time()
        try:
            yield
        finally:
            elapsed = time() - start_time
            self.add_step_time(elapsed, name)

def maybe_time(stats, name):
    return stats.timer(name) if stats else nullcontext()

def prune_top_p(candidates, top_p):
    # Prune candidates based on cumulative probability mass.
    Ws = [b.weight for b in candidates]
    Z = logsumexp(Ws)
    ps = np.exp(Ws - Z)
    candidate_ps = list(zip(candidates, ps))
    candidate_ps.sort(key=lambda x: x[1], reverse=True)

    new_candidates = []
    accumulated = 0.0
    for candidate, p in candidate_ps:
        accumulated += p
        new_candidates.append(candidate)
        if accumulated >= top_p:
            break

    return new_candidates

class CharacterBeam(CharLM):

    def __init__(
        self, llm, K, top_p=1, eos='▪', log_stats=False, beam_cache_size=1, p_next_cache_size=0
    ):
        self.llm = llm
        self.K = K
        self.top_p = top_p
        self.log_stats = log_stats
        self.eos = eos

        # The `trie` created below is a data structure that tells us how to
        # build and navigate the `mass` vector.  It names nodes by integers and
        # stores the various mappings for doing that.
        self.trie = BatchTokenCharacterTrie(
            words = llm._decode,
            encode = llm._encode,
            old_eos = llm.eos,
            new_eos = self.eos
        )

        self._beam_cache = self._init_beam()
        self._p_next_cache = {}
        self.beam_cache_size = beam_cache_size
        self._beam_cache_keys = []
        self.p_next_cache_size = p_next_cache_size
        self._p_next_cache_keys = []

        self.char_V = set(''.join(llm._decode))

        self.stats = BeamStats() if log_stats else None

    def _init_beam(self):
        return {
            '': [
                Bundle(
                    key = (),
                    mass = self.trie.batch_mass_sum(
                        torch.Tensor([self.llm.logp_next(()).apply(np.exp)._p])
                    ),
                    node = self.trie.root,
                    trie = self.trie,
                    n_tokens = 0,
                    wpath = 0,
                )
            ]
        }

    def beam(self, qs):

        result = self._beam_cache.get(qs)
        if result is not None:
            return result

        beam = self.beam(qs[:-1])

        with maybe_time(self.stats, 'beam'):
            result = self._beam(beam, qs)

        if not result:
            raise ValueError("Empty beam.")

        self._beam_cache[qs] = result
        self._beam_cache_keys.append(qs)

        while len(self._beam_cache_keys) > self.beam_cache_size:
            oldest_key = self._beam_cache_keys.pop(0)
            del self._beam_cache[oldest_key]

        return result

    def _beam(self, beam, qs):
        curr_char = qs[-1]

        assert curr_char in self.char_V, curr_char

        candidates = []
        bundles_to_extend = []

        for bundle in beam:
            new_bundle = bundle.filter(curr_char)
            if new_bundle is not None:
                candidates.append(new_bundle)

            curr_leaf = bundle.get_curr_leaf()
            if curr_leaf is not None: # EOT avail
                if bundle._extend is not None: # Reuse cached value
                    new_bundle = bundle._extend.filter(curr_char)
                    if new_bundle is not None:
                        candidates.append(new_bundle)
                else:
                    bundles_to_extend.append((bundle, curr_leaf))

        if bundles_to_extend:
            keys = [(bundle.key, curr_leaf) for bundle, curr_leaf in bundles_to_extend]
            masses = self.batch_trie_update(keys).cpu().numpy()

            for i, (bundle, _) in enumerate(bundles_to_extend):
                new_bundle = bundle.extend(masses[i]).filter(curr_char)
                if new_bundle is not None:
                    candidates.append(new_bundle)

        new_candidates = prune_top_p(candidates, self.top_p) if self.top_p < 1 else candidates

        result = sorted(new_candidates, key=lambda bundle: -bundle.weight)[:self.K]

        return result

    def p_next(self, context):
        if self.stats:
            self.stats.next_step()

        with maybe_time(self.stats, 'p_next'):
            if self.p_next_cache_size > 0:
                result = self._p_next_cache.get(context)
                if result is not None:
                    return result
            result = self._p_next(context)

        if self.p_next_cache_size > 0:
            self._p_next_cache[context] = result
            self._p_next_cache_keys.append(context)
            while len(self._p_next_cache_keys) > self.max_p_next_cache_size:
                oldest_key = self._p_next_cache_keys.pop(0)
                del self._p_next_cache[oldest_key]

        return result

    def _p_next(self, context):
        beam = self.beam(context)

        Ws = [b.weight for b in beam]
        Z_beam = logsumexp(Ws)

        Q = Chart(0.0)
        bundles_to_extend = []
        for bundle in beam:
            q = bundle.p_next()
            ppath = np.exp(bundle.weight - Z_beam)
            for k in q:
                if k is not None:
                    Q[k] += ppath * q[k]

            curr_leaf = bundle.get_curr_leaf()
            if curr_leaf is not None:
                bundles_to_extend.append((bundle, curr_leaf))

        if self.log_stats:
            self.stats.add_beam_size(len(beam))
            self.stats.add_batch_size(len(bundles_to_extend))
            self.stats.add_n_tokens(sum([b.n_tokens for b in beam]))

        if bundles_to_extend:
            keys = [(bundle.key, curr_leaf) for bundle, curr_leaf in bundles_to_extend]
            masses = self.batch_trie_update(keys).cpu().numpy()

            for i, (bundle, _) in enumerate(bundles_to_extend):
                B = bundle.extend(masses[i])
                if B is not None:
                    q = B.p_next()
                    ppath = np.exp(B.weight - Z_beam)
                    for k in q:
                        if k is not None:
                            Q[k] += ppath * q[k]

        return Q.normalize() # redundant

    def logp_next(self, context):
        return self.p_next(context).map_values(np.log)

    def batch_trie_update(self, keys):
        p_nexts = self.llm.batch_p_next(
            keys, keep_on_gpu=True, stats=self.stats
        )
        masses = self.trie.batch_mass_sum(p_nexts)
        return masses

    def clear_cache(self):
        self._p_next_cache = {}
        self._beam_cache = self._init_beam()

    def show_html(self, context, limit=3):
        from IPython.display import HTML, display

        table = []
        for xs in prefixes(context):
            column = []
            for bundle in self.beam(xs):
                column.append(bundle.about_html(limit=limit))
            table.append(column + ['']*(self.K - len(column)))

        display(HTML(format_table(zip(*table), headings=list(map(repr, prefixes(context))))))


#_______________________________________________________________________________
#


def test_basics():
    llm = load_model_by_name('gpt2-large')
    C = CharacterBeam(llm, K=5)
    context = 'An apple a day keeps '
    beam = C.beam(context)
    print(beam)
    result = C.greedy(context, steps=20)
    print(result)



def test_generate():
    llm = load_model_by_name('gpt2')
    K = 5
    qs = 'An apple a day keeps the '
    M = CharacterBeam(llm, K=K)
    with timeit('greedy generation'):
        output = M.greedy(qs, steps=12, verbose=True)
    print(repr(output))
    assert output == 'An apple a day keeps the doctor away.'



if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
