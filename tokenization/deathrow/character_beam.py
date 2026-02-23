import numpy as np
import pylab as pl
from time import time
from arsenal import iterview, timers, timeit
from collections import defaultdict
from numpy import logaddexp

from tokenization.basics import Item, Beam
from tokenization.lm import load_model_by_name
from tokenization.util import Chart, prefixes, logsumexp
from tokenization.charlm import CharLM


class character_beam_estimator(CharLM):

    def __init__(self, llm, *, K, eos='▪'):
        self.eos = eos
        self.llm = llm
        self.K = K

        self._p_next_cache = {}
        self._beam_cache = {}

        self.ix = defaultdict(list)
        self.vocab = self.llm._decode
        for i, y in enumerate(self.vocab):
            self.ix[y[0]].append(i)
        self.lens = np.array([len(y) for y in self.vocab])

    def _beam(self, qs):

        N = len(qs)

        if N == 0:
            return Beam([Item(ps=0, ys=(), offset=0, parent=None)])

        beam = self.beam(qs[:-1])
        curr_char = qs[-1]

        candidates = Beam()
        n_expanded = 0

        for item in beam:  # Note: already sorted
            if item.offset < N:
                assert n_expanded < self.K
                n_expanded += 1
                logp = self.llm.logp_next(item.ys)
                _ps = item.ps + logp._p
                _offset = item.offset + self.lens
                for i in self.ix[curr_char]:
                    candidates.append(Item(
                        ps = _ps[i],
                        offset = _offset[i],
                        ys = (item.ys, self.vocab[i]),
                        parent = item,
                    ))
            else:
                next_char = item.ys[1][N - item.parent.offset - 1]
                if next_char == curr_char:
                    candidates.append(item)

        #result = candidates.sort()

        # sort by total bucket weight
        buckets = candidates.groupby(
            key=lambda item: (item.ys if item.offset == N else item.ys[0])
        )
        top_K_buckets = sorted(buckets.values(), key=lambda bucket: -bucket.logsumexp())[:self.K]
        result = Beam()
        for bucket in top_K_buckets:
            result.extend(bucket)
        result = result.sort()

        assert len(result) <= self.K * len(self.vocab)

        return result

    def beam(self, qs):
        result = self._beam_cache.get(qs)
        if result is not None:
            return result
        result = self._beam(qs)
        self._beam_cache[qs] = result
        return result

    def logp_next(self, context):
        result = self._p_next_cache.get(context)
        if result is not None:
            return result
        result = self._logp_next(context)
        self._p_next_cache[context] = result
        return result

    def _logp_next(self, context):
        candidates = self.beam(context)
        Q = Chart(-np.inf)
        N = len(context)
        #assert self.eos not in context
        #print(f'_logp_next {context=}, {len(candidates)=}')
        for item in candidates:
            # Note: this case cannot be triggered in the character_beam2
            if item.offset == N:
                # extend to include at least one more character
                logp = self.llm.logp_next(item.ys)
                Q[self.eos] = item.ps + logp[self.llm.eos]
                for y, logpy in logp.items():
                    if y == self.llm.eos: continue
                    next_char = y[0]
                    Q[next_char] = logaddexp(Q[next_char], item.ps + logpy)
            else:
                assert item.offset > N
                next_char = item.ys[1][N - item.parent.offset]
                Q[next_char] = logaddexp(Q[next_char], item.ps)
        Z = logsumexp(list(Q.values()))
        A = Chart(-np.inf)
        for next_char in Q:
            A[next_char] = Q[next_char] - Z
        return A



def test_new_character_beam():
    llm = load_model_by_name('gpt2')
    K = 5

#    qs = 'Therefore, I am unimpressed with the speedup.'
    qs = 'Therefore, I am unimpressed.'

    T = timers()

    logp1 = {}

    llm.clear_cache()
    M = character_beam_estimator(llm, K=K)
    for context in iterview(list(prefixes(qs))):
        with T['v1'](N=len(context)):
            logp1[context] = M.logp_next(context)

    T.plot_feature('N')

    T.compare()

    pl.show()


def test_generate():
    llm = load_model_by_name('gpt2-large')
    K = 5
    qs = 'An apple a day keeps the '
    M = character_beam_estimator(llm, K=K)
    with timeit('greedy generation'):
        output = M.greedy(qs, steps=12, verbose=True)
    print(repr(output))
    assert output == 'An apple a day keeps the doctor away.'


def test_profile():
    from arsenal.profiling import profiler

    llm = load_model_by_name('gpt2')
    K = 5

    qs = 'Therefore, I am unimpressed with the speedup.'

    M = character_beam_estimator(llm, K=K)

#    if 1:
#    with timeit('run'):
    with profiler():
        for context in iterview(list(prefixes(qs))):
            M.logp_next(context)


def test_memory_benchmark():
    from arsenal.viz import update_ax

    llm = load_model_by_name('gpt2')
    K = 5

    context = """There are now rumblings that Apple might soon invade the smart watch space, \
though the company is maintaining its customary silence. The watch doesn't have a \
microphone or speaker, but you can use it to control the music on your phone. You can \
glance at the watch face to view the artist and title of a song."""

    T = timers()

    C = character_beam_estimator(llm, K=K)

    ax = pl.figure().add_subplot(111)

    import gc
    gc.disable()

    surprisals = []
    for xs in iterview(list(prefixes(context))):
        if len(xs) == 0: continue
        with T['estimator'](N=len(xs)):

            before = time()
            logp = C.logp_next(xs)
            took = time() - before

        if len(xs) < len(context):
            x = context[len(xs)]
            print(f'{x!r} surprisal: {-logp[x]:.4f} {[len(xs), len(C.llm._cache), len(C._beam_cache)]} {took:.4f} sec')
            surprisals.append(-logp[x])

#        gc.collect()

        with update_ax(ax):
            T.plot_feature('N', ax=ax, show_curve=(len(xs) > 2))
#        if len(xs) >= 142:
#            break

    pl.ioff()
    pl.show()


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
