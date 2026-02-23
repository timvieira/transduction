import numpy as np
from tokenization.basics import Beam, Item
from tokenization.util import unflatten, logsumexp, LazyProb
from tokenization.bpe import BPE

from arsenal.maths import sample_dict

from collections import namedtuple
WeightedSample = namedtuple('WeightedSample', 'sample, logp, logweight')


# TODO: use fast filter
class GreedyCoverOurs:
    """
    An implementation of [Phan et al.'s (2024)](https://arxiv.org/abs/2410.09303)
    approach as a pruning heuristic in our :func:`tokenization.character_beam_trie.CharacterBeam`.
    """

    def __init__(self, lm):
        self.bpe = BPE.from_huggingface(lm.tokenizer)
        self.lm = lm

    def beam(self, qs):
        N = len(qs)
        if N == 0:
            return Beam([Item(ps=0, ys=(), offset=0, parent=None)])
        curr_char = qs[-1]
        candidates = Beam()
        lm = self.lm
        bpe = self.bpe
        for item in self.beam(qs[:-1]):
            if item.offset < N:
                logp = lm.logp_next(item.ys)
                for i in range(len(lm._decode)):
                    y = lm._decode[i]
                    if y[0] == curr_char:
                        if item.ys == () or bpe._incremental_canonicality(lm._encode[item.ys[1]], i):
                            candidates.append(Item(
                                ps = item.ps + logp._p[i],
                                offset = item.offset + len(y),
                                ys = (item.ys, y),
                                parent = item,
                            ))
            else:
                next_char = item.ys[1][N - item.parent.offset - 1]
                if next_char == curr_char:
                    candidates.append(item)
        return Beam(candidates).sort()


# TODO: use fast filter
class GreedyCoverFB:
    """
    Implementation of [Phan et al. (2024)](https://arxiv.org/abs/2410.09303)
    """
    def __init__(self, lm, strict=False, canonicalize=None):
        self.lm = lm
        self.bpe = BPE.from_huggingface(lm.tokenizer)
        self.strict = strict
        self.canonicalize = canonicalize

    def beam(self, xs):
        N = len(xs)
        Zs = Beam()
        bpe = self.bpe
        lm = self.lm
        # figure out where to split the string into two parts
        # (1) a partially matched final token (this is essentially a better version of token healing)
        # (2) a prefix that is stably tokenized (i.e., adding more characters to the string can't change it)
        # Not all splits match any tokens in part (1)
        # Not all splits + healed-token pass the canonical check
        for n in reversed(range(1 + N)):
            y = xs[N-n:]   # candidate for the partially matched last token
            if not y: continue
            ys = unflatten(bpe.encode_as_byte_chunks(xs[:N-n]))
            B = [v for v in lm.V if v.startswith(y)]
            # prefix weight of ys
            logP = self.lm.logprefix(ys)
            logp = self.lm.logp_next(ys)
            parent = Item(ps=logP, offset=N-n, ys=ys, parent=None)
            for yy in B:
                if (
                    not self.canonicalize
                    or parent.ys == ()
                    or bpe._incremental_canonicality(lm._encode[parent.ys[1]], lm._encode[yy])
                ):
                    Zs.append(Item(ps=logP + logp[yy], ys=(ys, yy), parent=parent, offset=len(yy)+N-n))
        return Zs.sort()



# TODO: This is a temporary measure until LocallyCanonical implements an
# efficient the stateful interface.
class LocallyCanonicalState:
    def __init__(self, lm, context):
        self.lm = lm
        self.context = context
        self.logp_next = self.lm.logp_next(context)
        self.logp = self.lm.logprefix(context)

    def __lshift__(self, a):
        return LocallyCanonicalState(self.lm, (self.context, a))


# TODO: stateful implementation
# TODO: many of these methods could be inherited from the LM class.
class LocallyCanonical:
    """Locally conditioned model / proposal distrbution.

    Defines a language model $q$ based on locally conditioning $p$ (i.e., `lm`).
    The specific condition employed in this distribution is given by
    :func:`tokenization.canonicality_filter.FastCanonicalityFilterBPE`.

    """
    def __init__(self, lm, overrides=()):
        self.lm = lm
        self.eos = lm.eos
        self.eos_id = lm._encode[lm.eos]
        self.V = lm.V

        self.bpe = BPE.from_huggingface(lm.tokenizer)
        self.bpe.overrides.update(overrides)
        self.fast_filter = self.bpe.make_fast_filter(self.eos_id)
        self._decode = lm._decode
        self._encode = lm._encode
        self.tokenizer = lm.tokenizer
        self.device = lm.device

    def initial(self):
        return LocallyCanonicalState(self, ())

    def logp_next(self, context, logZ=False):
        "log prefix probability over next tokens or eos."
        lm = self.lm
        mask = self.fast_filter(context)
        p = lm.logp_next(context)
        q = p.copy()
        q._p[~mask] = -np.inf
        logZZ = logsumexp(q._p)
        q._p = q._p - logZZ
        return (q, logZZ) if logZ else q

    def logwarp_prefix(self, context):
        "log p.prefix(context)/q.prefix(context)."
        if len(context) == 0:
            return 0.0
        else:
            # Note: warp is invariant to the last token of `context` that's
            # because it is the product of the normalizing constants up to the
            # last token.  The last token was sampled from the previous
            # normalizing constant.
            (context, _) = context
            return self.logwarp_prefix(context) + self.logp_next(context, logZ=True)[1]

    def logwarp(self, context):
        "log p(context)/q(context)."
        return self.logwarp_prefix((context, None))

    def logprefix(self, context):
        "log prefix probability"
        assert isinstance(context, tuple) and len(context) == 0 or len(context) == 2, context
        if len(context) == 0:
            return 0.0
        else:
            context, y = context
            return self.logprefix(context) + self.logp_next(context)[y]

    def logp(self, context):
        "log-probability of a [complete] string `context`."
        return self.logprefix(context) + self.logp_next(context)[self.eos]

    def sample(
        self,
        ys=(),
        draw=sample_dict,
        verbose=0,
        max_tokens=np.inf,
    ):
        "Draw a sample from this distribution."
        assert isinstance(ys, tuple) and len(ys) in {0, 2}, ys
        logP = 0
        logW = 0
        t = 0
        while True:
            logp, logw = self.logp_next(ys, logZ=True)
            p = logp.apply(np.exp)
            y = draw(p) if t < max_tokens else self.eos
            logP += logp[y]
            logW += logw
            t += 1
            if verbose:
                if y == self.eos:
                    print()
                else:
                    print(y, end='')
            if y == self.eos:
                return WeightedSample(ys, logP, logW)
            ys = (ys, y)

    def fancy_step(self, token_ids):
        return local_canonicalization_step(lm=self.lm, bpe=self.bpe, token_ids=token_ids)

    def fancy_path(self, context):
        from tokenization.util import LocalLeakageInteractive
        return LocalLeakageInteractive(self, context)


class local_canonicalization_step:
    """
    This class is just a helper for playing around with local canonicalization
    """
    def __init__(self, lm, bpe, token_ids):
        assert isinstance(token_ids, list) and all(isinstance(x, int) for x in token_ids)
        self.lm = lm
        self.bpe = bpe
        prompt_bytes = bpe.token_ids_to_byte_chunks(token_ids)
        logp_next = lm.logp_next(unflatten(prompt_bytes))
        total_blocked = -np.inf
        total_allowed = -np.inf
        blocked = []
        allowed = []
        # TODO: use the vectorized method here too
        for y in logp_next.top(None):
            i = lm._encode[y]
            if y == lm.eos or bpe.is_canonical_incremental(token_ids, i):
                total_allowed = np.logaddexp(total_allowed, logp_next[y])
                allowed.append(y)
            else:
                total_blocked = np.logaddexp(total_blocked, logp_next[y])
                blocked.append(y)

        p = logp_next
        q = LazyProb(p._p.copy(), encode=p._encode, decode=p._decode)
        for y in blocked:
            q._p[lm._encode[y]] = -np.inf
        q._p = q._p - logsumexp(q._p)

        self.raw_logp_next = logp_next
        self.new_logp_next = q

        self.total_blocked = total_blocked
        self.total_allowed = total_allowed

        self.blocked = blocked
        self.allowed = allowed
        self.token_ids = token_ids
        self.prompt_bytes = prompt_bytes

    def show_ranks(self, M=5000, ax=None):
        import pylab as pl
        if ax is None: ax = pl.figure(figsize=(18,4)).add_subplot(111)
        s = self
        order = np.argsort(-s.raw_logp_next._p)
        rank = {i: rank for rank, i in enumerate(order)}
        ps = np.exp(s.raw_logp_next._p)[order][:M]
        ax.plot(ps)
        ax.semilogy()

        # Calculate and plot cumulative blocked probability
        blocked_mass = np.zeros_like(ps)
        for i in s.blocked:
            r = rank[self.lm._encode[i]]
            if r < M:
                ax.axvline(r, lw=.1, color='red')
                blocked_mass[r] = ps[r]  # Just mark the probability at this position

        # Compute cumulative sum in one pass
        cumulative_blocked = np.cumsum(blocked_mass)

        # Plot cumulative blocked probability on the same axis
        ax.plot(cumulative_blocked, 'r--',
                label=f'Cumulative blocked prob (total: {cumulative_blocked[-1]:.3f})')

        ax.axhline(np.exp(self.total_blocked), c='r')

    def show_ranks_interactive(self, M=5000):
        "Create an interactive plot of ranks with tooltips for vertical lines."
        import plotly.graph_objects as go

        # Convert log probabilities to probabilities and sort
        probs = np.exp(self.raw_logp_next._p)
        order = np.argsort(-self.raw_logp_next._p)[:M]
        sorted_probs = probs[order]

        # Create rank mapping
        rank = {i: rank for rank, i in enumerate(order)}

        # Create the main probability line
        fig = go.Figure()

        # Add the main line trace
        fig.add_trace(go.Scatter(
            x=list(range(len(sorted_probs))),
            y=sorted_probs,
            mode='lines',
            name='Probability',
            line=dict(color='blue')
        ))

        # Add vertical lines for blocked tokens with tooltips
        for token_idx in self.blocked:
            encoded_idx = self.lm._encode[token_idx]
            if encoded_idx in rank and rank[encoded_idx] < M:
                r = rank[encoded_idx]
                fig.add_trace(go.Scatter(
                    x=[r, r],
                    y=[min(sorted_probs), max(sorted_probs)],
                    mode='lines',
                    name=f'Blocked token at rank {r}',
                    line=dict(color='red', width=1),
                    hoverinfo='text',
                    hovertext=f'Blocked token index: {token_idx}<br>Rank: {r}<br>Probability: {probs[encoded_idx]:.2e}'
                ))

        # Update layout for log scale on y-axis
        fig.update_layout(
            yaxis_type="log",
            title='Canonical Token Rank Distribution',
            xaxis_title='Rank',
            yaxis_title='Probability',
            showlegend=False,
            hovermode='closest'
        )

        return fig

    def about(self):
        b = self.blocked
        a = self.allowed
        return f"""\
blocked:  {len(b) / (len(b) + len(a)) * 100:.2f}% ({len(b)}/{len(b) + len(a)})
logprob:  {self.total_blocked:.2f} ({np.exp(self.total_blocked):.3g})
examples: {self.blocked[:5]}
"""
