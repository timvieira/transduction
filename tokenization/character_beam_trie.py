#import torch
import numpy as np
from collections import defaultdict
#from numpy import logaddexp
from functools import cached_property

from arsenal import colors
from arsenal.maths import logsubexp, sample

from tokenization.basics import Item
from tokenization.trie import TokenCharacterTrie
#from tokenization.batch_trie import BatchTokenCharacterTrie
from tokenization.util import prefixes, Chart, logsumexp, flatten
from IPython.display import display


class TrieState:

    def __init__(self, lm_state, trie, node, mass, weight, parent):
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.mass = mass
        self.weight = weight
        self.parent = parent

    @property
    def key(self):
        return self.lm_state.context

    def advance(self, actions):
        s = self
        for a in actions:
            s <<= a
            if s is None: return s
        return s

    def __repr__(self):
        return f'TrieState({self.weight:.2f}, {flatten(self.key)} {colors.arrow.right} {self.partial!r})'

    def __lshift__(self, a):
        next_node = self.trie.children[self.node].get(a)
        if next_node is None:
            return
        if a is None:
            token = self.trie.leaf2word[next_node]
            next_lm_state = self.lm_state << token
            next_mass = np.log(self.trie.mass_sum(next_lm_state.logp_next.apply(np.exp)))
            return TrieState(
                lm_state = next_lm_state,
                trie = self.trie,
                mass = next_mass,
                node = self.trie.root,
                parent = (self, a),
                weight = self.weight + self.mass[next_node] - self.mass[self.node],
            )
        else:
            return TrieState(
                lm_state = self.lm_state,
                trie = self.trie,
                mass = self.mass,
                node = next_node,
                parent = (self, a),
                weight = self.weight + self.mass[next_node] - self.mass[self.node],
            )

    @classmethod
    def initial(cls, lm, trie):
        lm_state = lm.initial()
        return cls(
            trie = trie,
            node = trie.root,
            lm_state = lm_state,
            mass = np.log(trie.mass_sum(lm_state.logp_next.apply(np.exp))),
            weight = 0,
            parent = (None, None),
        )

    def actions(self):
        return self.trie.children[self.node]

    @cached_property
    def logp_next(self):
        logZ = self.mass[self.node]
        return Chart(0, {a: self.mass[i] - logZ for a, i in self.actions().items()})

    def has_EOT(self):
        return None in self.actions()

    def items(self):
        "Method for inspecting a bundle"
        trie = self.trie
        mass = self.mass
        agenda = [self.node]
        while agenda:
            i = agenda.pop()
            for symbol, j in trie.children[i].items():
                if symbol is None:
                    yield (j, mass[j])
                else:
                    agenda.append(j)

    def unbundle(self):
        node = self.node
        trie = self.trie
        mass = self.mass
        for leaf, _ in self.items():
            yield Item(
                ps=self.weight + mass[leaf] - mass[node],
                ys=(self.key, trie.leaf2word[leaf]),
                offset=None,
                parent=None,
            )

    @property
    def partial(self):
        """Return the prefix (string) of the partially matched last token.

        Note: An equivalent way to think about the return value is as the
        string-valued name of the state in the trie (the implementation uses an
        integer name for efficiency).

        """
        return self.trie.node2prefix[self.node]

    def about_html(self, *args, **kwargs):
        from tokenization.viz import bundle_about_html
        return bundle_about_html(self, *args, **kwargs)

    def _repr_html_(self):
        return self.about_html().data

    def __lt__(self, other):
        return self.weight < other.weight


class CharacterBeamState:

    def __init__(self, alg, states):
        self.alg = alg
        self.states = states

    def __iter__(self):
        return iter(self.states)

    @classmethod
    def initial(cls, alg):
        return cls(alg, [alg.trie_init])

    def __len__(self):
        return len(self.states)

    def __lshift__(self, a):
        return CharacterBeamState(self.alg, [bundle << a for bundle in self.extend if a in bundle.actions()])

    @cached_property
    def _logp_next(self):
        A = self.actions()
        Z = logsumexp([bs.weight for _, bs in A.items()])
        return Chart(-np.inf, {k: bs.weight - Z for k, bs in A.items()})

    @cached_property
    def extend(self):
        # Some of the next state require an invisible `None` action that extends
        # the bundle by a next token -- this is an expensive action to take as
        # it requires evaluting the underlying LM.  It is also the case that it
        # benefits for batching.  In the code below, we determine which bundles
        # seem worth extending in this way by placing them in a buffer so that
        # they can be evalated in a batch.
        batch = []
        for bundle in self:
            if bundle.has_EOT():
                if (
                    self.alg.extend_threshold is None or
                    np.exp(bundle.weight + bundle.logp_next[None] - self.weight) >= self.alg.extend_threshold
                ):
                    batch.append(bundle)

        return CharacterBeamState(self.alg, self.states + self.extend_batch(batch))

    def actions(self):
        # group next states by the actions that take us to them
        A = defaultdict(list)
        for bundle in self.extend:
            for a in bundle.logp_next:
                if a is not None:
                    A[a].append(bundle << a)
        return {a: CharacterBeamState(self.alg, next_states) for a, next_states in A.items()}

    @cached_property
    def weight(self):
        return logsumexp([b.weight for b in self])

    # TODO: implement efficiently, use batch trie and batch LM state computation
    def extend_batch(self, batch):
        return [bundle << None for bundle in batch]

    def prune(self):
        S = sorted([x for x in self if x.weight > -np.inf], key=lambda bundle: -bundle.weight)

        if self.alg.relative_score_threshold is not None:
            if self.alg.verbosity > 1:
                print(f'we have {len(S)}; relative scores:', [np.exp(S[0].weight - S[k].weight) for k in range(len(S))])
            S = [
                bundle for bundle in S
                if np.exp(S[0].weight - bundle.weight) <= self.alg.relative_score_threshold
                or (self.alg.eot_immunity and not bundle.has_EOT())   # Bundles with no EOT are immune to avoid dead ends
            ]
            if self.alg.verbosity > 1:
                print(f'now, we have {len(S)}...')

        if not self.alg.eot_immunity:
            S = S[:self.alg.K]
        else:

            if self.alg.K is not None:
                # We make Bundles that do not have an EOT immune from pruning to avoid dead ends.
                tmp = []
                count = 0
                for bundle in S:
                    if bundle.has_EOT():  # if the bundle has EOT
                        count += 1
                    tmp.append(bundle)
                    if count >= self.alg.K:
                        break
                #assert len(tmp) <= 2 * self.alg.K, [len(tmp), self.alg.K]
                S = tmp

        return CharacterBeamState(self.alg, S)

    def about_html(self, *args, **kwargs):
        from tokenization.viz import show_beam_state
        return show_beam_state(list(self), *args, **kwargs)

    def _repr_html_(self):
        return self.about_html().data


class CharacterBeam:

    def __init__(self, llm, K, eos='▪', relative_score_threshold=None, eot_immunity=False,
                 extend_threshold=None, verbosity=0):

        self.llm = llm
        self.eos = eos
        self.V = {x for y in self.llm.V for x in y}
        # The `trie` created below is a data structure that tells us how to
        # build and navigate the `mass` vector.
        self.trie = TokenCharacterTrie(words = llm._decode,
                                       encode = llm._encode,
                                       old_eos = llm.eos,
                                       new_eos = self.eos)

        self.trie_init = TrieState.initial(self.llm, self.trie)

        self.K = K
        self.relative_score_threshold = relative_score_threshold
        self.extend_threshold = extend_threshold
        self.eot_immunity = eot_immunity
        self.verbosity = verbosity

        self._beam_cache = {}
        self._candidate_cache = {}

    def __matmul__(self, fst, *args, **kwargs):
        return self.transduce(fst, *args, **kwargs)

    def transduce(self, fst, *args, **kwargs):
        from tokenization.transduction import TransducedLM
        return TransducedLM(self, fst, *args, **kwargs)

    def unbundle(self, xs):
        for bundle in self.beam(xs):
            for item in bundle.unbundle():
                yield item

    def encodings(self, xs):
        for bundle in self.beam(xs):
            for item in bundle.unbundle():
                # TODO: can we do this more efficienly via EOT in the trie?
                if xs == b''.join(flatten(item.ys)):
                    yield item

    def logprefix(self, context):
        return logsumexp([bundle.weight for bundle in self.candidates(context)])

    def logp_next_seq(self, context, extension):
        return self.logprefix(context + extension) - self.logprefix(context)

    def surprisals(self, context):
        s = []
        N = len(context)
#        for n in iterview(range(N+1), transient=True):
        for n in range(N):
            logp = self.logp_next(context[:n])
            s.append(-logp[context[n]])
        # And, finally, the surprisal of EOS
        logp = self.logp_next(context)
        s.append(-logp[self.eos])
        return s
    
    def beam(self, context):
        y = self._beam_cache.get(context)
        if y is not None:
            return y
        y = self._beam(context)
        self._beam_cache[context] = y
        return y

    def _beam(self, context):
        if self.verbosity > 0: print(context)
        return self.candidates(context).prune()

    def candidates(self, context):
        y = self._candidate_cache.get(context)
        if y is not None:
            return y
        y = self._candidates(context)
        self._candidate_cache[context] = y
        return y

    def _candidates(self, context):
        if len(context) == 0:
            return CharacterBeamState.initial(self)
        else:
            return self.beam(context[:-1]) << context[-1]

    def logp_next(self, context):
        return self.candidates(context)._logp_next

    #___________________________________________________________________________
    # Visualization

    def describe_pruning(self, final_context):
        from tokenization.util import display_table
        for context in prefixes(final_context):
            candidates = self.candidates(context)
            A = logsumexp([b.weight for b in candidates])   # unpruned total weight
            print(colors.light.cyan % 'candidates:', repr(context))
            # adaptive beam selection based on the residual left by pruning
            B = -np.inf
            table = []
            for rank, b in enumerate(candidates, start=1):
                B = logsumexp([B, b.weight])
                gap = np.exp(float(logsubexp(A, B))) if A >= B else 0
                pct = np.exp(B - A) * 100
                table.append([
                    f'{rank:3d}',
                    f'{gap:.2g}',
                    f'{pct:6.2f}%',
                    f'{b.weight:.2f}',
                    (rank <= self.K),
                    [flatten(b.key), b.partial]
                ])
                #print(colors.cyan % '   └─', A, B)
            display_table(table, headings=['rank', 'gap', 'pct', 'weight', 'on beam', 'tokenization'])

    def show_html(self, *args, **kwargs):
        from tokenization.viz import character_beam_show_html
        return character_beam_show_html(self, *args, **kwargs)

    # TODO: use states
    def greedy(self, prompt, steps, verbose=False):
        """
        Generate character-by-character starting from `prompt` using LLM with
        the approximate conditional distribution.
        """
        context = prompt
        for _ in range(steps):
            p = self.logp_next(context)
            if verbose:
                print(repr(context), p.top(5))
            x = p.argmax()
            if x == self.eos:
                break
            context += x
        return context


class CharacterBeamStochastic(CharacterBeam):
    """
    Stochastic beam search estimator of the character-level distribution.

    Provides:
     * unbiased estimates of prefix probabilities
     * consistent estimates of conditional probabilities

    Notes:
     * bundle weights are random.
     * internal memoization causes sampls to be dependent. For a fresh sample, clear_cache.

    """

    def beam(self, context):

        K = self.K
        candidates = self.candidates(context)

        if len(candidates) <= K:
            return candidates

        # We sample a subset from candidates of size K as follows; Take the top
        # K-1 elements deterministically Sample the Kth element
        # non-deterministically with probability proportional to its original
        # probability, but renormalized by the remaining mass.
        weights = np.array([b.weight for b in candidates])
        weights = weights - logsumexp(weights)

        ranking = sorted(range(len(candidates)), key=lambda i: -weights[i])
        selected = [candidates[i] for i in ranking[:K-1]]

        wildcards = ranking[K-1:]
        w = np.array([weights[i] for i in wildcards])
        R = logsumexp(w)
        i = sample(np.exp(w - R))
        wildcard = candidates[wildcards[i]]

        # The weight of each deterministically chosen element is unchanged.
        # However, the weight of the nondeterministically chosen wildcard is
        # adjusted to account for the weight the elements that it was randomly
        # selected among.
        wildcard.weight = wildcard.weight + R

        selected.append(wildcard)

        return selected


class TokenHealingHeuristic:
    """
    This pruning heuristic is inspired by token healing.  Note that it is not
    token healing, as it is a generate method for generating character strings.
    """
    def __init__(self, lm, adaptive=False, **kwargs):
        self.lm = lm
        self.C = CharacterBeam(lm, K=np.inf, **kwargs)
        self.trie = self.C.trie
        self.lookup = {y: x for x, y in self.trie.node2prefix.items() if x not in self.trie.leaf2word}
        self.adaptive = adaptive

    def logp_next(self, prompt):
        return CharacterBeamState(alg=self.C, states=[self.healing(prompt)]).extend._logp_next

    def healing(self, prompt):
        if self.adaptive:
            return self.adaptive_healing_trie_state(prompt)
        else:
            return self.basic_healing_trie_state(prompt)

    # TODO: the implementation would be nicer if we did this search directly in
    # the `_healing_trie_state` method.
    def adaptive_healing_trie_state(self, prompt, verbosity=0):
        """
        An adaptive version of the token-healing heuristic; Returns the `TrieState`
        corresponding to the highest-scoring `(context, partial_token)` for the
        tokenization of `prompt`.
        """

        s = self.basic_healing_trie_state(prompt)

        ss = s
        suffix = []

        best = s

        while True:
            a = ss.parent[1]
            if a is None:
                if verbosity > 0: print('state:', ss, f'{bytes(suffix) = }', 'prev action:', a if a is None else bytes([a]))

                key = self.trie.node2prefix[ss.node] + bytes(suffix)
                if key in self.lookup:
                    if verbosity > 0:
                        print(colors.light.cyan % 'AVAILABLE:', self.trie.node2prefix[ss.node], bytes(suffix), '=', key)

                    backup_candidate = ss.advance(key)
                    if verbosity > 1:
                        display(backup_candidate)

                    best = max(best, backup_candidate)

                else:
                    if verbosity > 1:
                        print(colors.light.yellow % 'UNAVAILABLE:', self.trie.node2prefix[ss.node], bytes(suffix), '=', key)

            #if ss.parent is None: break
            (ss, a) = ss.parent
            if a is not None:
                suffix = [a] + suffix
            if ss is None: break

        return best

    def basic_healing_trie_state(self, prompt):
        if len(prompt) == 0:
            token_prefix, last_token = (), prompt
        else:
            token_prefix, last_token = self.lm.encode_prompt(prompt.decode())   # good luck with invalid utf-8

        # Below is a more detail version that does not use the abstractions of the library
        #lm_state = self.lm.initial()
        #for y in flatten(token_prefix):
        #    lm_state <<= y
        #mass = np.log(self.trie.mass_sum(lm_state.logp_next.apply(np.exp)))
        ## option 1 (faster): jump straight to the relevant trie state
        ##node = self.lookup[last_token]
        ##s = TrieState(lm_state=lm_state, trie=self.trie, node=node, mass=mass, weight=lm_state.logp + mass[node], parent=None)
        ## option 2 (slightly slower): advance trie state step-by-step
        #s = TrieState(lm_state=lm_state, trie=self.trie, node=self.trie.root, mass=mass, weight=lm_state.logp, parent=None)
        #for x in last_token:
        #    s <<= x

        s = TrieState.initial(self.lm, self.trie)
        for token in flatten(token_prefix):
            for x in token:
                s <<= x
            s <<= None
        for x in last_token:
            s <<= x

        return s


#class OldTokenHealingHeuristic:
#    def __init__(self, lm, eos='▪'):
#        self.lm = lm
#        self.eos = eos
#        self.trie = TokenCharacterTrie(words = lm._decode,
#                                       encode = lm._encode,
#                                       old_eos = lm.eos,
#                                       new_eos = self.eos)
#        self.lookup = {y:x for x, y in self.trie.node2prefix.items()
#                       if x not in self.trie.leaf2word}
#
#    def logp_next(self, text, verbosity=0):
#        # TODO: handle the empty prompt
#        if len(text) == 0:
#            context, partial = (), text
#        else:
#            context, partial = self.lm.encode_prompt(text.decode())   # good luck with invalid utf-8
#
#        logp_next = self.lm.logp_next(context)
#        if verbosity > 0: print(flatten(context), colors.light.yellow % partial, logp_next.top(5))
#        mass = np.log(self.trie.mass_sum(logp_next.apply(np.exp)))
#        node = self.lookup[partial]
#
#        # We might also be at the end of a token, so we need to evaluate
#        # the LM assuming under with context = tokenized, too!
#        #
#        # For example, `Hello, world` should predict punctionation rather than a continuation of `world`.
#
#        Q = Chart(-np.inf, {
##FOO            z: mass[c] #- mass[node]
#            z: mass[c] - mass[node]
#            for z, c in self.trie.children[node].items()
#            if z is not None
#        })
#
#        eot_node = self.trie.children[node].get(None)
#        if eot_node is not None:
#            mass_eot = mass[eot_node]
#
#            logp_next = self.lm.logp_next((context, partial))
#            if verbosity > 0: print('extend=', logp_next.top(5))
#
#            mass = np.log(self.trie.mass_sum(logp_next.apply(np.exp)))
#            node = self.trie.root
#
#            assert abs(mass[node]) < 1e-5, abs(mass[node])
#
#            for z, c in self.trie.children[node].items():
#                assert z is not None
#                Q[z] = np.logaddexp(Q[z], mass_eot + mass[c] - mass[node])
##FOO:                Q[z] = np.logaddexp(Q[z], mass_eot + mass[c]) #- mass[node])
##                Q[z] = np.logaddexp(Q[z], mass[c] - mass[node])
#
#        logZ = logsumexp(list(Q.values()))
#        return Chart(-np.inf, {
#            k: (value - logZ)
#            for k, value in Q.items()
#        })
