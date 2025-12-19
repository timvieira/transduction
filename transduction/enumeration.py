from transduction import FST, FSA, EPSILON, PrecoverDecomp, examples, Precover
from transduction.util import display_table

import numpy as np
from arsenal.datastructures import LocatorMaxHeap
from dataclasses import dataclass

from arsenal import colors
from arsenal.maths import sample
from tokenization.util import logsumexp, Chart


@dataclass(frozen=False, eq=True, unsafe_hash=True)
class Item:
    "Items used in search queue"
    weight: float
    state: object
    source: object
    def __repr__(self):
        return f'Item({self.weight:.3f}, {self.state}, {repr(self.source)})'


class prioritized_enumeration:
    def __init__(self, lm, fst, target, max_steps, empty='', extend=(lambda x,y: x+y)):

        precover = Precover(fst, target)
        dfa = precover.quotient
        Q = precover.quotient.stop
        R = precover.remainder.stop
        dfa.stop |= R
        self.dfa = dfa.trim()

        if not (set(target) <= fst.B):
            print(f'[warn] OOVs: {set(target) - fst.B}')

        self.remainder_terms = []
        self.quotient_terms = []
        self.queue = LocatorMaxHeap()

        for q in self.dfa.start:
            self.queue[Item(weight = 0, state = q, source = lm)] = 0

        self.Q = Q
        self.R = R
        self.precover = precover
        self.dfa = dfa

        self.run(max_steps)

    def run(self, max_steps):
        t = 0
        while self.queue:
            t += 1
            if t > max_steps:
                print(colors.light.red % 'stopped early')
                break
            (item, _) = self.queue.pop()
            print('pop:', item)
            lm_logp_next = item.source.logp_next
            if item.state in self.Q:
                self.quotient_terms.append(item)
                continue
            if item.state in self.R:
                # add the eos probability here
                self.remainder_terms.append(Item(
                    weight = item.weight + lm_logp_next[EOS],
                    state = item.state,
                    source = item.source,   # << EOS?
                ))
                remainder.append(item)
            for x, next_state in self.dfa.arcs(item.state):
                next_weight = float(item.weight + lm_logp_next[x])   # use LM state here
                if next_weight == -np.inf: continue
                next_item = Item(
                    weight = next_weight,
                    state = next_state,
                    source = item.source << x.bytes,
                )
                #print('push:', next_item)
                self.queue[next_item] = next_weight


class importance_sampling:

    def __init__(self, lm, fst, target):

        precover = Precover(fst, target)
        dfa = precover.quotient
        Q = precover.quotient.stop
        R = precover.remainder.stop
        dfa.stop |= R
        self.dfa = dfa.trim()

        if not (set(target) <= fst.B):
            print(f'[warn] OOVs: {set(target) - fst.B}')

        self.Q = Q
        self.R = R
        self.precover = precover
        self.dfa = dfa
        self.lm = lm

    def sample(self, max_length=np.inf):
        EOS = self.lm.lm.eos

        t = 0
        for i in self.dfa.lazy().start():
            item = Item(weight = 0, state = i, source = self.lm)
        while True:
            t += 1
            if t > max_length:
                print(colors.light.red % 'stopped early')
                break

            lm_logp_next = item.source.logp_next
            if item.state in self.Q:
                return item

            q = {}
            T = {}
            if item.state in self.R:
                q[EOS] = lm_logp_next[EOS]
            for x, next_state in self.dfa.arcs(item.state):
                q[x] = lm_logp_next[x]
                T[x] = next_state

            keys = list(q.keys())
            vals = np.array(list(q.values()))
            Z = logsumexp(vals)
            vals = np.exp(vals - Z)
            x_t = keys[sample(vals)]

            if x_t == EOS:
                return item

            item = Item(
                weight = item.weight + Z,
                state = T[x_t],
                source = item.source << x_t.bytes,
            )
