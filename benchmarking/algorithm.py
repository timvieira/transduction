from transduction.lazy_recursive import LazyRecursive
from transduction.eager_nonrecursive import LazyPrecoverNFA

import numpy as np
from arsenal.datastructures import LocatorMaxHeap
from dataclasses import dataclass


@dataclass(frozen=False, eq=True, unsafe_hash=True)
class Item:
    "Items used in search queue"
    weight: float
    state: object
    source: object
    def __repr__(self):
        return f'Item({self.weight:.3f}, {self.state}, {repr(self.source)})'
    

def prioritized_enumeration(lm, fst, target, max_steps, EOS, threshold=0.001, trim=False):

    precover = LazyRecursive(fst, empty_target=(), empty_source=(), extend=lambda x, y: x + (y,))(target)
    print(" | Precover instantiated")
    fsa = precover.quotient
    Q = precover.quotient.stop
    R = precover.remainder.stop
    print("| Precover built")
    fsa.stop |= R
    print("| Remainder removed from acceptance")
    if trim:
        print("trimming")
        fsa = fsa.trim()
        print("done trimming")
    
    if not (set(target) <= fst.B):
        print(f'[warn] OOVs: {set(target) - fst.B}')

    remainder = []
    quotient = []
    queue = LocatorMaxHeap()
    
    for q in fsa.start:
        queue[Item(weight = 0, state = q, source = '')] = 0
    t = 0
    while queue:
        t += 1
        if t > max_steps:
            print('stopped early')
            break
        (item, _) = queue.pop()
        lm_logp_next = lm.logp_next(item.source)

        if item.state in Q:
            quotient.append(item)
            continue
        if item.state in R:
            # add the eos probability here
            remainder.append(Item(
                weight = item.weight + lm_logp_next[EOS],
                state = item.state,
                source = item.source,
            ))
            remainder.append(item)
        for x, next_state in fsa.arcs(item.state):
            next_weight = float(item.weight + lm_logp_next[x])   # use LM state here
            if next_weight == -np.inf: continue
            next_item = Item(
                weight = next_weight,
                state = next_state,
                source = item.source + (x, ),
            )
            queue[next_item] = next_weight

    
    return (quotient, remainder)
