from transduction.eager_nonrecursive import Precover

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

    precover = Precover(fst, target)
    print(" | Precover instantiated")
    Q = precover.quotient
    R = precover.remainder
    Q_final = Q.stop  # Final states of quotient (universal states)
    R_final = R.stop  # Final states of remainder
    # Use quotient FSA for search; it shares structure with remainder
    fsa = Q
    # Note: we track Q_final and R_final separately instead of modifying fsa.stop
    print("| Precover built")
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

        if item.state in Q_final:
            quotient.append(item)
            continue
        if item.state in R_final:
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