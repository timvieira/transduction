from time import time
from tokenization.basics import Beam, Item
from tokenization.util import flatten, covers
from arsenal.datastructures import LocatorMaxHeap


class BruteForceModel:

    def __init__(self, lm):
        self.lm = lm

    def logprefix(self, context):
        return self.covering(context).total

    def logp(self, context):
        return self.encodings(context).total

    def encodings(self, context):
        "Return the set of token-string encodings of `context`"
        # TODO: probably faster to get the encodings directly rather than
        # filtering like we do below.
        N = len(context)
        return Beam([
            Item(
                ps = state.logp + state.logp_next[self.lm.eos],
                ys = state.context,
                offset = None,
                parent = None,
            )
            for state in self._lazy_covering(context)
            if N == sum(len(y) for y in flatten(state.context))
        ])

    def covering(self, context):

        # Prints progress reports when things are taking long...
        def update():
            print(cnt, f'elapsed: {t - start:.2f} log-rel-max: {B[0].logp - threshold:.3g}')
            print(f'   {item.logp:.2f}, {flatten(item.context)}', )
            update.last_t = t
            print('  lm calls:', self.lm._calls)

        cnt = 0
        start = update.last_t = time()
        B = []
        threshold = 0
        for item in self._lazy_covering(context):
            B.append(item)
            assert item.logp <= threshold
            threshold = item.logp
            cnt += 1
            t = time()
            if t - update.last_t > 5:
                update()
        return Beam([
            Item(
                ps = state.logp,
                ys = state.context,
                offset = None,
                parent = None,
            )
            for state in B
        ])

    def _lazy_covering(self, context):
        """
        Lazily enumerate the covering of `context` in sorted order if prefix
        probability.
        """
        lm = self.lm
        Q = LocatorMaxHeap()
        state = lm.initial()
        Q[state] = 0
        offset = {state: 0}

        prev_best = 0
        while Q:

            (state, _) = Q.pop()

            assert state.logp <= prev_best
            prev_best = state.logp

            if covers(context, state.context):
                yield state
                continue

            logp = state.logp_next

            remainder = context[offset[state]:]
            for y, logpy in logp.items():

                # We use the following equivalences to improve efficiency
                # ysy = (item.context, y)
                # xsx = ''.join(flatten(ysy))
                # assert context.startswith(xsx) == remainder.startswith(y)
                # assert xsx.startswith(context) == y.startswith(remainder)
                # assert xsx[item.offset:] == y
                # assert (context.startswith(xsx) or xsx.startswith(context))
                #         == (remainder.startswith(y) or y.startswith(remainder))
                if y == lm.eos:
                    continue

                if remainder.startswith(y) or y.startswith(remainder):
                    next_state = state << y
                    Q[next_state] = next_state.logp
                    offset[next_state] = offset[state] + len(y)
