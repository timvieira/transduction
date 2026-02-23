from time import time
from tokenization.basics import Beam, Item
from tokenization.util import flatten, covers
from arsenal.datastructures import LocatorMaxHeap


class BruteForceModel:

    def __init__(self, llm):
        self.llm = llm

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
                ps = item.ps + self.llm.logp_next(item.ys)[self.llm.eos],
                ys = item.ys,
                offset = item.offset,
                parent = item.parent,
            )
            for item in self.covering(context)
            if item.offset == N
        ])

    def covering(self, context):

        # Prints progress reports when things are taking long...
        def update():
            print(cnt, f'elapsed: {t - start:.2f} log-rel-max: {B[0].ps - threshold:.3g}')
            print(f'   {item.ps:.2f}, {flatten(item.ys)}', )
            update.last_t = t

        cnt = 0
        start = update.last_t = time()
        B = []
        threshold = 0
        for item in self._lazy_covering(context):
            B.append(item)
            #if len(B) == 1:
            #    print('best:', item)
            assert item.ps <= threshold
            threshold = item.ps
            cnt += 1
            t = time()
            if t - update.last_t > 5:
                update()
        #update()
        #print('done')
        return Beam(B)

    def _lazy_covering(self, context):
        """
        Lazily enumerate the covering of `context` in sorted order if prefix
        probability.
        """
        llm = self.llm
        Q = LocatorMaxHeap()
        Q[Item(ps=0, ys=(), offset=0, parent=None)] = 0

        prev_best = 0
        while Q:

            (item, _) = Q.pop()

            assert item.ps <= prev_best
            prev_best = item.ps

            if covers(context, item.ys):
                yield item
                continue

            logp = llm.logp_next(item.ys)

            remainder = context[item.offset:]
            for y, logpy in logp.items():

                # We use the following equivalences to improve efficiency
                # ysy = (item.ys, y)
                # xsx = ''.join(flatten(ysy))
                # assert context.startswith(xsx) == remainder.startswith(y)
                # assert xsx.startswith(context) == y.startswith(remainder)
                # assert xsx[item.offset:] == y
                # assert (context.startswith(xsx) or xsx.startswith(context))
                #         == (remainder.startswith(y) or y.startswith(remainder))
                if y == llm.eos:
                    continue

                if remainder.startswith(y) or y.startswith(remainder):
                    new_item = Item(
                        ps = item.ps + logpy,
                        ys = (item.ys, y),
                        offset = item.offset + len(y),
                        parent = item,
                    )
                    Q[new_item] = new_item.ps
