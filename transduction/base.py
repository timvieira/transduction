from transduction.fst import EPSILON
from arsenal import colors
from collections import deque


class PrecoverDecomp:
    """
    This class represents the precover of some target string $\boldsymbol{y} \in \mathcal{Y}^*$ as
    a quotient and remainder set:

    $$\mathcal{P}(\boldsymbol{y}) = \mathcal{Q}(\boldsymbol{y}) \mathcal{X}^* \sqcup \mathcal{R}(\boldsymbol{y})$$

    """

    def __init__(self, quotient, remainder):
        self.quotient = quotient
        self.remainder = remainder

    def __repr__(self):
        return repr((self.quotient, self.remainder))

    def __iter__(self):
        return iter((self.quotient, self.remainder))

    def __eq__(self, other):
        return tuple(other) == tuple(self)


class AbstractAlgorithm:

    def __init__(self, fst, empty_source = '', extend = lambda x,y: x + y, max_steps=float('inf')):
        self.fst = fst
        self.empty_source = empty_source
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.extend = extend
        self.max_steps = max_steps

    def __call__(self, target):
        precover = PrecoverDecomp(set(), set())
        worklist = deque()
        for xs in self.initialize(target):
            worklist.append(xs)
        t = 0
        while worklist:
            xs = worklist.popleft()
            t += 1
            if t > self.max_steps:
                print(colors.light.red % 'stopped early')
                break
            if self.continuity(xs, target):
                precover.quotient.add(xs)
                continue
            if self.discontinuity(xs, target):
                precover.remainder.add(xs)
            for next_xs in self.candidates(xs, target):
                worklist.append(next_xs)
        return precover

    def initialize(self, target):
        raise NotImplementedError()

    def candidates(self, xs, target):
        raise NotImplementedError()

    def discontinuity(self, xs, target):
        raise NotImplementedError()

    def continuity(self, xs, target):
        raise NotImplementedError()
