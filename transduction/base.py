from abc import ABC, abstractmethod
from transduction.fst import EPSILON
from arsenal import colors
from collections import deque


class DecompositionResult:
    r"""Base class for quotient--remainder decomposition results.

    $$\mathcal{P}(\boldsymbol{y}) = \mathcal{Q}(\boldsymbol{y}) \mathcal{X}^* \sqcup \mathcal{R}(\boldsymbol{y})$$

    Can be used directly as a simple pair::

        result = DecompositionResult(quotient, remainder)
        result.quotient
        result.remainder

    Or subclassed with lazy properties (e.g., ``Precover``, ``NonrecursiveDFADecomp``).

    The quotient and remainder may be sets of strings or FSAs depending on the
    producer.
    """

    def __init__(self, quotient, remainder):
        self.quotient = quotient
        self.remainder = remainder

    def decompose_next(self):
        """Decompose for every next target symbol, returning a dict {y: DecompositionResult}."""
        target_alphabet = self.fst.B - {EPSILON}
        return {y: type(self)(self.fst, self.target + y) for y in target_alphabet}


class DecompositionFunction(ABC):
    """Interface for reusable decomposition algorithms.

    Constructed once with an FST, then called with different target strings.
    Returns a DecompositionResult with set-valued quotient and remainder.

    Usage::

        alg = SomeAlgorithm(fst)
        result = alg(target)     # -> DecompositionResult
        result.quotient          # set of source strings
        result.remainder         # set of source strings

    Implementations: AbstractAlgorithm
    """
    @abstractmethod
    def __call__(self, target) -> 'DecompositionResult': ...


class IncrementalDecomposition(DecompositionResult):
    """Interface for incremental (symbol-by-symbol) decomposition.

    Extends DecompositionResult with the ``>>`` operator to advance the
    target by one symbol, reusing computation from the previous step.

    Usage::

        state = SomeDecomp(fst, '')
        state = state >> 'a'     # extend target to 'a'
        state = state >> 'b'     # extend target to 'ab'
        state.quotient           # FSA for target 'ab'
        state.remainder          # FSA for target 'ab'

    Implementations: PeekabooState, IncrementalDFADecomp, LazyIncremental
    """
    @abstractmethod
    def __rshift__(self, y) -> 'IncrementalDecomposition': ...

    def decompose_next(self):
        """Decompose for every next target symbol, returning a dict {y: IncrementalDecomposition}."""
        target_alphabet = self.fst.B - {EPSILON}
        return {y: self >> y for y in target_alphabet}

    def __call__(self, ys):
        """Advance state by a sequence of target symbols. Returns the final state."""
        s = self
        for y in ys:
            s = s >> y
        return s


class AbstractAlgorithm(DecompositionFunction):

    def __init__(self, fst, empty_source = '', extend = lambda x,y: x + y, max_steps=float('inf')):
        self.fst = fst
        self.empty_source = empty_source
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.extend = extend
        self.max_steps = max_steps

    def __call__(self, target):
        precover = DecompositionResult(set(), set())
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
