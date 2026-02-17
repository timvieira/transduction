"""
Precover decomposition: core concepts and abstract interfaces.

The Central Problem
-------------------
Given an FST ``f : X* -> Y*`` and a target prefix **y**, we want to know which
source strings produce output that begins with **y**.  This set is called the
**precover**:

    P(y) = { x in X* : f(x) starts with y }

The precover splits into two disjoint parts:

    P(y) = Q(y) · X*  ⊔  R(y)

- **Quotient** Q(y): source strings that have produced **y** so far and can
  still continue (the FST has not yet reached a final state, or has reached
  one but could keep going).  Every extension x·a for a in X is also in P(y).

- **Remainder** R(y): source strings that have produced **y** and terminated
  (the FST is in a final state and no further output is possible).  These are
  complete — no extension of x is in P(y).

Why This Matters
~~~~~~~~~~~~~~~~
To compute P(next_symbol | y) for a language model composed with an FST, we
need Q(y·z) and R(y·z) for each candidate next symbol z.  The probability is:

    P(z | y) ∝ Σ_{x ∈ Q(y·z)} P_LM(x) + Σ_{x ∈ R(y·z)} P_LM(x · EOS)

The quotient strings contribute ongoing probability mass; the remainder strings
contribute their EOS probability.  All decomposition algorithms in this library
compute Q and R — they differ in how efficiently they do so (truncation
strategy, incrementality, batching, language).

Terminology
~~~~~~~~~~~
- **Precover NFA**: An NFA whose accepted language is P(y).  Each NFA state
  pairs an FST state with a target-output buffer tracking how much of **y**
  has been produced.  See ``precover_nfa.py`` for implementations.

- **Universality**: A DFA state is *universal* for symbol z if every source
  string reachable from it belongs to Q(y·z).  Detecting universality early
  lets us stop expanding the DFA — the key insight behind the Peekaboo
  algorithm.

- **Truncation**: Bounding the target-output buffer to prevent infinite state
  spaces.  Algorithms that truncate (Peekaboo, DFA decomp, Rust backends)
  terminate on all FSTs; those that don't (Lazy variants) may diverge on FSTs
  with infinite quotients.

- **Dirty state**: Incremental algorithms that persist DFA states across
  ``>>`` steps.  "Dirty" means the cached DFA may contain states from a
  previous target prefix that need re-expansion.
"""

from abc import ABC, abstractmethod
from transduction.fst import EPSILON
from transduction.util import colors
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

    def __iter__(self):
        return iter((self.quotient, self.remainder))

    def __repr__(self):
        return f'DecompositionResult({self.quotient!r}, {self.remainder!r})'

    def __rshift__(self, y):
        """Advance the target by one symbol, returning a new decomposition."""
        return type(self)(self.fst, self.target + (y,))

    def decompose_next(self):
        """Decompose for every next target symbol, returning a dict {y: DecompositionResult}."""
        target_alphabet = self.fst.B - {EPSILON}
        return {y: self >> y for y in target_alphabet}


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

    Extends DecompositionResult with an optimized ``>>`` operator that
    reuses computation from the previous step rather than decomposing
    from scratch.

    Usage::

        state = SomeDecomp(fst, '')
        state = state >> 'a'     # extend target to 'a'
        state = state >> 'b'     # extend target to 'ab'
        state.quotient           # FSA for target 'ab'
        state.remainder          # FSA for target 'ab'

    Implementations: PeekabooState, TruncatedIncrementalDFADecomp, LazyIncremental
    """
    @abstractmethod
    def __rshift__(self, y) -> 'IncrementalDecomposition': ...

    def __call__(self, ys):
        """Advance state by a sequence of target symbols. Returns the final state."""
        s = self
        for y in ys:
            s = s >> y
        return s


class AbstractAlgorithm(DecompositionFunction):

    def __init__(self, fst, empty_source='', extend=lambda x, y: x + y, max_steps=float('inf')):
        """
        Args:
            fst: The FST to decompose.
            empty_source: Identity element for source strings (default ``''``).
                Used as the seed for BFS over source language strings.
            extend: Binary function to append a symbol to a string
                (default ``x + y``, i.e., Python string concatenation).
                Override both ``empty_source`` and ``extend`` to use a
                non-string sequence type (e.g., tuples).
            max_steps: Maximum BFS steps before stopping (default unlimited).
        """
        self.fst = fst
        self.empty_source = empty_source
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.extend = extend
        self.max_steps = max_steps

    def __call__(self, target):
        """Compute the Q/R decomposition for ``target`` via BFS over source strings."""
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

    @abstractmethod
    def initialize(self, target):
        """Return the initial seed strings for the BFS worklist."""
        ...

    @abstractmethod
    def candidates(self, xs, target):
        """Return source-string extensions of ``xs`` to add to the worklist."""
        ...

    @abstractmethod
    def discontinuity(self, xs, target):
        """Return True if ``xs`` belongs in the remainder R(target)."""
        ...

    @abstractmethod
    def continuity(self, xs, target):
        """Return True if ``xs`` belongs in the quotient Q(target)."""
        ...
