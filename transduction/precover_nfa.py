"""
Collection of Precover NFA implementations for FST decomposition.

Each class implements a lazy NFA whose language is (a subset of) the precover
of a target string under an FST.  They differ in buffer representation,
truncation strategy, and intended use case.

Summary
-------
- PrecoverNFA: Reference push implementation, no truncation.
  Used by: Precover, EagerNonrecursive, LazyNonrecursive, NonrecursiveDFADecomp,
  enumeration.
- TruncationMarkerPrecoverNFA: Like PrecoverNFA but tracks a truncation bit.
  Used by: Precover (impl='push-truncated').
- PopPrecoverNFA: Pop/suffix-oriented variant — buffer shrinks from the target.
  Used by: Precover (impl='pop'). Experimental.
- TargetSideBuffer: Simplest unbounded buffer accumulation; needs Relevance wrapper.
- Relevance: Filter wrapper that prunes irrelevant buffer states (wraps TargetSideBuffer).
- PeekabooLookaheadNFA: K-lookahead truncation with truncation bit.
  Used by: PeekabooState (peekaboo_incremental).
- PeekabooFixedNFA: Fixed N+1 truncation, no truncation bit.
  Used by: Peekaboo, PeekabooStrings (peekaboo_nonrecursive).
"""

from transduction.lazy import Lazy, EPSILON


class PrecoverNFA(Lazy):
    r"""
    Reference precover NFA implementation (push, no truncation).

    ``PrecoverNFA(f, target)`` implements the precover for the string ``target``
    in the FST ``f`` as a lazy, nondeterministic finite-state automaton.
    Mathematically, the precover is given by:

    $$
    \mathrm{proj}_{\mathcal{X}}\Big( \texttt{f} \circ \boldsymbol{y}\mathcal{Y}^* \Big)
    $$

    State format: ``(i, ys)`` where ``i`` is an FST state and ``ys`` is the
    target output buffer (string, grows via push).

    Used by: Precover, EagerNonrecursive, LazyNonrecursive, NonrecursiveDFADecomp,
    enumeration.
    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = tuple(target)
        self.N = len(self.target)
        fst.ensure_arc_indexes()
        self._has_eps = EPSILON in fst.A

    def epsremove(self):
        if self._has_eps:
            return super().epsremove()
        return self

    def arcs(self, state):
        (i, ys) = state
        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:      # target and ys are not prefixes of one another.
            return
        if m == self.N:                    # i.e, target <= ys
            for x, j in self.fst.index_i_xj.get(i, ()):
                yield (x, (j, self.target))
        else:                              # i.e, ys < target)
            for x, j in self.fst.index_iy_xj.get((i, EPSILON), ()):
                yield (x, (j, ys))
            for x, j in self.fst.index_iy_xj.get((i, self.target[n]), ()):
                yield (x, (j, self.target[:n+1]))

    def arcs_x(self, state, x):
        (i, ys) = state
        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:
            return
        if m == self.N:
            for j in self.fst.index_ix_j.get((i, x), ()):
                yield (j, self.target)
        else:
            for j in self.fst.index_ixy_j.get((i, x, EPSILON), ()):
                yield (j, ys)
            for j in self.fst.index_ixy_j.get((i, x, self.target[n]), ()):
                yield (j, self.target[:n+1])

    def start(self):
        for i in self.fst.start:
            yield (i, self.target[:0])

    def is_final(self, state):
        (i, ys) = state
        return self.fst.is_final(i) and ys[:self.N] == self.target


class TruncationMarkerPrecoverNFA(Lazy):
    """
    Precover NFA with an explicit truncation bit.

    Like PrecoverNFA but augments states with a boolean indicating whether
    the output buffer was truncated (i.e., there are more output symbols
    than captured in the state).  This gives finer state distinctions for
    determinization.

    State format: ``(i, ys, truncated)``

    Used by: Precover (impl='push-truncated').
    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = tuple(target)
        self.N = len(self.target)
        fst.ensure_arc_indexes()
        self._has_eps = EPSILON in fst.A

    def epsremove(self):
        if self._has_eps:
            return super().epsremove()
        return self

    def arcs(self, state):
        (i, ys, truncated) = state
        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:      # target and ys are incompatible
            return
        if m == self.N:                    # i.e, target <= ys
            if truncated:
                for x, j in self.fst.index_i_xj.get(i, ()):
                    yield (x, (j, self.target, True))
            else:
                # Need y to distinguish EPSILON from non-EPSILON
                for x, y, j in self.fst.arcs(i):
                    if y == EPSILON:
                        yield (x, (j, self.target, False))
                    else:
                        yield (x, (j, self.target, True))
        else:                              # i.e, ys < target)
            assert not truncated
            for x, j in self.fst.index_iy_xj.get((i, EPSILON), ()):
                yield (x, (j, ys, False))
            for x, j in self.fst.index_iy_xj.get((i, self.target[n]), ()):
                yield (x, (j, self.target[:n+1], False))

    def arcs_x(self, state, x):
        (i, ys, truncated) = state
        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:
            return
        if m == self.N:
            if truncated:
                for j in self.fst.index_ix_j.get((i, x), ()):
                    yield (j, self.target, True)
            else:
                # Need y to distinguish EPSILON from non-EPSILON
                for y, j in self.fst.arcs(i, x):
                    if y == EPSILON:
                        yield (j, self.target, False)
                    else:
                        yield (j, self.target, True)
        else:
            assert not truncated
            for j in self.fst.index_ixy_j.get((i, x, EPSILON), ()):
                yield (j, ys, False)
            for j in self.fst.index_ixy_j.get((i, x, self.target[n]), ()):
                yield (j, self.target[:n+1], False)

    def start(self):
        for i in self.fst.start:
            yield (i, self.target[:0], False)

    def is_final(self, state):
        (i, ys, _) = state
        return self.fst.is_final(i) and ys[:self.N] == self.target


# [2025-12-14 Sun] Is it possible that with the pop version of the precover
#   automaton that we don't have to worry about truncation and other inefficient
#   things like that?  Is there some way in which we could run in "both
#   direction" some how?  I have this feeling that it is possible to switch
#   between the states as they ought to be totally isomorphic.  My worry with
#   the pop version below is that there would appear to be inherently less
#   sharing of work because the precovers of prefixes aren't guaranteed to be
#   useful (that said, due to truncation they aren't).  Maybe this is an
#   empirical question?
class PopPrecoverNFA(Lazy):
    """
    Pop/suffix-oriented precover NFA.

    Equivalent to PrecoverNFA, but the buffer starts with the full target
    string and shrinks as symbols are consumed.  This construction is
    better-suited for exploiting work on common suffixes.

    State format: ``(i, ys)`` where ``ys`` is the remaining target suffix.

    Used by: Precover (impl='pop'). Experimental.
    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = tuple(target)
        self.N = len(self.target)

    def arcs(self, state):
        (i, ys) = state
        n = len(ys)
        if n == 0:
            for x, _, j in self.fst.arcs(i):
                yield (x, (j, ys))
        else:
            for x, y, j in self.fst.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                elif y == ys[0]:
                    yield (x, (j, ys[1:]))

    def arcs_x(self, state, x):
        (i, ys) = state
        n = len(ys)
        if n == 0:
            for _, j in self.fst.arcs(i, x):
                yield (j, ys)
        else:
            for y, j in self.fst.arcs(i, x):
                if y == EPSILON:
                    yield (j, ys)
                elif y == ys[0]:
                    yield (j, ys[1:])

    def start(self):
        for i in self.fst.start:
            yield (i, self.target)

    def is_final(self, state):
        (i, ys) = state
        return self.fst.is_final(i) and len(ys) == 0


class TargetSideBuffer(Lazy):
    """
    Simplest buffer NFA — unconditional accumulation, no truncation.

    Buffer grows unboundedly, which can produce infinite state spaces.
    Must be wrapped with Relevance to bound the search.

    State format: ``(i, ys)`` where ``ys`` accumulates all FST output.

    See also: Relevance (filter wrapper).
    """

    def __init__(self, f):
        self.f = f

    def arcs(self, state):
        (i, ys) = state
        for x,y,j in self.f.arcs(i):
            yield x, (j, ys if y == EPSILON else ys + (y,))

    def arcs_x(self, state, x):
        (i, ys) = state
        for y, j in self.f.arcs(i, x):
            yield (j, ys if y == EPSILON else ys + (y,))

    def start(self):
        for i in self.f.start:
            yield (i, ())

    def is_final(self, state):
        raise NotImplementedError()


class Relevance(Lazy):
    """
    Filter wrapper that prunes states with irrelevant target-side buffers.

    Wraps another Lazy automaton and only keeps states where the buffer
    ``ys`` is a prefix of the target or vice versa.

    Wraps TargetSideBuffer.
    """

    def __init__(self, base, target):
        self.base = base
        self.target = tuple(target)

    def arcs(self, state):
        for x, (i, ys) in self.base.arcs(state):
            m = min(len(self.target), len(ys))
            if self.target[:m] == ys[:m]:
                yield x, (i, ys)

    def arcs_x(self, state, x):
        for (i, ys) in self.base.arcs_x(state, x):
            m = min(len(self.target), len(ys))
            if self.target[:m] == ys[:m]:
                yield (i, ys)

    def start(self):
        yield from self.base.start()

    def is_final(self, state):
        raise NotImplementedError()


class PeekabooLookaheadNFA(Lazy):
    """
    Precover NFA with K-lookahead truncation and truncation bit.

    Buffer grows up to ``N + K`` symbols (where N = len(target)), then
    gets truncated with a flag to track information loss.  This gives
    bounded state spaces while preserving lookahead information for
    the peekaboo decomposition.

    State format: ``(i, ys, truncated)``

    Used by: PeekabooState (peekaboo_incremental).
    """

    def __init__(self, fst, target, K=1):
        self.fst = fst
        self.target = tuple(target)
        self.N = len(self.target)
        self.K = K
        assert K >= 1
        fst.ensure_arc_indexes()
        # Arc labels are FST input symbols; epsilon arcs exist only if the
        # FST has input-epsilon arcs.  When absent, skip EpsilonRemove in
        # det() — the wrapper is a no-op but costs ~12-15% of runtime on
        # trivial {state} closures.
        self._has_eps = EPSILON in fst.A

    def epsremove(self):
        if self._has_eps:
            return super().epsremove()
        return self

    def arcs(self, state):
        (i, ys, truncated) = state
        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:      # target and ys are not prefixes of one another.
            return
        if m >= self.N:                    # i.e, target <= ys
            if truncated:
                for x, j in self.fst.index_i_xj.get(i, ()):
                    yield (x, (j, ys, True))
            else:
                for x, y, j in self.fst.arcs(i):
                    if y == EPSILON:
                        yield (x, (j, ys, False))
                    else:
                        was = ys + (y,)
                        now = was[:self.N+self.K]
                        yield (x, (j, now, (was != now)))
        else:                              # i.e, ys < target)
            assert not truncated
            for x, j in self.fst.index_iy_xj.get((i, EPSILON), ()):
                yield (x, (j, ys, False))
            for x, j in self.fst.index_iy_xj.get((i, self.target[n]), ()):
                yield (x, (j, self.target[:n+1], False))

    def arcs_x(self, state, x):
        (i, ys, truncated) = state
        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:
            return
        if m >= self.N:
            if truncated:
                for j in self.fst.index_ix_j.get((i, x), ()):
                    yield (j, ys, True)
            else:
                for y, j in self.fst.arcs(i, x):
                    if y == EPSILON:
                        yield (j, ys, False)
                    else:
                        was = ys + (y,)
                        now = was[:self.N+self.K]
                        yield (j, now, (was != now))
        else:
            for j in self.fst.index_ixy_j.get((i, x, EPSILON), ()):
                yield (j, ys, False)
            for j in self.fst.index_ixy_j.get((i, x, self.target[n]), ()):
                yield (j, self.target[:n+1], False)

    def start(self):
        for i in self.fst.start:
            yield (i, self.target[:0], False)

    def is_final(self, state):
        (i, ys, _) = state
        return self.fst.is_final(i) and ys[:self.N] == self.target and len(ys) > self.N


class PeekabooFixedNFA(Lazy):
    """
    Precover NFA with fixed N+1 truncation (no truncation bit).

    Simpler than PeekabooLookaheadNFA: the buffer grows to exactly
    ``N + 1`` symbols and then stops.  Three phases: grow to N,
    extend to N+1, truncate at N+1.

    State format: ``(i, ys)``

    Used by: Peekaboo, PeekabooStrings (peekaboo_nonrecursive).
    """

    def __init__(self, fst, target):
        self.fst = fst
        self.target = tuple(target)
        self.N = len(self.target)
        fst.ensure_arc_indexes()
        self._has_eps = EPSILON in fst.A

    def epsremove(self):
        if self._has_eps:
            return super().epsremove()
        return self

    def arcs(self, state):
        (i, ys) = state
        n = len(ys)
        m = min(n, self.N)
        if self.target[:m] != ys[:m]: return

        # case: grow the buffer until we have covered all of the target string
        if n < self.N:
            for x, j in self.fst.index_iy_xj.get((i, EPSILON), ()):
                yield (x, (j, ys))
            for x, j in self.fst.index_iy_xj.get((i, self.target[n]), ()):
                yield (x, (j, self.target[:n+1]))

        # extend the buffer beyond the target string by one symbol
        elif n == self.N:
            # Need y for ys + (y,), keep fst.arcs(i)
            for x, y, j in self.fst.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                else:
                    yield (x, (j, ys + (y,)))

        # truncate the buffer after the (N+1)th symbol
        elif n == self.N + 1:
            for x, j in self.fst.index_i_xj.get(i, ()):
                yield (x, (j, ys))

    def arcs_x(self, state, x):
        (i, ys) = state
        n = len(ys)
        m = min(n, self.N)
        if self.target[:m] != ys[:m]: return
        if n < self.N:
            for j in self.fst.index_ixy_j.get((i, x, EPSILON), ()):
                yield (j, ys)
            for j in self.fst.index_ixy_j.get((i, x, self.target[n]), ()):
                yield (j, self.target[:n+1])
        elif n == self.N:
            # Need y for ys + (y,), keep fst.arcs(i, x)
            for y, j in self.fst.arcs(i, x):
                if y == EPSILON:
                    yield (j, ys)
                else:
                    yield (j, ys + (y,))
        elif n == self.N + 1:
            for j in self.fst.index_ix_j.get((i, x), ()):
                yield (j, ys)

    def start(self):
        for i in self.fst.start:
            yield (i, ())

    def is_final(self, state):
        (i, ys) = state
        return self.fst.is_final(i) and ys[:self.N] == self.target and len(ys) == self.N+1


# Backward-compatible aliases for old names
LazyPrecoverNFA = PrecoverNFA
LazyPrecoverNFAWithTruncationMarker = TruncationMarkerPrecoverNFA
PopPrecover = PopPrecoverNFA
PeekabooPrecover = PeekabooLookaheadNFA  # note: the nonrecursive one was also called PeekabooPrecover
