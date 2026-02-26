"""Incremental FST decomposition.

Naming conventions (automata-theory style):
  x, y       — source / target symbols
  xs, ys     — source / target strings (tuples of symbols)
  s, t       — FST states (source of an arc / destination of an arc)
  N          — len(target)
  R, Q       — remainder, quotient (sets of source strings)
  F, G       — frontiers: sets of (state, buffer) pairs
"""
from transduction.fst import EPSILON
from transduction.util import validate_target, LogVector, logsumexp, memoize


class Incremental:

    def __init__(self, fst, lm=None, EOS=None):
        self.fst = fst
        self.lm = lm
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.EOS = EOS   # sentinel for end-of-string

    @memoize
    def decompose(self, target):
        validate_target(target, self.target_alphabet)

        N = len(target)
        if N == 0:
            worklist = {()}
        else:
            (parent_remainder, parent_quotient) = self.decompose(target[:-1])
            worklist = parent_quotient | parent_remainder

        remainder, quotient = set(), set()
        while worklist:
            candidates = []
            for xs in worklist:
                if self.is_cylinder(xs, target):
                    quotient.add(xs)
                    continue
                if self.is_member(xs, target):
                    remainder.add(xs)
                for x in self.source_alphabet:
                    next_xs = xs + (x,)
                    if self.is_live(next_xs, target):
                        candidates.append(next_xs)
            worklist = self.prune(candidates)

        return (remainder, quotient)

    @memoize
    def _lm_state(self, xs):
        """Return LMState for source string xs, caching incrementally."""
        if len(xs) == 0:
            return self.lm.initial()
        return self._lm_state(xs[:-1]) >> xs[-1]

    def logprefix(self, target):
        """Log P(output starts with target)."""
        R, Q = self.decompose(target)
        parts = []
        for xs in Q:
            parts.append(self._lm_state(xs).logprefix)
        for xs in R:
            parts.append(self._lm_state(xs).logprob)
        return logsumexp(parts) if parts else float('-inf')

    def logprob(self, target):
        """Log P(output = target exactly).

        BFS-expands Q strings to find all source strings that produce exactly
        target, then sums their string probabilities.
        """
        R, Q = self.decompose(target)
        N = len(target)
        parts = []
        for xs in R:
            if self.is_exact_member(xs, target):
                parts.append(self._lm_state(xs).logprob)
        worklist = list(Q)
        visited = set(Q)
        while worklist:
            candidates = []
            for xs in worklist:
                if self.is_exact_member(xs, target):
                    parts.append(self._lm_state(xs).logprob)
                for x in self.source_alphabet:
                    next_xs = xs + (x,)
                    if next_xs not in visited and self._is_exact_live(next_xs, target, N):
                        visited.add(next_xs)
                        candidates.append(next_xs)
            worklist = self.prune(candidates)
        return logsumexp(parts) if parts else float('-inf')

    def _is_exact_live(self, xs, target, N):
        """Check if xs can still reach a final state with buffer == target.

        Stricter than is_live: requires at least one frontier state whose
        buffer has not exceeded len(target).
        """
        return any(
            target[:min(N, len(ys))] == ys[:min(N, len(ys))] and len(ys) <= N
            for (_, ys) in self.run(xs, target)
        )

    def logp_next_v2(self, target):
        logp = LogVector()
        for y in self.target_alphabet:
            logp[y] = self.logprefix(target + (y,))
        logp[self.EOS] = self.logprob(target)
        return logp.normalize()

    def logp_next(self, target):
        """Next-symbol distribution over target_alphabet ∪ {EOS}.

        Single decompose call, joint BFS across all target symbols.
        """
        logp = LogVector()
        R, Q = self.decompose(target)
        seeds = R | Q
        all_source_strings = set(seeds)

        seen = set()
        queue = [(xs, y) for xs in seeds for y in self.reachable_outputs(xs, target)]
        seen.update(queue)

        while queue:
            next_queue = []
            for xs, y in queue:
                ty = target + (y,)
                if self.is_cylinder(xs, ty):
                    logp.logaddexp(y, self._lm_state(xs).logprefix)
                else:
                    if self.is_member(xs, ty):
                        logp.logaddexp(y, self._lm_state(xs).logprob)
                    for x in self.source_alphabet:    # TODO: how can we speed this loop up?
                        next_xs = xs + (x,)
                        if ((next_xs, y) not in seen
                                and self.is_live(next_xs, ty)
                                and self._lm_state(next_xs).logprefix > float('-inf')):
                            seen.add((next_xs, y))
                            next_queue.append((next_xs, y))
                            all_source_strings.add(next_xs)
            queue = next_queue    # TODO: needs pruning!

        for xs in all_source_strings:
            if self.is_exact_member(xs, target):
                logp.logaddexp(self.EOS, self._lm_state(xs).logprob)

        return logp.normalize()

    def reachable_outputs(self, xs, target):
        """Target symbols reachable from source string xs given target prefix."""
        result = set()
        F = self.run(xs, target)
        N = len(target)
        for s, ys in F:
            if len(ys) > N and ys[:N] == target:
                result.add(ys[N])
            for y in self.target_alphabet:
                extended = target + (y,)
                M = N + 1
                for y2, t in self.fst.arcs(s, EPSILON):
                    next_buf = ys if y2 == EPSILON else ys + (y2,)
                    if extended[:min(M, len(next_buf))] == next_buf[:min(M, len(next_buf))]:
                        result.add(y)
                for x in self.source_alphabet:
                    for y2, t in self.fst.arcs(s, x):
                        next_buf = ys if y2 == EPSILON else ys + (y2,)
                        if extended[:min(M, len(next_buf))] == next_buf[:min(M, len(next_buf))]:
                            result.add(y)
        return result

    def is_exact_member(self, xs, target):
        return any(
            self.fst.is_final(s)
            for (s, ys) in self.run(xs, target)
            if ys == target
        )

    def prune(self, candidates):
        if self.lm is None:
            return candidates
        return [xs for xs in candidates if self._lm_state(xs).logprefix > float('-inf')]

    def is_live(self, xs, target):
        N = len(target)
        return any(
            target[:min(N, len(ys))] == ys[:min(N, len(ys))]
            for (_, ys) in self.run(xs, target)
        )

    def is_member(self, xs, target):
        N = len(target)
        return any(self.fst.is_final(s) for (s, ys) in self.run(xs, target) if ys[:N] == target)

    @memoize
    def is_cylinder(self, xs, target):

        N = len(target)

        def refine(F):
            "On-demand truncation and filtering for the frontier's buffer to the target string's prefixes."
            return frozenset({
                (s, ys[:N]) for s, ys in F
                if ys[:min(N, len(ys))] == target[:min(N, len(ys))]
            })

        F0 = refine(self.run(xs, target))
        worklist = [F0]
        visited = {F0}

        while worklist:
            F = worklist.pop()

            # Accepting
            if not any(self.fst.is_final(s) for (s, ys) in F if ys[:N] == target):
                return False

            # Complete
            for x in self.source_alphabet:
                G = refine(self.step(F, x, target))
                if len(G) == 0: return False   # dead state
                if G not in visited:
                    visited.add(G)
                    worklist.append(G)

        return True

    @memoize
    def run(self, xs, target):
        if len(xs) == 0:
            return frozenset(self.closure({(s, ()) for s in self.fst.start}, target))
        else:
            return frozenset(self.step(self.run(xs[:-1], target), xs[-1], target))

    def step(self, F, x, target):
        assert x != EPSILON
        N = len(target)
        G = set()
        for s, ys in F:
            for y, t in self.fst.arcs(s, x):
                next_buf = ys if y == EPSILON else ys + (y,)
                if target[:min(N, len(next_buf))] == next_buf[:min(N, len(next_buf))]:
                    G.add((t, next_buf))
        return self.closure(G, target)

    def closure(self, F, target):
        worklist = set(F)
        G = set(F)
        N = len(target)
        while worklist:
            (s, buf) = worklist.pop()
            for y, t in self.fst.arcs(s, EPSILON):
                next_buf = buf if y == EPSILON else buf + (y,)
                if target[:min(N, len(next_buf))] == next_buf[:min(N, len(next_buf))]:
                    if (t, next_buf) not in G:
                        worklist.add((t, next_buf))
                        G.add((t, next_buf))
        return G
