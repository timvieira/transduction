"""Incremental FST decomposition with LM-weighted next-symbol prediction.

Given an FST T and a language model LM over source strings, computes the
pushforward distribution:  P_T(y₁…yₙ) = Σ_{xs : T(xs) starts with y₁…yₙ} P_LM(xs).

The key data structure is the *decomposition* of the preimage T⁻¹(target·Σ*)
into a finite union  R ∪ Q·Σ*  where R (remainder) and Q (quotient) are finite
sets of source strings.  This lets us split the pushforward sum:

    P(output starts with target) = Σ_{xs∈R} P_LM(xs)  +  Σ_{xs∈Q} P_LM(xs·Σ*)
                                 = Σ_{xs∈R} logprob(xs) +  Σ_{xs∈Q} logprefix(xs)

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
        """Decompose T⁻¹(target·Σ*) into R ∪ Q·Σ* (finite remainder + quotient).

        Incrementally refines the parent decomposition: seeds are R∪Q from
        decompose(target[:-1]).  Parent Q strings may lose universality at
        the new target symbol; parent R strings may gain new extensions.

        BFS terminates because:
        - is_cylinder absorbs entire subtrees (Q grows monotonically in depth)
        - is_live filters dead prefixes (buffer incompatible with target)
        - prune filters LM-dead prefixes (zero probability under LM)
        """
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
                    continue   # absorb: no need to explore extensions of xs
                if self.is_member(xs, target):
                    remainder.add(xs)
                # not a cylinder → must explore extensions
                for x in self.reachable_inputs(xs, target):
                    next_xs = xs + (x,)
                    assert self.is_live(next_xs, target)
                    candidates.append(next_xs)
            worklist = self.prune(candidates)

        return (remainder, quotient)

    @memoize
    def _lm_state(self, xs):
        """Return LMState for source string xs, built incrementally.

        Composing state(xs) = state(xs[:-1]) >> xs[-1] means each prefix is
        computed at most once (memoized), so total cost across all source
        strings explored is O(sum of their lengths), not O(sum of lengths²).
        """
        if len(xs) == 0:
            return self.lm.initial()
        return self._lm_state(xs[:-1]) >> xs[-1]

    def logprefix(self, target):
        """Log P(output starts with target).

        Correctness: R ∪ Q·Σ* partitions the source strings whose output starts
        with target.  Q strings are cylinders — every extension also produces
        target — so we sum the LM *prefix* probability (all continuations).
        R strings are isolated members, so we sum their *string* probability.
        The two sets are disjoint by construction, so there is no double-counting.
        """
        R, Q = self.decompose(target)
        parts = []
        for xs in Q:
            parts.append(self._lm_state(xs).logprefix)   # P_LM(xs·Σ*)
        for xs in R:
            parts.append(self._lm_state(xs).logprob)      # P_LM(xs·EOS)
        return logsumexp(parts) if parts else float('-inf')

    def logprob(self, target):
        """Log P(output = target exactly).

        R strings may or may not produce exactly target (some produce longer
        outputs starting with target), so we filter by is_exact_member.
        Q strings are cylinders — their extensions can also produce exactly
        target — so we BFS-expand Q, collecting exact members along the way.
        Uses _is_exact_live (buffer must not overshoot target length) for
        tighter pruning than the prefix-compatible is_live.
        """
        R, Q = self.decompose(target)
        N = len(target)
        parts = []
        for xs in R:
            if self.is_exact_member(xs, target):
                parts.append(self._lm_state(xs).logprob)
        # BFS through Q extensions: cylinders guarantee every continuation
        # produces output *starting with* target, but we need *exactly* target.
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

        Computes P(y | target) for each y simultaneously via a joint BFS over
        (source_string, target_symbol) pairs.  Uses a single decompose(target)
        call to seed the search, then for each seed and each reachable output
        symbol y, extends source strings until they are classified:

        - Cylinder for target·y → contributes logprefix  (all continuations ok)
        - Member for target·y  → contributes logprob     (this string only)
        - Otherwise            → extend by one source symbol and continue

        EOS is handled separately: after the BFS, all discovered source strings
        are checked for exact match with target.

        Correctness: equivalent to logp_next_v2 (one logprefix/logprob call per
        symbol), but shares source-string discovery across symbols.

        Efficiency vs logp_next_v2: with memoized decompose, both make the same
        decompose calls.  logp_next avoids redundant is_live/is_cylinder checks
        and shares LM state lookups across symbols.  reachable_outputs prunes
        the (xs, y) pairs to only those where y is reachable in one FST step.
        """
        logp = LogVector()
        R, Q = self.decompose(target)
        seeds = R | Q
        # Track all source strings encountered so we can check EOS at the end.
        all_source_strings = set(seeds)

        # R strings are not cylinders for target, hence not for any target·y
        # (target·y is strictly more restrictive).  Skip the expensive powerset
        # universality check for these seeds.
        non_cylinders = set(R)

        # Seed the BFS: for each seed xs and each target symbol y reachable
        # from xs's frontier, create a (xs, y) work item.  reachable_outputs
        # filters to only those y that appear on at least one FST arc from
        # the frontier — avoiding wasted work for unreachable symbols.
        seen = set()
        queue = [(xs, y) for xs in seeds for y in self.reachable_outputs(xs, target)]
        seen.update(queue)

        # Functional-FST shortcut: a source string can be a cylinder for at
        # most one y (see proof in logp_next_v4).  Once absorbed, skip it for
        # all other symbols.
        absorbed = set()

        # BFS loop: classify each (xs, y) pair.  Each iteration processes one
        # "level" of source-string extensions (all strings of the same length).
        while queue:
            next_queue = []
            for xs, y in queue:
                if xs in absorbed:
                    continue   # cylinder for another y → skip entirely
                ty = target + (y,)
                if xs not in non_cylinders and self.is_cylinder(xs, ty):
                    # xs is universal for target·y: every continuation of xs
                    # produces output starting with target·y.  The entire
                    # subtree is absorbed — contributes prefix probability
                    # and no further extensions are needed.
                    logp.logaddexp(y, self._lm_state(xs).logprefix)
                    absorbed.add(xs)
                else:
                    # xs is not a cylinder for target·y.  If it's a member
                    # (produces target·y at a final state), it contributes
                    # its string probability.  Either way, we must continue
                    # exploring extensions — the subtree is not fully classified.
                    if self.is_member(xs, ty):
                        logp.logaddexp(y, self._lm_state(xs).logprob)
                    for x in self.reachable_inputs(xs, ty):    # TODO: how can we speed this loop up?
                        next_xs = xs + (x,)
                        assert self.is_live(next_xs, ty)
                        if ((next_xs, y) not in seen and self._lm_state(next_xs).logprefix > float('-inf')):
                            seen.add((next_xs, y))
                            next_queue.append((next_xs, y))
                            all_source_strings.add(next_xs)
            queue = next_queue    # TODO: needs pruning!

        # EOS: check all discovered source strings for exact match with target.
        # This is separate from the per-y BFS because EOS means the output is
        # *exactly* target (no further symbols), which is orthogonal to whether
        # xs can produce target·y for some y.
        for xs in all_source_strings:
            if self.is_exact_member(xs, target):
                logp.logaddexp(self.EOS, self._lm_state(xs).logprob)

        return logp.normalize()

    def logp_next_v3(self, target):
        """Next-symbol distribution over target_alphabet ∪ {EOS}.

        Same as logp_next but with separate BFS queues per target symbol y.
        This makes per-symbol termination and pruning natural: each y's search
        runs independently, and fast symbols (all strings classified early)
        stop without waiting for slower ones.
        """
        logp = LogVector()
        R, Q = self.decompose(target)
        seeds = R | Q
        all_source_strings = set(seeds)

        # R strings are not cylinders for target, hence not for any target·y.
        non_cylinders = set(R)

        # Build per-y seed lists from reachable outputs (only reachable y's).
        from collections import defaultdict
        queues = defaultdict(list)
        for xs in seeds:
            for y in self.reachable_outputs(xs, target):
                queues[y].append(xs)

#        def run_queue(queue, target):
#            while queue:
#                candidates = []
#                for xs in worklist:
#
#                    if self.is_cylinder(xs, target):
#                        quotient.add(xs)
#                        continue   # absorb: no need to explore extensions of xs
#                    if self.is_member(xs, target):
#                        remainder.add(xs)
#                    # not a cylinder → must explore extensions
#                    for x in self.reachable_inputs(xs, target):
#                        next_xs = xs + (x,)
#                        assert self.is_live(next_xs, target)
#                        next_queue.append(next_xs)
#
#                queue = self.prune(next_queue)


        # Functional-FST shortcut: a source string can be a cylinder for at
        # most one y (see proof in logp_next_v4).  Once absorbed, skip it for
        # all other symbols.
        absorbed = set()

        # Per-symbol BFS: one queue per reachable target symbol y.
        for y, queue in queues.items():
            ty = target + (y,)
            seen = set(queue)

#            run_queue(queue, ty)
            while queue:
                next_queue = []
                for xs in queue:
                    if xs in absorbed:
                        continue   # cylinder for another y → skip entirely
                    if xs not in non_cylinders and self.is_cylinder(xs, ty):
                        logp.logaddexp(y, self._lm_state(xs).logprefix)
                        absorbed.add(xs)
                    else:
                        if self.is_member(xs, ty):
                            logp.logaddexp(y, self._lm_state(xs).logprob)
                        for x in self.reachable_inputs(xs, ty):
                            next_xs = xs + (x,)
                            if (next_xs not in seen
                                    and self._lm_state(next_xs).logprefix > float('-inf')):
                                seen.add(next_xs)
                                next_queue.append(next_xs)
                                all_source_strings.add(next_xs)
                queue = next_queue    # TODO: needs pruning!

        # EOS: check all discovered source strings for exact match with target.
        for xs in all_source_strings:
            if self.is_exact_member(xs, target):
                logp.logaddexp(self.EOS, self._lm_state(xs).logprob)

        return logp.normalize()

    def logp_next_v4(self, target):
        """Next-symbol distribution over target_alphabet ∪ {EOS}.

        Per-symbol BFS that mirrors decompose's structure: each y's search
        collects sets of cylinders and members (idempotent add), then sums
        contributions at the end.  No 'seen' set needed — set idempotency
        handles duplicates, just like decompose.
        """
        logp = LogVector()
        R, Q = self.decompose(target)
        seeds = R | Q
        all_source_strings = set(seeds)

        # R strings are not cylinders for target, hence not for any target·y.
        non_cylinders = set(R)

        # Build per-y seed sets from reachable outputs (only reachable y's).
        from collections import defaultdict
        queues = defaultdict(set)
        for xs in seeds:
            for y in self.reachable_outputs(xs, target):
                queues[y].add(xs)

        # Functional-FST shortcut (ported from peekaboo_incremental.py):
        # a source string can be a cylinder for at most one target symbol y.
        # Once found to be a cylinder for y, skip it in all other symbols'
        # BFS loops — its entire subtree maps to target·y, contributing
        # nothing to z ≠ y.
        #
        # NOTE: This assumes the FST is functional.  A productive
        # input-epsilon cycle (eps-input arcs that produce non-epsilon
        # output) makes an FST non-functional, since the cycle can be
        # traversed any number of times yielding distinct outputs for
        # the same input.  Non-functional FSTs may violate the
        # uniqueness invariant below.
        #
        # Proof (functional FSTs): Suppose xs is a cylinder for both y
        # and z (y ≠ z).  Being a cylinder for y means every extension
        # of xs produces output starting with target·y.  Likewise for z.
        # For a functional FST each input has a unique output, so the
        # preimages of target·y·Σ* and target·z·Σ* are disjoint.
        # But xs and all its extensions are in both preimages — contradiction.
        absorbed = set()

        # Per-symbol BFS: collect cylinder/member sets per y, then sum.
        for y, worklist in queues.items():
            ty = target + (y,)
            cylinders, members = set(), set()

            while worklist:
                candidates = set()
                for xs in worklist:
                    if xs in absorbed:
                        continue   # cylinder for another y → skip entirely
                    if xs not in non_cylinders and self.is_cylinder(xs, ty):
                        cylinders.add(xs)
                        absorbed.add(xs)
                        continue   # absorb: no need to explore extensions
                    if self.is_member(xs, ty):
                        members.add(xs)
                    # not a cylinder → must explore extensions
                    for x in self.reachable_inputs(xs, ty):
                        next_xs = xs + (x,)
                        assert self.is_live(next_xs, ty)
                        candidates.add(next_xs)
                        all_source_strings.add(next_xs)
                worklist = self.prune(candidates)    # TODO: needs pruning!

            for xs in cylinders:
                logp.logaddexp(y, self._lm_state(xs).logprefix)
            for xs in members:
                logp.logaddexp(y, self._lm_state(xs).logprob)

            assert (members, cylinders) == self.decompose(ty)
            
        # EOS: check all discovered source strings for exact match with target.
        for xs in all_source_strings:
            if self.is_exact_member(xs, target):
                logp.logaddexp(self.EOS, self._lm_state(xs).logprob)

        return logp.normalize()

    @memoize
    def reachable_inputs(self, xs, target):
        """Source symbols x with arcs from xs's frontier states.

        Dual of reachable_outputs: prunes the source-symbol loop in decompose
        and logp_next to only source symbols that produce a target-compatible
        buffer.  Guarantees is_live(xs+(x,), target) for every x returned:
        if x is in the result, step(run(xs, target), x, target) has at least
        one state before ε-closure, and closure only adds states.
        """
        result = set()
        N = len(target)
        for s, ys in self.run(xs, target):
            for a, b, t in self.fst.arcs(s):
                if a != EPSILON:
                    next_buf = ys if b == EPSILON else ys + (b,)
                    if target[:min(N, len(next_buf))] == next_buf[:min(N, len(next_buf))]:
                        result.add(a)
        return result

    def reachable_outputs(self, xs, target):
        """Target symbols y reachable from xs's frontier in one FST step.

        Checks which y could appear at position N of the output by looking at
        arcs (both ε and source-symbol) from each frontier state.  This prunes
        the (xs, y) pairs in logp_next, avoiding BFS work for unreachable y.

        Soundness (no y with nonzero probability is missed): for y to have
        nonzero probability, some seed's frontier must have a state (s, ys)
        on a path to producing y at position N.  At the first FST step:
        - Buffer already past N (len(ys) > N): ys[N] adds y directly.
        - Buffer at length N, arc outputs y: target+(y)==target+(y2) adds y.
        - Buffer shorter than N (or at N with ε-output): the compatibility
          check never sees position N, so it passes for *all* y — a sound
          over-approximation (spurious y's get filtered by is_live later).
        """
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
        """True if T(xs) = target exactly (buffer matches target at a final state)."""
        return any(
            self.fst.is_final(s)
            for (s, ys) in self.run(xs, target)
            if ys == target
        )

    def prune(self, candidates):
        """Drop source strings with zero LM prefix probability."""
        if self.lm is None:
            return candidates
        return [xs for xs in candidates if self._lm_state(xs).logprefix > float('-inf')]

    def is_live(self, xs, target):
        """True if xs has at least one frontier state with buffer compatible with target.

        Buffer compatibility: target and buffer agree on their shared prefix,
        i.e., target[:k] == ys[:k] where k = min(|target|, |ys|).  This allows
        ys to be shorter (not yet produced enough output) or longer (already
        produced past target — still live for prefix matching).
        """
        N = len(target)
        return any(
            target[:min(N, len(ys))] == ys[:min(N, len(ys))]
            for (_, ys) in self.run(xs, target)
        )

    def is_member(self, xs, target):
        """True if xs produces output starting with target (final state + buffer match)."""
        N = len(target)
        return any(self.fst.is_final(s) for (s, ys) in self.run(xs, target) if ys[:N] == target)

    @memoize
    def is_cylinder(self, xs, target):
        """True if xs is a cylinder: every extension of xs produces target prefix.

        Performs powerset determinization on the frontier, checking that the
        resulting DFA is universal (all reachable states accepting, no dead
        states).  Each powerset state is a frozenset of (fst_state, buffer)
        pairs with buffers truncated to length N (positions beyond N are
        irrelevant for target-prefix checking).

        Two conditions checked at every reachable powerset state F:
        1. Accepting: F contains a final state with buffer matching target.
           (Ensures every string reaching F has *some* path producing target.)
        2. Complete: stepping F by each source symbol yields a non-empty set.
           (Ensures no continuation gets stuck in a dead state.)

        Terminates because the number of distinct powerset states is finite
        (bounded by 2^(|states| · N), though much smaller in practice due to
        buffer truncation and filtering).
        """
        N = len(target)

        def refine(F):
            """Filter to target-compatible states; truncate buffers to length N."""
            return frozenset({
                (s, ys[:N]) for s, ys in F
                if ys[:min(N, len(ys))] == target[:min(N, len(ys))]
            })

        F0 = refine(self.run(xs, target))
        worklist = [F0]
        visited = {F0}

        while worklist:
            F = worklist.pop()

            # Accepting: at least one path through F reaches a final state
            # with buffer matching target.
            if not any(self.fst.is_final(s) for (s, ys) in F if ys[:N] == target):
                return False

            # Complete: no source symbol leads to a dead (empty) powerset state.
            for x in self.source_alphabet:
                G = refine(self.step(F, x, target))
                if len(G) == 0: return False   # dead state → not universal
                if G not in visited:
                    visited.add(G)
                    worklist.append(G)

        return True

    @memoize
    def run(self, xs, target):
        """Compute the frontier: set of (state, buffer) pairs reachable by xs.

        Built incrementally: run(xs) = step(run(xs[:-1]), xs[-1]).  Memoized
        on (xs, target), so each prefix is computed exactly once across all
        callers (decompose, is_live, is_member, is_cylinder, etc.).

        Returns frozenset for hashability (needed by memoize and is_cylinder's
        visited set).
        """
        if len(xs) == 0:
            return frozenset(self.closure({(s, ()) for s in self.fst.start}, target))
        else:
            return frozenset(self.step(self.run(xs[:-1], target), xs[-1], target))

    def step(self, F, x, target):
        """Advance frontier F by one source symbol x, then ε-close.

        Buffer filtering invariant: only keeps (state, buffer) pairs where
        buffer is compatible with target — they agree on their shared prefix.
        Sound because FSTs only append to the buffer, never backtrack: once
        the buffer disagrees with target, no future transition can fix it.
        """
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
        """Follow ε-transitions to fixed point, maintaining buffer filtering."""
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
