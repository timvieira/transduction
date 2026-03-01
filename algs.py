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
from transduction.util import validate_target, LogVector, logsumexp
from transduction.universality import compute_ip_universal_states
from collections import defaultdict


class Incremental:

    def __init__(self, fst, lm=None, EOS=None):
        self.fst = fst
        self.lm = lm
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.EOS = EOS   # sentinel for end-of-string
        self._ip_universal = compute_ip_universal_states(fst)
        # Per-instance caches (keys do NOT include self)
        self._decompose_cache = {}
        self._run_cache = {}
        self._is_cylinder_cache = {}
        self._reachable_inputs_cache = {}
        self._lm_state_cache = {}

    def decompose(self, target):
        """Decompose T⁻¹(target·Σ*) into R ∪ Q·Σ* (remainder + quotient).

        Incrementally refines the parent decomposition: seeds are R∪Q from
        decompose(target[:-1]).  Parent Q strings may lose universality at
        the new target symbol; parent R strings may gain new extensions.

        BFS terminates because:
        - is_cylinder absorbs entire subtrees (Q grows monotonically in depth)
        - is_live filters dead prefixes (buffer incompatible with target)
        - prune filters LM-dead prefixes (zero probability under LM)
        """
        if target in self._decompose_cache:
            return self._decompose_cache[target]

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
                    candidates.append(xs + (x,))
            worklist = self.prune(candidates)

        result = (remainder, quotient)
        self._decompose_cache[target] = result
        return result

    def _lm_state(self, xs):
        """Return LMState for source string xs, built incrementally.

        Composing state(xs) = state(xs[:-1]) >> xs[-1] means each prefix is
        computed at most once (memoized), so total cost across all source
        strings explored is O(sum of their lengths), not O(sum of lengths²).
        """
        if xs in self._lm_state_cache:
            return self._lm_state_cache[xs]
        if len(xs) == 0:
            result = self.lm.initial()
        else:
            result = self._lm_state(xs[:-1]) >> xs[-1]
        self._lm_state_cache[xs] = result
        return result

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
            parts.append(self._lm_state(xs).logprefix)
        for xs in R:
            parts.append(self._lm_state(xs).logprob)
        return logsumexp(parts)

    def logprob(self, target):
        """Log P(output = target exactly).

        Reuses decompose_next, which computes the preimage (source strings
        mapping to exactly target) as a byproduct of its BFS.
        """
        (_, _, preimage) = self.decompose_next(target)
        return logsumexp([
            self._lm_state(xs).logprob for xs in preimage
        ])

    def logp_next_bruteforce(self, target):
        """Next-symbol distribution via independent per-symbol decompositions.

        Reference implementation: computes logprefix(target·y) for each y and
        logprob(target) for EOS.  Correct but redundant — each call triggers
        its own decompose/BFS.  Used as a test oracle for logp_next.
        """
        logp = LogVector()
        for y in self.target_alphabet:
            logp[y] = self.logprefix(target + (y,))
        logp[self.EOS] = self.logprob(target)
        return logp.normalize()

    def decompose_next(self, target):
        """Decompose T⁻¹(target·y·Σ*) for all next target symbols y at once.

        Single-worklist BFS over source strings, checking all reachable
        target symbols per string in one pass — directly mirroring peekaboo's
        per-state multi-symbol classification.  When xs is a cylinder for y,
        the entire subtree is absorbed and extension is skipped for all symbols.

        Returns (remainders, quotients, preimage) where:
        - remainders[y] = R(target·y), quotients[y] = Q(target·y)
        - preimage = {xs : T(xs) = target exactly}

        Functional-FST cylinder uniqueness: a source string can be a cylinder
        for at most one target symbol y.  Once found to be a cylinder for y,
        skip extension for all symbols — its entire subtree maps to target·y,
        contributing nothing to z ≠ y.

        NOTE: This assumes the FST is functional.  A productive input-epsilon
        cycle (eps-input arcs that produce non-epsilon output) makes an FST
        non-functional, since the cycle can be traversed any number of times
        yielding distinct outputs for the same input.

        Proof (functional FSTs): Suppose xs is a cylinder for both y and z
        (y ≠ z).  Being a cylinder for y means every extension of xs produces
        output starting with target·y.  Likewise for z.  For a functional FST
        each input has a unique output, so the preimages of target·y·Σ* and
        target·z·Σ* are disjoint.  But xs and all its extensions are in both
        preimages — contradiction.
        """
        R, Q = self.decompose(target)
        seeds = R | Q

        # R strings are not cylinders for target, hence not for any target·y
        # (target·y is strictly more restrictive).  Skip the expensive powerset
        # universality check for these seeds.
        non_cylinders = set(R)

        quotients = defaultdict(set)    # y → Q(target·y)
        remainders = defaultdict(set)   # y → R(target·y)
        preimage = set()

        worklist = set(seeds)
        while worklist:
            candidates = set()
            for xs in worklist:

                # Safe: if xs is a cylinder for target·y, every extension
                # produces output starting with target·y (strictly longer
                # than target), so no extension can be an exact preimage of
                # target.
                if self.is_exact_member(xs, target):
                    preimage.add(xs)

                # Check all reachable y's for this xs in one pass.
                # At most one y can be continuous (functional FST).
                continuous = None
                for y in self.reachable_outputs(xs, target):
                    ty = target + (y,)
                    if xs not in non_cylinders and self.is_cylinder(xs, ty):
                        assert continuous is None, \
                            f"cylinder for both {continuous!r} and {y!r} — non-functional?"
                        quotients[y].add(xs)
                        continuous = y
                        continue
                    if self.is_member(xs, ty):
                        remainders[y].add(xs)

                if continuous is not None:
                    continue   # absorbed — skip extension for all symbols

                for x in self.reachable_inputs(xs, target):
                    candidates.add(xs + (x,))

            worklist = self.prune(candidates)

        # Populate decompose cache: we already computed R(target·y)
        # and Q(target·y) for every y, so a subsequent decompose(target·y)
        # call (e.g., from decompose_next(target·y)) hits the cache instead
        # of re-running its own BFS.
        for y in set(quotients) | set(remainders):
            self._decompose_cache[target + (y,)] = (remainders[y], quotients[y])

        return remainders, quotients, preimage

    def logp_next(self, target):
        """Next-symbol distribution over target_alphabet ∪ {EOS}.

        Correctness: equivalent to logp_next_bruteforce (one logprefix/logprob
        call per symbol), but shares source-string discovery across symbols.
        """
        remainders, quotients, preimage = self.decompose_next(target)
        logp = LogVector()
        for y in set(quotients) | set(remainders):
            for xs in quotients[y]:
                logp.logaddexp(y, self._lm_state(xs).logprefix)
            for xs in remainders[y]:
                logp.logaddexp(y, self._lm_state(xs).logprob)
        for xs in preimage:
            logp.logaddexp(self.EOS, self._lm_state(xs).logprob)
        return logp.normalize()

    def reachable_inputs(self, xs, target):
        """Source symbols x with arcs from xs's frontier states.

        Dual of reachable_outputs: prunes the source-symbol loop in decompose
        and logp_next to only source symbols that produce a target-compatible
        buffer.  Guarantees is_live(xs+(x,), target) for every x returned:
        if x is in the result, step(run(xs, target), x, target) has at least
        one state before ε-closure, and closure only adds states.
        """
        key = (xs, target)
        if key in self._reachable_inputs_cache:
            return self._reachable_inputs_cache[key]
        result = set()
        N = len(target)
        for s, ys in self.run(xs, target):
            for a, b, t in self.fst.arcs(s):
                if a != EPSILON:
                    next_buf = ys if b == EPSILON else ys + (b,)
                    if target[:min(N, len(next_buf))] == next_buf[:min(N, len(next_buf))]:
                        result.add(a)
        self._reachable_inputs_cache[key] = result
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

    def _combined_universal(self, xs, target):
        """Sufficient condition for is_cylinder using precomputed ip-universal states.

        Given frontier F = run(xs, parent_target):
        - Committed (S_c): states with buffer ≽ parent·y (already committed to y)
        - Boundary (S_b): states with buffer = parent_target (undetermined)

        Returns True if:
        1. Every state in S_c ∪ S_b is ip-universal
        2. Every transition from a boundary state produces y (output exclusivity)
        3. At least one state in S_c ∪ S_b is accepting

        Speed–accuracy tradeoff: the proposition (prop:combined-universality)
        requires only *set-level* universality of S_c ∪ S_b (the NFA started
        from those states accepts Σ*), which is weaker than requiring every
        individual state to be ip-universal.  We use the per-state check here
        because it runs in O(|F|) with precomputed U, whereas a set-level
        universality check requires a powerset BFS that may be as expensive as
        the full is_cylinder check this shortcut is meant to avoid.
        """
        if len(target) == 0:
            return False
        parent_target = target[:-1]
        if (xs, parent_target) not in self._run_cache:
            return False
        F = self._run_cache[(xs, parent_target)]
        N = len(parent_target)
        y = target[-1]

        has_final = False
        for s, ys in F:
            if len(ys) > N:
                if ys[N] != y:
                    continue          # dead for this y, ignore
                if s not in self._ip_universal:
                    return False
                if self.fst.is_final(s):
                    has_final = True
            elif len(ys) == N:
                # Boundary state: must be ip-universal and exclusively produce y
                if s not in self._ip_universal:
                    return False
                for a, b, _t in self.fst.arcs(s):
                    # Non-ε-input arcs must produce exactly y (σ:ε is wrong:
                    # consumes input without producing y, successor unchecked).
                    # ε-input arcs must produce y or ε (ε:ε successors are in
                    # S_b via frontier ε-closure; ε:y successors are in S_c).
                    if not (b == y or (a == EPSILON and b == EPSILON)):
                        return False
                if self.fst.is_final(s):
                    has_final = True
            # else: behind parent, ignore
        return has_final

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
        key = (xs, target)
        if key in self._is_cylinder_cache:
            return self._is_cylinder_cache[key]

        # Cheap shortcut: check ip-universal states
        if len(target) > 0 and self._combined_universal(xs, target):
            self._is_cylinder_cache[key] = True
            return True

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
                self._is_cylinder_cache[key] = False
                return False

            # Complete: no source symbol leads to a dead (empty) powerset state.
            for x in self.source_alphabet:
                G = refine(self.step(F, x, target))
                if len(G) == 0:
                    self._is_cylinder_cache[key] = False
                    return False   # dead state → not universal
                if G not in visited:
                    visited.add(G)
                    worklist.append(G)

        self._is_cylinder_cache[key] = True
        return True

    def run(self, xs, target):
        """Compute the frontier: set of (state, buffer) pairs reachable by xs.

        Two recursion strategies, chosen opportunistically:
        - xs-recursion: run(xs, t) = step(run(xs[:-1], t), xs[-1], t).
          Extends one source symbol with target fixed.
        - target-recursion: run(xs, t·y) = filter(run(xs, t)).
          Derives from parent target by dropping elements with buffer[N] ≠ y.
          Sound because buffers only grow: once buffer[N] = z ≠ y, all
          descendants also have position N = z.

        Target-recursion is used when run(xs, target[:-1]) is already cached
        (typical in decompose_next, where we move from target to target·y).
        This avoids recomputing the entire xs chain for the new target.

        Returns frozenset for hashability (needed by is_cylinder's visited set).
        """
        key = (xs, target)
        if key in self._run_cache:
            return self._run_cache[key]

        if len(xs) == 0:
            result = frozenset(self.closure({(s, ()) for s in self.fst.start}, target))
        elif len(target) > 0:
            # Opportunistic target-recursion: if parent-target frontier is cached,
            # derive by filtering instead of recomputing the xs chain.
            parent_target = target[:-1]
            parent_key = (xs, parent_target)
            if parent_key in self._run_cache:
                y = target[-1]
                N = len(parent_target)
                result = frozenset(
                    (s, ys) for s, ys in self._run_cache[parent_key]
                    if len(ys) <= N or ys[N] == y
                )
            else:
                # Default: xs-recursion
                result = frozenset(self.step(self.run(xs[:-1], target), xs[-1], target))
        else:
            # Default: xs-recursion
            result = frozenset(self.step(self.run(xs[:-1], target), xs[-1], target))

        self._run_cache[key] = result
        return result

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
