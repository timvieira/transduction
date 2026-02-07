from transduction.base import PrecoverDecomp
from transduction.lazy import Lazy
from transduction.fsa import FSA, frozenset
from transduction.fst import (
    EPSILON, check_all_input_universal, compute_ip_universal_states,
    UniversalityFilter,
)
from transduction.precover_nfa import PeekabooLookaheadNFA as PeekabooPrecover

from collections import deque

#_______________________________________________________________________________
#
# [2025-12-09 Tue] TRUNCATION STRATEGIES: COST-BENEFIT ANALYSIS - The strategy
#   that we have taken in the current implementation is truncate as early as
#   possible - this minimizes the work in the current iteration.  However, it
#   might lead to more work in a later iteration because more nodes are marked
#   as truncated, meaning that they cannot be used in the later iterations.  If
#   we used a different truncation policy it might be the case that we could
#   share more work.  For example, if there is a small number of nodes that we
#   could in principle enumerate now (like the dfa_decomp strategy does) then we
#   could get away with that.  It is not possible, in general, to never truncate
#   if we want to terminate.  However, the truncation strategy has a
#   cost-benefit analysis, which I am trying to elucidate a bit.  The knob that
#   controls this is the "truncation policy" and there are smarter things than
#   truncating at N+1 (even for the triplets of doom example).
#_______________________________________________________________________________
#

def _trimmed_fsa(start_states, stop_states, incoming):
    """Build a trimmed FSA by backward BFS from stop states through the
    reverse-arc graph.  All states in `incoming` are forward-reachable
    (guaranteed by the BFS that built them), so backward reachability
    from stops gives exactly the trim (forward ∩ backward reachable) set."""
    if not stop_states:
        return FSA()
    backward_reachable = set()
    worklist = deque(stop_states)
    while worklist:
        state = worklist.popleft()
        if state in backward_reachable:
            continue
        backward_reachable.add(state)
        for _, pred in incoming.get(state, ()):
            if pred not in backward_reachable:
                worklist.append(pred)
    arcs = [
        (pred, x, state)
        for state in backward_reachable
        for x, pred in incoming.get(state, ())
        if pred in backward_reachable
    ]
    return FSA(
        start={s for s in start_states if s in backward_reachable},
        arcs=arcs,
        stop=stop_states,
    )


class FstUniversality:
    """Precomputed universality info for an FST.  Computed once, shared across
    all PeekabooState instances via the ``>>`` chain."""

    def __init__(self, fst):
        self.all_input_universal = check_all_input_universal(fst)
        self.ip_universal_states = (
            frozenset() if self.all_input_universal
            else compute_ip_universal_states(fst)
        )

    def make_filter(self, fst, target, dfa, source_alphabet):
        """Build a `UniversalityFilter` for one target symbol extension."""
        witnesses = (
            frozenset((q, target, False) for q in self.ip_universal_states)
            if not self.all_input_universal else frozenset()
        )
        return UniversalityFilter(
            fst=fst, target=target, dfa=dfa,
            source_alphabet=source_alphabet,
            all_input_universal=self.all_input_universal,
            witnesses=witnesses,
        )


class Peekaboo:
    """
    Recursive, batched computation of next-target-symbol optimal DFA-decomposition.
    """
    def __init__(self, fst):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self._univ = FstUniversality(fst)

    def __call__(self, target):
        # Merge reverse-arc (incoming) dicts from every depth into one
        # shared graph, then extract per-symbol trimmed FSAs via backward
        # reachability from Q/R stop states.
        #
        # Cross-depth merging correctness argument
        # ------------------------------------------
        # Each depth d uses a different DFA (PeekabooPrecover with
        # target[:d]).  At resume-frontier boundaries a single DFA state
        # may have incoming arcs from *two* depths (d and d+1) with
        # potentially different predecessors for the same input symbol,
        # making the merged graph nondeterministic.  We argue the extra
        # arcs from earlier depths are harmless:
        #
        #   Structural invariant: at depth d the PeekabooPrecover NFA
        #   states have output buffers of length ≤ d+1 (= N+K with
        #   K=1).  DFA states are frozensets of such NFA states, so
        #   every DFA state reachable via depth-d arcs contains only
        #   NFA states with len(ys) ≤ d+1.
        #
        #   The Q/R stop states live at the final depth N=len(target)
        #   and contain NFA states with len(ys) = N+1 (specifically
        #   ys = target+y for the quotient of symbol y).  For any
        #   earlier depth d < N we have d+1 < N+1, so depth-d DFA
        #   states are distinct frozensets from depth-N Q/R stops.
        #
        #   Therefore backward reachability from Q/R stops can never
        #   reach a state that only exists at depth d < N via a
        #   depth-d-only arc.  The backward BFS follows the merged
        #   incoming arcs, but the only paths that reach Q/R stops
        #   traverse the correct depth sequence 0 → 1 → … → N through
        #   the resume-frontier boundaries.  Depth-d arcs that lead
        #   into depth-d-only states are never visited because those
        #   states are not backward-reachable from the stops.
        #
        # Combined with forward reachability (every state in `incoming`
        # was discovered by BFS from start), the backward BFS from
        # stops produces exactly the trim machine.

        s = PeekabooState(self.fst, '', parent=None, univ=self._univ)
        # TODO: Revisit whether merging incoming dicts across depths is the
        # right approach vs. walking the PeekabooState chain on demand.
        merged_incoming = dict(s.incoming)
        for x in target:
            s >>= x
            for state, arcs in s.incoming.items():
                if state in merged_incoming:
                    merged_incoming[state] |= arcs
                else:
                    merged_incoming[state] = set(arcs)

        start_states = set(s.dfa.start())
        result = {}
        _empty = PrecoverDecomp(set(), set())
        for y in self.target_alphabet:
            d = s.decomp.get(y, _empty)
            q = _trimmed_fsa(start_states, d.quotient, merged_incoming)
            r = _trimmed_fsa(start_states, d.remainder, merged_incoming)
            result[y] = PrecoverDecomp(q, r)

        return result

    def graphviz(self, target):
        #
        # TODO: [2025-12-06 Sat] The funny thing about this picture is that the
        # "plates" are technically for the wrong target string.  specifically,
        # they are the precover of the next-target symbol extension of the
        # target target context thus, we have in each plate the *union* of the
        # precovers.
        #
        # TODO: use the integerizer here so that nodes are not improperly
        # equated with their string representations.
        #
        # TODO: show the active nodes in the graph of the outer most plate
        # (e.g., by coloring them yellow (#f2d66f), as in the precover
        # visualization); inactive node are white.  An additional option would
        # be to color the active vs. inactive edges differently as there is some
        # possibility of misinterpretation.
        #
        # TODO: another useful option would be for each plate to have "output
        # ports" for the nodes that should be expose to the next plate.  (I did
        # something like this in my dissertation code base, which was based on
        # using HTML tables inside node internals.)
        #
        from graphviz import Digraph

        def helper(target, outer):
            with outer.subgraph(name=f"cluster_{target}") as inner:
                inner.attr(label=target, style='rounded, filled', color='black', fillcolor='white')
                if target == '':
                    curr = PeekabooState(self.fst, '', parent=None)
                else:
                    curr = helper(target[:-1], inner) >> target[-1]
                for j, arcs in curr.incoming.items():
                    for x,i in arcs:
                        inner.edge(str(i), str(j), label=x)
                for y, parts in curr.decomp.items():
                    for j in parts.quotient:
                        inner.node(str(j), fillcolor='#90EE90')
                    for j in parts.remainder:
                        inner.node(str(j), fillcolor='#f26fec')
                return curr

        dot = Digraph(
            graph_attr=dict(rankdir='LR'),
            node_attr=dict(
                fontname='Monospace',
                fontsize='8',
                height='.05',
                width='.05',
                margin="0.055,0.042",
                shape='box',
                style='rounded, filled',
            ),
            edge_attr=dict(arrowsize='0.3', fontname='Monospace', fontsize='8'),
        )

        with dot.subgraph(name='outer') as outer:
            helper(target, outer)

        return dot

    def check(self, target):
        from arsenal import colors
        from transduction.eager_nonrecursive import Precover
        from transduction import display_table
        from IPython.display import HTML

        Have = self(target)

        for y in self.target_alphabet:

            want = Precover(self.fst, target + y)
            have = Have[y]

            q_ok = have.quotient.equal(want.quotient)
            r_ok = have.remainder.equal(want.remainder)

            if q_ok and r_ok:
                print(colors.mark(True), 'sym:', repr(y))
            else:
                print(colors.mark(False), 'sym:', repr(y), 'q:', colors.mark(q_ok), 'r:', colors.mark(r_ok))
                display_table([
                    [HTML('<b>quotient</b>'), have.quotient, want.quotient],
                    [HTML('<b>remainder</b>'), have.remainder, want.remainder],
                ], headings=['', 'have', 'want'])


class PeekabooState:

    def __init__(self, fst, target, parent, *, univ=None):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        self.target = target
        self.parent = parent

        assert parent is None or parent.target == target[:-1]

        self._univ = parent._univ if parent is not None else (univ or FstUniversality(fst))

        dfa = PeekabooPrecover(self.fst, target).det()

        if len(target) == 0:
            assert parent is None
            worklist = deque()
            incoming = {}
            for state in dfa.start():
                worklist.append(state)
                incoming[state] = set()

        else:

            resume_states = parent.resume_frontiers.get(target[-1], set())

            # put previous and Q and R states on the worklist
            worklist = deque()
            incoming = {}
            for state in resume_states:
                assert not any(truncated for _, _, truncated in state)
                worklist.append(state)
                incoming[state] = set()

        # `decomp` is a map from next target symbols to their quotient and
        # remainder states.  Built lazily — only symbols that actually appear
        # as relevant during the BFS get entries.
        decomp = {}

        # `resume_frontiers` maps each next-target symbol y to the set of
        # non-truncated DFA states on the truncation boundary, from which
        # the next PeekabooState (for target+y) must resume BFS expansion.
        resume_frontiers = {}

        _all_input_universal = self._univ.all_input_universal
        if _all_input_universal:
            def ensure_symbol(y):
                if y not in decomp:
                    decomp[y] = PrecoverDecomp(set(), set())
                    resume_frontiers[y] = set()
        else:
            def ensure_symbol(y):
                if y not in decomp:
                    decomp[y] = PrecoverDecomp(set(), set())
                    resume_frontiers[y] = set()
                    trunc_dfa = TruncatedDFA(dfa=dfa, fst=self.fst, target=target + y)
                    truncated_dfas[y] = trunc_dfa
                    univ_filters[y] = self._univ.make_filter(
                        self.fst, target + y, trunc_dfa, self.source_alphabet,
                    )

        truncated_dfas = {}
        univ_filters = {}

        N = len(target)
        _fst_is_final = self.fst.is_final

        # DFA states where the source has fully produced `target` and the
        # FST is in a final state — needed for P(EOS | target_so_far).
        preimage_stops = set()

        while worklist:
            state = worklist.popleft()

            # Single pass over NFA states: extract relevant symbols,
            # final-for-symbol set, and the truncated flag.
            relevant_symbols = set()
            final_symbols = set()
            state_has_truncated = False
            state_is_preimage = False
            for i, ys, truncated in state:
                if len(ys) == N and _fst_is_final(i):
                    state_is_preimage = True
                if len(ys) > N:
                    y = ys[N]
                    relevant_symbols.add(y)
                    if ys.startswith(target) and _fst_is_final(i):
                        final_symbols.add(y)
                state_has_truncated = state_has_truncated or truncated

            if state_is_preimage:
                preimage_stops.add(state)

            # A state is "continuous" (universal) for symbol y if it accepts
            # all source strings — meaning Q covers everything and no further
            # BFS expansion is needed for that symbol.  At most one of the
            # `relevant_symbols` can be continuous; if we find one we can
            # stop expanding.
            #
            # Proof (functional FSTs): Suppose state S is universal for
            # both y and z (y != z).  TruncatedDFA(target+y) recognises
            # precover(target+y) (the buffer length N+1 = len(target+y)
            # suffices for an exact match).  Universality from S means
            # Reach(S)·Σ* ⊆ precover(target+y), and likewise for z.
            # For a functional FST each input has a unique output, so
            # precover(target+y) ∩ precover(target+z) = ∅.  Therefore
            # Reach(S)·Σ* ⊆ ∅, but Reach(S) is non-empty (S is on the
            # worklist), giving a contradiction.
            continuous = False
            for y in relevant_symbols:
                ensure_symbol(y)

                if not continuous:
                    # For AUI (all-input-universal) FSTs, universality ↔ finality (no filter needed).
                    is_univ = (
                        y in final_symbols if _all_input_universal
                        else univ_filters[y].is_universal(state)
                    )
                    if is_univ:
                        decomp[y].quotient.add(state)
                        continuous = True
                        continue

                if y in final_symbols:
                    decomp[y].remainder.add(state)

            if continuous:
                continue    # we have found a quotient and can skip

            for x, next_state in dfa.arcs(state):

                if next_state not in incoming:
                    worklist.append(next_state)
                    incoming[next_state] = set()

                incoming[next_state].add((x, state))

                # If `state` is non-truncated but `next_state` contains a
                # truncated element, then `state` sits on the truncation
                # boundary — record it so the next iteration can resume here.
                if not state_has_truncated:
                    for _, ys, truncated in next_state:
                        if truncated:
                            y = ys[-1]
                            ensure_symbol(y)
                            resume_frontiers[y].add(state)

        # Q and R states that are non-truncated also sit on the boundary:
        # the next iteration needs to expand from them because they are
        # "live" endpoints of the current precover.
        for y in decomp:
            for state in decomp[y].quotient | decomp[y].remainder:
                if not any(truncated for _, ys, truncated in state):
                    resume_frontiers[y].add(state)

        self.decomp = decomp
        self.resume_frontiers = resume_frontiers
        self.dfa = dfa
        self.incoming = incoming
        self.preimage_stops = preimage_stops

    def __rshift__(self, y):
        assert y in self.target_alphabet, repr(y)
        return PeekabooState(self.fst, self.target + y, parent=self)


class TruncatedDFA(Lazy):
    """Augments a determinized `PeekabooPrecover` semi-automaton with an `is_final` method,
    producing a valid FSA that encodes `Precover(fst, target)`.

    Invariant: for all target strings,
        `TruncatedDFA(dfa=dfa, fst=fst, target=target).materialize().equal(Precover(fst, target).dfa)`
    """

    def __init__(self, *, dfa, fst, target):
        self.dfa = dfa
        self.fst = fst
        self.target = target

    def start(self):
        return self.dfa.start()

    # Refine is not required for correctness — the TruncatedDFA recognizes
    # the same language without it.  It is a normalization step that removes
    # NFA states committed to a different next symbol (e.g., buffer target+z
    # when we're computing Precover(target+y) for z ≠ y).  These states are
    # "passengers" that don't affect is_final results, but they inflate
    # powerset states (more elements to hash/compare) and generate irrelevant
    # successors during the universality sub-BFS.  In practice the impact is
    # unclear: most universality checks short-circuit via witnesses or the
    # monotonicity cache before reaching the sub-BFS.
    def refine(self, frontier):
        """Clip buffer strings to target length and filter to prefix-compatible
        states.  This normalization ensures equivalent DFA states are identified."""
        N = len(self.target)
        return frozenset(
            (i, ys[:N], truncated) for i, ys, truncated in frontier
            if ys[:min(N, len(ys))] == self.target[:min(N, len(ys))]
        )

    def arcs(self, state):
        for x, next_state in self.dfa.arcs(state):
            yield x, self.refine(next_state)

    def arcs_x(self, state, x):
        for next_state in self.dfa.arcs_x(state, x):
            yield self.refine(next_state)

    def is_final(self, state):
        return any(ys.startswith(self.target) and self.fst.is_final(i) for (i, ys, _) in state)
