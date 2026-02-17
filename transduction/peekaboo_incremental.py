from transduction.base import DecompositionResult, IncrementalDecomposition
from transduction.lazy import Lazy
from transduction.fsa import FSA, frozenset
from transduction.fst import EPSILON
from transduction.universality import (
    check_all_input_universal, compute_ip_universal_states,
    UniversalityFilter,
)
from transduction.precover_nfa import PeekabooLookaheadNFA as PeekabooPrecover

from collections import deque
from functools import cached_property

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

    def initial(self):
        """Return the initial PeekabooState (empty target)."""
        return PeekabooState(self.fst, (), parent=None, univ=self._univ)

    def __call__(self, target):
        """Decompose for every next target symbol after `target`.

        Returns {y: child} where child.quotient and child.remainder are FSAs.
        """
        return self.initial()(target).decompose_next()

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
        target = tuple(target)

        def helper(target, outer):
            label = ''.join(str(s) for s in target)
            with outer.subgraph(name=f"cluster_{label}") as inner:
                inner.attr(label=label, style='rounded, filled', color='black', fillcolor='white')
                if not target:
                    curr = PeekabooState(self.fst, (), parent=None)
                else:
                    curr = helper(target[:-1], inner) >> target[-1]
                for j, arcs in curr.incoming.items():
                    for x,i in arcs:
                        inner.edge(str(i), str(j), label=x)
                for _y, parts in curr.decomp.items():
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
        from transduction.util import colors
        from transduction.precover import Precover
        from transduction import display_table
        from IPython.display import HTML

        target = tuple(target)
        Have = self(target)

        for y in self.target_alphabet:

            want = Precover(self.fst, target + (y,))
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


class PeekabooState(IncrementalDecomposition):
    """Incremental peekaboo decomposition state.

    Peekaboo computes the quotient/remainder decomposition for *all* next
    target symbols simultaneously from a single truncated DFA.  The key
    insight is that the DFA for P(y) already contains enough information
    to determine Q(y·z) and R(y·z) for every z — we just need one extra
    symbol of lookahead in the NFA buffer.

    Construction is O(1) — the expensive BFS runs lazily on first access
    to ``.decomp``, ``.dfa``, ``.incoming``, ``.resume_frontiers``, or
    ``.preimage_stops``.  This mirrors the transformer LM pattern where
    ``>> y`` is a cheap state advance and the heavy compute happens when
    the results are actually needed (e.g., by ``decompose_next()`` or by
    ``TransducedLM._compute_logp_next()``).

    Lazy BFS Attributes
    -------------------
    These are computed on demand by ``_ensure_bfs()`` and cached:

    decomp : dict[symbol, DecompositionResult]
        Maps each next target symbol z to its Q/R stop states within the
        DFA.  ``decomp[z].quotient`` is the set of DFA states that are
        *universal* for z (every reachable source string belongs to Q(y·z)).
        ``decomp[z].remainder`` contains DFA states where the FST is final
        and the buffer matches y·z.

    dfa : PeekabooLookaheadNFA.det()
        The determinized precover NFA for the current target string.  DFA
        states are frozensets of NFA triples ``(fst_state, buffer, truncated)``.
        The buffer has at most ``N+K`` symbols (K=1 by default), giving a
        one-symbol lookahead beyond the current target.

    incoming : dict[state, set[(symbol, predecessor)]]
        Reverse-arc graph built during BFS.  Maps each DFA state to the
        set of ``(input_symbol, predecessor_state)`` pairs that reach it.
        Used by ``_trimmed_fsa()`` to extract the Q/R automata via
        backward reachability from stop states.

    resume_frontiers : dict[symbol, set[state]]
        Maps each next target symbol z to the set of non-truncated DFA
        states on the truncation boundary.  When we later compute
        ``self >> z``, the child's BFS starts from these states instead
        of the DFA's start states — this is the incremental reuse.
        States end up here in two ways: (1) they are non-truncated but
        have a truncated successor, or (2) they are Q/R stop states
        that are non-truncated (live endpoints of the current precover).

    preimage_stops : set[state]
        DFA states where the source string has produced exactly the
        current target ``y`` (buffer length = N) and the FST is in a
        final state.  These represent source strings whose transduction
        is exactly ``y`` — needed for computing P(EOS | y).

    Incremental Chain
    -----------------
    ``state >> z`` creates a child PeekabooState for target ``y·z``.  The
    child's BFS picks up from ``parent.resume_frontiers[z]`` rather than
    re-expanding from scratch.  The parent chain is walked to merge
    ``incoming`` dicts for FSA extraction (see ``_merged_incoming()``).
    """

    # Attributes computed by _ensure_bfs(); accessed lazily via __getattr__.
    _LAZY_BFS_ATTRS = frozenset({
        'decomp', 'dfa', 'resume_frontiers', 'preimage_stops', 'incoming',
    })

    def __init__(self, fst, target=(), parent=None, *, univ=None):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}
        target = tuple(target)
        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")
        self.target = target
        self.parent = parent

        assert parent is None or parent.target == target[:-1]

        self._univ = parent._univ if parent is not None else (univ or FstUniversality(fst))
        self._bfs_done = False

    def __getattr__(self, name):
        if name in PeekabooState._LAZY_BFS_ATTRS:
            self._ensure_bfs()
            return self.__dict__[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    def _ensure_bfs(self):
        """Run the peekaboo BFS on demand.  No-op if already completed."""
        if self._bfs_done:
            return

        target = self.target
        parent = self.parent

        dfa = PeekabooPrecover(self.fst, target).det()

        worklist = deque()
        incoming = {}
        if parent is not None:
            for state in parent.resume_frontiers.get(target[-1], set()):
                assert not any(truncated for _, _, truncated in state)
                worklist.append(state)
                incoming[state] = set()
        else:
            for state in dfa.start():
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
                    decomp[y] = DecompositionResult(set(), set())
                    resume_frontiers[y] = set()
        else:
            def ensure_symbol(y):
                if y not in decomp:
                    decomp[y] = DecompositionResult(set(), set())
                    resume_frontiers[y] = set()
                    trunc_dfa = TruncatedDFA(dfa=dfa, fst=self.fst, target=target + (y,))
                    truncated_dfas[y] = trunc_dfa
                    univ_filters[y] = self._univ.make_filter(
                        self.fst, target + (y,), trunc_dfa, self.source_alphabet,
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
                    if ys[:N] == target and _fst_is_final(i):
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
            # NOTE: This assumes the FST is functional.  A productive
            # input-epsilon cycle (eps-input arcs that produce non-epsilon
            # output) makes an FST non-functional, since the cycle can be
            # traversed any number of times yielding distinct outputs for
            # the same input.  Non-functional FSTs may violate the
            # uniqueness invariant below.
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
            continuous = None
            for y in relevant_symbols:
                ensure_symbol(y)

                # For AUI (all-input-universal) FSTs, universality ↔ finality (no filter needed).
                is_univ = (
                    y in final_symbols if _all_input_universal
                    else univ_filters[y].is_universal(state)
                )
                if is_univ:
                    if continuous is not None:
                        raise ValueError(
                            f"State is universal for both {continuous!r} and {y!r} — "
                            f"FST is likely non-functional (see FST.is_functional())"
                        )
                    decomp[y].quotient.add(state)
                    continuous = y
                    continue

                if y in final_symbols:
                    decomp[y].remainder.add(state)

            if continuous is not None:
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
        self._bfs_done = True

    def __rshift__(self, y):
        """Advance by one target symbol.  O(1) if decompose_next() was
        already called (cheap lookup); otherwise creates a deferred child
        whose BFS runs lazily on first attribute access."""
        assert y in self.target_alphabet, repr(y)
        if hasattr(self, '_children') and y in self._children:
            return self._children[y]
        return PeekabooState(self.fst, self.target + (y,), parent=self)

    def _merged_incoming(self):
        """Walk the parent chain and merge all incoming dicts into one.

        Cross-depth merging correctness argument
        ------------------------------------------
        Each depth d uses a different DFA (PeekabooPrecover with
        target[:d]).  At resume-frontier boundaries a single DFA state
        may have incoming arcs from *two* depths (d and d+1) with
        potentially different predecessors for the same input symbol,
        making the merged graph nondeterministic.  We argue the extra
        arcs from earlier depths are harmless:

          Structural invariant: at depth d the PeekabooPrecover NFA
          states have output buffers of length ≤ d+1 (= N+K with
          K=1).  DFA states are frozensets of such NFA states, so
          every DFA state reachable via depth-d arcs contains only
          NFA states with len(ys) ≤ d+1.

          The Q/R stop states live at the final depth N=len(target)
          and contain NFA states with len(ys) = N+1 (specifically
          ys = target+y for the quotient of symbol y).  For any
          earlier depth d < N we have d+1 < N+1, so depth-d DFA
          states are distinct frozensets from depth-N Q/R stops.

          Therefore backward reachability from Q/R stops can never
          reach a state that only exists at depth d < N via a
          depth-d-only arc.  The backward BFS follows the merged
          incoming arcs, but the only paths that reach Q/R stops
          traverse the correct depth sequence 0 → 1 → … → N through
          the resume-frontier boundaries.  Depth-d arcs that lead
          into depth-d-only states are never visited because those
          states are not backward-reachable from the stops.

        Combined with forward reachability (every state in `incoming`
        was discovered by BFS from start), the backward BFS from
        stops produces exactly the trim machine.
        """
        merged = dict(self.incoming)
        node = self.parent
        while node is not None:
            for state, arcs in node.incoming.items():
                if state in merged:
                    merged[state] |= arcs
                else:
                    merged[state] = set(arcs)
            node = node.parent
        return merged

    def decompose_next(self):
        """Returns {y: PeekabooState} for all next target symbols.

        Each child's ``.quotient`` and ``.remainder`` are computed on demand
        (cached after first access).  The child's own BFS is likewise
        deferred until its ``decompose_next()`` or attribute access triggers it.
        """
        if not hasattr(self, '_children'):
            self._children = {y: self >> y for y in self.target_alphabet}
        return self._children

    @cached_property
    def _qr(self):
        """Compute quotient and remainder FSAs from the parent's decomposition."""
        parent = self.parent
        assert parent is not None, "Root PeekabooState has no quotient/remainder"
        y = self.target[-1]
        _empty = DecompositionResult(set(), set())
        d = parent.decomp.get(y, _empty)
        merged_incoming = parent._merged_incoming()
        # Walk to root for start states
        node = parent
        while node.parent is not None:
            node = node.parent
        start_states = set(node.dfa.start())
        return (
            _trimmed_fsa(start_states, d.quotient, merged_incoming),
            _trimmed_fsa(start_states, d.remainder, merged_incoming),
        )

    @property
    def quotient(self):
        return self._qr[0]

    @property
    def remainder(self):
        return self._qr[1]


class TruncatedDFA(Lazy):
    """Augments a determinized PeekabooPrecover semi-automaton with ``is_final``.

    Used by ``PeekabooState`` (the incremental variant).  Like ``FilteredDFA``
    in ``peekaboo_nonrecursive``, but additionally applies ``refine()`` to each
    successor state: clips buffer strings to the target length and discards
    NFA elements whose buffer is incompatible with the target prefix.  This
    normalization ensures that equivalent DFA states are identified across
    incremental ``>>`` steps, preventing powerset blowup from "passenger"
    NFA elements committed to a different next symbol.

    Invariant: for all target strings,
        ``TruncatedDFA(dfa=dfa, fst=fst, target=target).materialize().equal(Precover(fst, target).dfa)``
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
        return any(ys[:len(self.target)] == self.target and self.fst.is_final(i) for (i, ys, _) in state)
