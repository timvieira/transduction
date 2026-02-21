"""
Generalized token-level decomposition for token-decomposable FSTs.

Works on any FST whose DFA transitions are determined by position sets alone
(token-decomposability). This includes BPE and PTB tokenizer FSTs.

Unlike TokenDecompose (which requires all_input_universal and hub structure),
this works on arbitrary TD FSTs by building the PrecoverNFA DFA but
quotienting by position sets during BFS — only one representative DFA state
per position set is expanded.

For PTB (174x compression): expands ~217 position sets instead of ~37,803
full DFA states for the full 45-symbol target.
"""

from collections import deque
from transduction.base import DecompositionResult
from transduction.fsa import FSA, EPSILON
from transduction.precover_nfa import PrecoverNFA
from transduction.universality import check_all_input_universal


def _positions(dfa_state):
    """Extract position set from a DFA state (frozenset of (fst_state, buf) pairs)."""
    return frozenset(len(buf) for (q, buf) in dfa_state)


class GeneralTokenDecompose(DecompositionResult):
    """
    Position-set decomposition for token-decomposable FSTs.

    Builds the PrecoverNFA DFA via BFS, but canonicalizes states by their
    position set. Only one representative DFA state per position set is
    expanded (arcs computed). For TD FSTs, this produces identical Q/R
    languages with potentially much fewer state expansions.

    Assumes the FST is token-decomposable. If not, the Q/R may be incorrect.
    Use reports/token_decomposability.py to verify TD before using this class.
    """

    def __init__(self, fst, target):
        self.fst = fst
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        target = tuple(target)
        self.target = target
        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        aiu = check_all_input_universal(fst)
        dfa = PrecoverNFA(fst, target).det()

        # Phase 1: BFS over position sets, expanding only one representative
        # per position set.
        canonical = {}   # pos_set -> representative DFA state (first seen)
        pos_arcs = {}    # pos_set -> {symbol: succ_pos_set}
        pos_final = {}   # pos_set -> bool

        worklist = deque()
        expanded = set()
        start_ps = None

        for i in dfa.start():
            ps = _positions(i)
            if ps not in canonical:
                canonical[ps] = i
                worklist.append(ps)
            start_ps = ps

        while worklist:
            ps = worklist.popleft()
            if ps in expanded:
                continue
            expanded.add(ps)

            rep = canonical[ps]
            is_final = dfa.is_final(rep)
            pos_final[ps] = is_final

            # For aiu FSTs, all final states are universal — prune here
            if aiu and is_final:
                pos_arcs[ps] = {}
                continue

            arcs = {}
            for a, j in dfa.arcs(rep):
                succ_ps = _positions(j)
                arcs[a] = succ_ps
                if succ_ps not in canonical:
                    canonical[succ_ps] = j
                    worklist.append(succ_ps)
                else:
                    # TD check: verify finality consistency
                    succ_final = dfa.is_final(j)
                    rep_final = dfa.is_final(canonical[succ_ps])
                    if succ_final != rep_final:
                        raise ValueError(
                            f"FST is not token-decomposable: finality "
                            f"mismatch at position set {succ_ps}"
                        )
            pos_arcs[ps] = arcs

        # Phase 2: Compute universality on the position-set DFA.
        # For aiu: trivial (all final states are universal).
        # For non-aiu: greatest-fixpoint on the position-set graph.
        if aiu:
            universal = {ps for ps in expanded if pos_final.get(ps, False)}
        else:
            universal = _compute_universality(
                expanded, pos_arcs, pos_final, self.source_alphabet
            )

        # Phase 3: Build Q and R using position sets as states.
        Q = FSA()
        R = FSA()
        Q.add_start(start_ps)
        R.add_start(start_ps)

        for ps in expanded:
            if pos_final[ps] and ps in universal:
                Q.add_stop(ps)
                continue  # don't expand past universal states in Q

            if pos_final[ps]:
                R.add_stop(ps)  # non-universal final -> R stop

            for a, succ_ps in pos_arcs.get(ps, {}).items():
                Q.add_arc(ps, a, succ_ps)
                R.add_arc(ps, a, succ_ps)

        self.quotient = Q
        self.remainder = R
        self.n_position_sets = len(expanded)
        self.n_canonical = len(canonical)


def _compute_universality(states, arcs, final, source_alphabet):
    """Greatest fixpoint: find position sets that accept source_alphabet*.

    A position set ps is universal iff:
    1. It is final
    2. For every source symbol a, arcs[ps][a] exists and is also universal

    This is the greatest fixpoint of these constraints.
    """
    # Initial candidates: final states with arcs for every source symbol
    candidates = set()
    for ps in states:
        if not final.get(ps, False):
            continue
        state_arcs = arcs.get(ps, {})
        if all(a in state_arcs for a in source_alphabet):
            candidates.add(ps)

    changed = True
    while changed:
        changed = False
        to_remove = set()
        for ps in candidates:
            state_arcs = arcs.get(ps, {})
            for a in source_alphabet:
                if state_arcs.get(a) not in candidates:
                    to_remove.add(ps)
                    break
        if to_remove:
            candidates -= to_remove
            changed = True

    return candidates
