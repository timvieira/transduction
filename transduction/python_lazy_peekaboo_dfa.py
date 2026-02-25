"""Python fallback for RustLazyPeekabooDFA.

Provides the same classify/arcs/run interface used by FusedTransducedLM and
GeneralizedBeam when ``helper="python"``.  Uses the standard
PeekabooLookaheadNFA (no trie-dispatch specialization).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any

from transduction.base import DecompositionResult
from transduction.fsa import EPSILON
from transduction.precover_nfa import PeekabooLookaheadNFA
from transduction.peekaboo_incremental import FstUniversality, TruncatedDFA
from transduction.util import Integerizer, Str


@dataclass(frozen=True)
class PythonPeekabooClassify:
    quotient_sym: int | None
    remainder_syms: list[int]
    is_preimage: bool
    has_truncated: bool
    trunc_output_syms: list[int]


class PythonLazyPeekabooDFAHelper:
    """Python stand-in for RustLazyPeekabooDFA.

    Uses the Peekaboo BFS logic to provide classify/arcs/run for
    FusedTransducedLM and GeneralizedBeam.
    """

    def __init__(self, fst):
        self.fst = fst
        self._sym_map = Integerizer()
        # Pre-seed with input/output symbols (excluding EPSILON) for stable IDs.
        for s in (fst.A - {EPSILON}):
            self._sym_map(s)
        for s in (fst.B - {EPSILON}):
            self._sym_map(s)
        self._inv_sym = {v: k for k, v in self._sym_map.items()}

        self._univ = FstUniversality(fst)
        self._dfa = None
        self._decomp = None
        self._resume_frontiers = None
        self._preimage_stops = None
        self._classify_cache = {}
        self._arcs_cache = {}
        self._target: Str = ()

    def idx_to_sym_map(self):
        # Output alphabet only, in the Integerizer insertion order.
        out = [s for s in self._sym_map if s in self.fst.B and s != EPSILON]
        return [self._sym_map(s) for s in out]

    def new_step(self, target_u32):
        target = tuple(self._inv_sym[u] for u in target_u32)
        self._target = target
        self._classify_cache.clear()
        self._arcs_cache.clear()

        dfa = PeekabooLookaheadNFA(self.fst, target).det()
        self._dfa = dfa
        self._build_decomp(dfa, target)

    def _build_decomp(self, dfa, target):
        worklist = deque()
        incoming = {}

        for state in dfa.start():
            worklist.append(state)
            incoming[state] = set()

        decomp = {}
        resume_frontiers = {}

        _all_input_universal = self._univ.all_input_universal
        if _all_input_universal:
            def ensure_symbol(y):
                if y not in decomp:
                    decomp[y] = DecompositionResult(set(), set())
                    resume_frontiers[y] = set()
        else:
            truncated_dfas = {}
            univ_filters = {}

            def ensure_symbol(y):
                if y not in decomp:
                    decomp[y] = DecompositionResult(set(), set())
                    resume_frontiers[y] = set()
                    trunc_dfa = TruncatedDFA(dfa=dfa, fst=self.fst, target=target + (y,))
                    truncated_dfas[y] = trunc_dfa
                    univ_filters[y] = self._univ.make_filter(
                        self.fst, target + (y,), trunc_dfa, self.fst.A - {EPSILON},
                    )

        N = len(target)
        _fst_is_final = self.fst.is_final
        preimage_stops = set()

        while worklist:
            state = worklist.popleft()

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

            continuous = None
            for y in relevant_symbols:
                ensure_symbol(y)
                is_univ = (
                    y in final_symbols if _all_input_universal
                    else univ_filters[y].is_universal(state)
                )
                if is_univ:
                    if continuous is not None:
                        raise ValueError(
                            f"State is universal for both {continuous!r} and {y!r}"
                        )
                    decomp[y].quotient.add(state)
                    continuous = y
                    continue
                if y in final_symbols:
                    decomp[y].remainder.add(state)

            if continuous is not None:
                continue

            for x, next_state in dfa.arcs(state):
                if next_state not in incoming:
                    worklist.append(next_state)
                    incoming[next_state] = set()

                incoming[next_state].add((x, state))

                if not state_has_truncated:
                    for _, ys, truncated in next_state:
                        if truncated:
                            y = ys[-1]
                            ensure_symbol(y)
                            resume_frontiers[y].add(state)

        for y in decomp:
            for state in decomp[y].quotient | decomp[y].remainder:
                if not any(truncated for _, _, truncated in state):
                    resume_frontiers[y].add(state)

        self._decomp = decomp
        self._resume_frontiers = resume_frontiers
        self._preimage_stops = preimage_stops

    def start_ids(self):
        return list(self._dfa.start())

    def arcs(self, sid):
        cached = self._arcs_cache.get(sid)
        if cached is not None:
            return cached
        arcs = [(self._sym_map(x), j) for x, j in self._dfa.arcs(sid)]
        self._arcs_cache[sid] = arcs
        return arcs

    def single_arc(self, sid, x_u32):
        """Compute the DFA destination for a single input symbol."""
        for lbl, dest in self.arcs(sid):
            if lbl == x_u32:
                return dest
        return None

    def run(self, source_path):
        # source_path is list of u32; map back to symbols.
        path = [self._inv_sym[x] for x in source_path]
        [state] = list(self._dfa.start())
        for x in path:
            successors = list(self._dfa.arcs_x(state, x))
            if not successors:
                return None
            [state] = successors
        return state

    def classify(self, sid):
        cached = self._classify_cache.get(sid)
        if cached is not None:
            return cached

        quotient_sym = None
        remainder_syms = []
        for y, d in self._decomp.items():
            if sid in d.quotient:
                if quotient_sym is not None and quotient_sym != y:
                    raise ValueError("Multiple quotient symbols for one state")
                quotient_sym = y
            if sid in d.remainder:
                remainder_syms.append(y)

        is_preimage = sid in self._preimage_stops
        has_truncated = any(truncated for _, _, truncated in sid)
        trunc_output_syms = [
            y for y, states in self._resume_frontiers.items()
            if sid in states
        ]

        result = PythonPeekabooClassify(
            quotient_sym=(self._sym_map(quotient_sym) if quotient_sym is not None else None),
            remainder_syms=[self._sym_map(y) for y in remainder_syms],
            is_preimage=is_preimage,
            has_truncated=has_truncated,
            trunc_output_syms=[self._sym_map(y) for y in trunc_output_syms],
        )
        self._classify_cache[sid] = result
        return result
