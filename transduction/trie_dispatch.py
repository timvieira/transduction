"""
Prototype: Trie-dispatch precover + DFA decomposition.

This is a correctness-first Python prototype for a trie-specialized path that
can be locally dispatched inside the general precover determinization.
It is intentionally conservative: if the FST does not match the strict
token->bytes trie shape, it falls back to the standard PrecoverNFA behavior.

Intended for tinkering and test validation; performance is not the goal yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import atexit
from typing import Any, Iterable

from transduction.base import DecompositionResult
from transduction.fsa import EPSILON
from transduction.precover_nfa import PrecoverNFA
from transduction.precover_nfa import PeekabooLookaheadNFA
from transduction.peekaboo_incremental import FstUniversality, TruncatedDFA
from transduction.universality import UniversalityFilter
from transduction import FSA
from transduction.util import Integerizer
from collections import deque


@dataclass(frozen=True)
class TrieInfo:
    nodes: set[Any]
    children: dict[Any, dict[Any, Any]]          # state -> output symbol -> state
    eps_output_arcs: dict[Any, tuple[tuple[Any, Any], ...]]  # state -> (input, dest)


# Lightweight counters to measure trie dispatch usage.
_STATS = {
    "trie_states": 0,
    "fallback_states": 0,
    "trie_arcs": 0,
    "fallback_arcs": 0,
}
_TAG_STATS: dict[str, dict[str, int]] = {}
_CURRENT_TAG: str | None = None


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


def get_stats() -> dict[str, int]:
    return dict(_STATS)


def set_tag(tag: str | None) -> None:
    global _CURRENT_TAG
    _CURRENT_TAG = tag
    if tag is not None and tag not in _TAG_STATS:
        _TAG_STATS[tag] = {k: 0 for k in _STATS}


def get_tag_stats(tag: str) -> dict[str, int]:
    return dict(_TAG_STATS.get(tag, {k: 0 for k in _STATS}))


def _bump(key: str, amount: int = 1) -> None:
    _STATS[key] += amount
    if _CURRENT_TAG is not None:
        _TAG_STATS[_CURRENT_TAG][key] += amount


def _maybe_report_stats() -> None:
    if os.environ.get("TRIE_DISPATCH_STATS"):
        stats = get_stats()
        total_states = stats["trie_states"] + stats["fallback_states"]
        total_arcs = stats["trie_arcs"] + stats["fallback_arcs"]
        print("\n[trie_dispatch stats]")
        print(f"  trie_states: {stats['trie_states']} / {total_states}")
        print(f"  fallback_states: {stats['fallback_states']} / {total_states}")
        print(f"  trie_arcs: {stats['trie_arcs']} / {total_arcs}")
        print(f"  fallback_arcs: {stats['fallback_arcs']} / {total_arcs}")
        if _TAG_STATS:
            def ratio(n, d):
                return 0.0 if d == 0 else n / d
            rows = []
            for tag, r in _TAG_STATS.items():
                total_states = r["trie_states"] + r["fallback_states"]
                total_arcs = r["trie_arcs"] + r["fallback_arcs"]
                rows.append({
                    "test": tag,
                    "trie_states": r["trie_states"],
                    "total_states": total_states,
                    "trie_arcs": r["trie_arcs"],
                    "total_arcs": total_arcs,
                    "state_pct": ratio(r["trie_states"], total_states),
                    "arc_pct": ratio(r["trie_arcs"], total_arcs),
                })
            rows.sort(key=lambda x: x["state_pct"], reverse=True)
            print("\n[trie_dispatch stats table]")
            print("| test | trie_states | total_states | trie_arcs | total_arcs | trie_state_% | trie_arc_% |")
            print("|---|---:|---:|---:|---:|---:|---:|")
            for r in rows:
                print(
                    f"| {r['test']} | {r['trie_states']} | {r['total_states']} | "
                    f"{r['trie_arcs']} | {r['total_arcs']} | "
                    f"{r['state_pct']*100:.1f} | {r['arc_pct']*100:.1f} |"
                )
            out_path = os.environ.get("TRIE_DISPATCH_STATS_FILE", "output/trie_dispatch_stats.md")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("## Trie Dispatch Stats\n\n")
                    f.write("| test | trie_states | total_states | trie_arcs | total_arcs | trie_state_% | trie_arc_% |\n")
                    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
                    for r in rows:
                        f.write(
                            f"| {r['test']} | {r['trie_states']} | {r['total_states']} | "
                            f"{r['trie_arcs']} | {r['total_arcs']} | "
                            f"{r['state_pct']*100:.1f} | {r['arc_pct']*100:.1f} |\n"
                        )
                print(f"[trie_dispatch stats table] wrote {out_path}")
            except OSError as exc:
                print(f"[trie_dispatch stats table] failed to write {out_path}: {exc}")


atexit.register(_maybe_report_stats)


def _infer_local_trie(fst) -> TrieInfo | None:
    """Detect trie-like structure *locally* (per-state), not globally.

    A state is trie-like if:
    - EPS-input arcs produce exactly one output symbol (non-EPSILON) with
      deterministic destinations per symbol.
    - Non-EPS input arcs may produce EPS or non-EPS output.
    - EPS-input arcs must produce a non-EPS output symbol.
    """
    nodes: set[Any] = set()
    children: dict[Any, dict[Any, Any]] = {}
    eps_output_arcs: dict[Any, list[tuple[Any, Any]]] = {}

    for i in fst.states:
        ok = True
        local_children: dict[Any, Any] = {}
        local_eps_output_arcs: list[tuple[Any, Any]] = []
        for a, b, j in fst.arcs(i):
            if a == EPSILON:
                if b == EPSILON:
                    ok = False
                    break
                # deterministic byte transition
                if b in local_children and local_children[b] != j:
                    ok = False
                    break
                local_children[b] = j
            else:
                if b == EPSILON:
                    local_eps_output_arcs.append((a, j))

        if ok:
            nodes.add(i)
            if local_children:
                children[i] = local_children
            if local_eps_output_arcs:
                eps_output_arcs[i] = local_eps_output_arcs

    if not nodes:
        return None

    eps_output_arcs_frozen = {k: tuple(v) for k, v in eps_output_arcs.items()}
    return TrieInfo(nodes=nodes, children=children, eps_output_arcs=eps_output_arcs_frozen)


class TrieDispatchPrecoverNFA(PrecoverNFA):
    """PrecoverNFA with a conservative trie-specialized arc path."""

    def __init__(self, fst, target):
        super().__init__(fst, target)
        self._trie = _infer_local_trie(fst)

    def arcs(self, state):
        if self._trie is None:
            _bump("fallback_states")
            yield from super().arcs(state)
            return

        (i, ys) = state
        if i not in self._trie.nodes:
            _bump("fallback_states")
            yield from super().arcs(state)
            return
        _bump("trie_states")

        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:
            return

        if m == self.N:
            for x, j in self.fst._arcs_all.get(i, ()):
                _bump("trie_arcs")
                yield (x, (j, self.target))
        else:
            for x, j in self.fst._arcs_by_output.get((i, EPSILON), ()):
                _bump("trie_arcs")
                yield (x, (j, ys))
            y = self.target[n]
            for x, j in self.fst._arcs_by_output.get((i, y), ()):
                _bump("trie_arcs")
                yield (x, (j, self.target[:n+1]))

    def arcs_x(self, state, x):
        if self._trie is None:
            yield from super().arcs_x(state, x)
            return

        (i, ys) = state
        if i not in self._trie.nodes:
            yield from super().arcs_x(state, x)
            return

        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:
            return

        if m == self.N:
            for y, j in self.fst._arcs_by_input.get((i, x), ()):
                yield (j, self.target)
        else:
            if x == EPSILON:
                y = self.target[n]
                child = self._trie.children.get(i, {}).get(y)
                if child is not None:
                    yield (child, self.target[:n+1])
                return
            for y, j in self.fst._arcs_by_input.get((i, x), ()):
                if y == EPSILON:
                    yield (j, ys)
                elif y == self.target[n]:
                    yield (j, self.target[:n+1])


class TrieDispatchPeekabooPrecover(PeekabooLookaheadNFA):
    """PeekabooLookaheadNFA with the same trie-dispatch logic as above."""

    def __init__(self, fst, target, K=1):
        super().__init__(fst, target, K=K)
        self._trie = _infer_local_trie(fst)

    def arcs(self, state):
        if self._trie is None:
            _bump("fallback_states")
            yield from super().arcs(state)
            return

        (i, ys, truncated) = state
        if i not in self._trie.nodes:
            _bump("fallback_states")
            yield from super().arcs(state)
            return
        _bump("trie_states")

        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:
            return

        if m >= self.N:
            if truncated:
                for x, j in self.fst._arcs_all.get(i, ()):
                    _bump("trie_arcs")
                    yield (x, (j, ys, True))
            else:
                for x, y, j in self.fst.arcs(i):
                    if y == EPSILON:
                        _bump("trie_arcs")
                        yield (x, (j, ys, False))
                    else:
                        was = ys + (y,)
                        now = was[:self.N + self.K]
                        _bump("trie_arcs")
                        yield (x, (j, now, (was != now)))
        else:
            assert not truncated
            y = self.target[n]
            for x, j in self.fst._arcs_by_output.get((i, EPSILON), ()):
                _bump("trie_arcs")
                yield (x, (j, ys, False))
            for x, j in self.fst._arcs_by_output.get((i, y), ()):
                _bump("trie_arcs")
                yield (x, (j, self.target[:n+1], False))

    def arcs_x(self, state, x):
        if self._trie is None:
            yield from super().arcs_x(state, x)
            return

        (i, ys, truncated) = state
        if i not in self._trie.nodes:
            yield from super().arcs_x(state, x)
            return

        n = len(ys)
        m = min(self.N, n)
        if self.target[:m] != ys[:m]:
            return

        if m >= self.N:
            if truncated:
                for y, j in self.fst._arcs_by_input.get((i, x), ()):
                    yield (j, ys, True)
            else:
                if x == EPSILON:
                    for y, child in self._trie.children.get(i, {}).items():
                        was = ys + (y,)
                        now = was[:self.N + self.K]
                        yield (child, now, (was != now))
                else:
                    for y, j in self.fst._arcs_by_input.get((i, x), ()):
                        if y == EPSILON:
                            yield (j, ys, False)
        else:
            if x == EPSILON:
                y = self.target[n]
                child = self._trie.children.get(i, {}).get(y)
                if child is not None:
                    yield (child, self.target[:n+1], False)
            else:
                for y, j in self.fst._arcs_by_input.get((i, x), ()):
                    if y == EPSILON:
                        yield (j, ys, False)
                    elif y == self.target[n]:
                        yield (j, self.target[:n+1], False)


class _PythonPeekabooClassify:
    def __init__(self, quotient_sym, remainder_syms, is_preimage, has_truncated, trunc_output_syms):
        self.quotient_sym = quotient_sym
        self.remainder_syms = remainder_syms
        self.is_preimage = is_preimage
        self.has_truncated = has_truncated
        self.trunc_output_syms = trunc_output_syms


class PythonLazyPeekabooDFAHelper:
    """Python stand-in for RustLazyPeekabooDFA.

    Uses the Peekaboo BFS logic (with trie-dispatch precover) to provide
    classify/arcs/run for FusedTransducedLM.
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
        self._target = ()

    def idx_to_sym_map(self):
        # Output alphabet only, in the Integerizer insertion order.
        out = [s for s in self._sym_map if s in self.fst.B and s != EPSILON]
        return [self._sym_map(s) for s in out]

    def new_step(self, target_u32):
        target = tuple(self._inv_sym[u] for u in target_u32)
        self._target = target
        self._classify_cache.clear()
        self._arcs_cache.clear()

        # Build DFA and peekaboo metadata using trie-dispatch precover.
        dfa = TrieDispatchPeekabooPrecover(self.fst, target).det()
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

        result = _PythonPeekabooClassify(
            quotient_sym=(self._sym_map(quotient_sym) if quotient_sym is not None else None),
            remainder_syms=[self._sym_map(y) for y in remainder_syms],
            is_preimage=is_preimage,
            has_truncated=has_truncated,
            trunc_output_syms=[self._sym_map(y) for y in trunc_output_syms],
        )
        self._classify_cache[sid] = result
        return result


class TrieDispatchDFADecomp(DecompositionResult):
    """Nonrecursive DFA decomposition using trie-dispatch precover.

    This is a drop-in alternative to NonrecursiveDFADecomp, intended as a
    prototype for local trie specialization. Correctness should match the
    reference Precover, and it can be used in test_general.py.
    """

    def __init__(self, fst, target):
        self.source_alphabet = fst.A - {EPSILON}
        self.target_alphabet = fst.B - {EPSILON}

        target = tuple(target)
        oov = set(target) - self.target_alphabet
        if oov:
            raise ValueError(f"Out of vocabulary target symbols: {oov}")

        dfa = TrieDispatchPrecoverNFA(fst, target).det()
        filt = UniversalityFilter(fst, target, dfa, self.source_alphabet)

        Q = FSA()
        R = FSA()

        worklist = deque()
        visited = set()

        for i in dfa.start():
            worklist.append(i)
            visited.add(i)
            Q.add_start(i)
            R.add_start(i)

        while worklist:
            i = worklist.popleft()

            if dfa.is_final(i):
                if filt.is_universal(i):
                    Q.add_stop(i)
                    continue
                else:
                    R.add_stop(i)

            for a, j in dfa.arcs(i):
                if j not in visited:
                    worklist.append(j)
                    visited.add(j)

                Q.add_arc(i, a, j)
                R.add_arc(i, a, j)

        self.fst = fst
        self.dfa = dfa
        self.target = target
        self.quotient = Q
        self.remainder = R
