"""Rho-arc compression for the determinized PrecoverNFA.

After determinization, the PrecoverNFA DFA often has states where many arcs
go to the same destination (e.g., for BPE FSTs, all ~50K token arcs at the
boundary return to the same powerset state).  SymbolicLazyDeterminize
detects these complete states and replaces the most common destination with
a single rho arc.

Terminology (OpenFST convention):
    RHO — "rest" label; matches any input symbol not on an explicit arc,
           consuming the input symbol.

Pipeline for exact compact precover DFA:
    PrecoverNFA(fst, t)
    → SymbolicLazyDeterminize(_, alphabet)
    → ExpandRho(_, alphabet)
    → .materialize()

Classes:
    SymbolicLazyDeterminize — subset construction with DFA-level rho factoring
    ExpandRho — expand rho arcs into explicit arcs (given alphabet)
"""

from collections import defaultdict

from transduction.lazy import Lazy, EPSILON


class _Rho:
    """Sentinel arc label: matches any input symbol not explicitly listed (consumes input)."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self):
        return 'RHO'
    def __hash__(self):
        return hash('RHO')
    def __eq__(self, other):
        return self is other

RHO = _Rho()


class SymbolicLazyDeterminize(Lazy):
    """Subset construction that factors complete DFA states into rho form.

    Standard subset construction, but when a resulting DFA state has arcs
    for every symbol in the given alphabet (i.e., is complete), the most
    common destination is replaced with a single RHO arc.  Incomplete DFA
    states are emitted with explicit arcs only.

    This is an exact transformation: the language is preserved.
    """

    def __init__(self, fsa, alphabet):
        self.fsa = fsa.epsremove()
        self.alphabet = frozenset(alphabet)

    def start(self):
        yield frozenset(self.fsa.start())

    def is_final(self, i):
        return any(self.fsa.is_final(q) for q in i)

    def arcs(self, i):
        # Try the memory-efficient path: compute destinations one symbol
        # at a time via arcs_x, deduplicating identical frozensets.  For
        # BPE boundary states this avoids building V large sets (O(V*K)
        # memory) and instead uses O(D*K) where D is distinct destinations
        # (typically 1-2 for BPE).
        #
        # If any alphabet symbol has no destination (incomplete state),
        # fall back to full by_symbol construction (cheap because
        # incomplete states have small powersets).
        fs_cache = {}          # frozenset -> frozenset (dedup)
        dfa_arcs = {}          # symbol -> frozenset
        complete = True

        for a in self.alphabet:
            dest = frozenset(j for q in i for j in self.fsa.arcs_x(q, a))
            if not dest:
                complete = False
                break
            if dest in fs_cache:
                dfa_arcs[a] = fs_cache[dest]
            else:
                fs_cache[dest] = dest
                dfa_arcs[a] = dest

        if not complete:
            # Incomplete state: fall back to full subset construction.
            # Incomplete states (growing phase) have small powersets, so
            # the naive approach is fast and memory-friendly.
            by_symbol = defaultdict(set)
            for q in i:
                for a, j in self.fsa.arcs(q):
                    by_symbol[a].add(j)
            for a in list(by_symbol):
                dests = by_symbol.pop(a)
                yield (a, frozenset(dests))
            return

        # Complete state: factor into RHO.
        # Group symbols by destination (identity-based, safe after dedup).
        by_dest_id = defaultdict(list)
        dest_by_id = {}
        for a, dest in dfa_arcs.items():
            did = id(dest)
            by_dest_id[did].append(a)
            dest_by_id[did] = dest

        if len(by_dest_id) == 1:
            yield (RHO, next(iter(dest_by_id.values())))
        else:
            rho_did = max(by_dest_id, key=lambda d: len(by_dest_id[d]))
            for did, syms in by_dest_id.items():
                if did != rho_did:
                    dest = dest_by_id[did]
                    for a in syms:
                        yield (a, dest)
            yield (RHO, dest_by_id[rho_did])

    def arcs_x(self, i, x):
        dests = set()
        for q in i:
            for j in self.fsa.arcs_x(q, x):
                dests.add(j)
        if dests:
            yield frozenset(dests)


class ExpandRho(Lazy):
    """Expand RHO arcs into explicit arcs for every symbol in the alphabet.

    Given a Lazy automaton with RHO arcs and a concrete alphabet, produces
    an equivalent automaton where each RHO arc is replaced by one explicit
    arc per symbol not already present.
    """

    def __init__(self, base, alphabet):
        self.base = base
        self.alphabet = set(alphabet)

    def start(self):
        return self.base.start()

    def is_final(self, i):
        return self.base.is_final(i)

    def arcs(self, i):
        rho_dests = []
        explicit_symbols = set()
        for a, j in self.base.arcs(i):
            if a is RHO:
                rho_dests.append(j)
            else:
                explicit_symbols.add(a)
                yield (a, j)
        for j in rho_dests:
            for x in self.alphabet:
                if x not in explicit_symbols:
                    yield (x, j)

    def arcs_x(self, i, x):
        return self.base.arcs_x(i, x)
