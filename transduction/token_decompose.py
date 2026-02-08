"""
Token-level decomposition for BPE-like FSTs.

**EXPERIMENTAL** — This module is a specialized optimization that only applies
to FSTs satisfying the ``all_input_universal`` property (see
``fst.check_all_input_universal``). It is not a general-purpose decomposition
algorithm. See ``reports/token_decompose_applicability.ipynb`` for a detailed
analysis of which FST classes qualify.

Instead of tracking (fst_state, buf_pos) pairs through intermediate states,
this module collapses each token into a single transition that advances buf_pos
by the token's byte length. DFA states are position subsets {0..target_len}
instead of NFA state subsets over (fst_state x buf_pos).

For a target of length N, the resulting DFA typically has O(N) states instead of
the O(|fst_states| * N) states produced by the generic approach.
"""

from collections import defaultdict, deque
from transduction.base import DecompositionResult
from transduction.fsa import FSA, EPSILON
from transduction.universality import check_all_input_universal


def extract_token_bytes(fst):
    """
    Extract (token_id, byte_sequence) pairs from a BPE-like FST.

    Assumes the FST has a hub structure: start state(s) with non-epsilon input
    arcs leading into chains of epsilon-input arcs that return to a start state.

    Returns a list of (token_id, (byte, ...)) tuples.
    """
    start_set = set(fst.start)
    tokens = []

    for start in fst.start:
        for a, b, j in fst.arcs(start):
            if a == EPSILON:
                continue

            token_id = a
            bytes_out = []
            if b != EPSILON:
                bytes_out.append(b)

            # Follow the epsilon-input chain back to a start state
            current = j
            while current not in start_set:
                found = False
                for a2, b2, j2 in fst.arcs(current):
                    if a2 == EPSILON:
                        if b2 != EPSILON:
                            bytes_out.append(b2)
                        current = j2
                        found = True
                        break
                if not found:
                    break

            tokens.append((token_id, tuple(bytes_out)))

    return tokens


class ByteTrie:
    """Trie over byte sequences for fast prefix matching."""

    def __init__(self):
        self.children = [{}]       # node_id -> {byte: child_node_id}
        self.completions = [[]]    # node_id -> [(token_id, byte_length)]

    def insert(self, token_id, byte_seq):
        node = 0
        for b in byte_seq:
            if b in self.children[node]:
                node = self.children[node][b]
            else:
                next_id = len(self.children)
                self.children.append({})
                self.completions.append([])
                self.children[node][b] = next_id
                node = next_id
        self.completions[node].append((token_id, len(byte_seq)))

    def matches_at(self, target, pos):
        """
        Collect all tokens whose byte sequences match target[pos:].

        Two cases:
        1. Full match: the token's entire byte sequence fits within
           target[pos:]. advance = byte_length.
        2. Partial match: the token's byte sequence extends beyond the
           target. The first (target_len - pos) bytes match, and the
           remaining bytes are consumed post-target. advance = target_len - pos.
        """
        result = []
        node = 0
        target_len = len(target)

        for i in range(pos, target_len):
            if target[i] in self.children[node]:
                node = self.children[node][target[i]]
                result.extend(self.completions[node])
            else:
                return result

        # We've consumed all remaining target bytes. Tokens in the subtree
        # below `node` have byte sequences that START with target[pos:]
        # but extend further. Their extra bytes are consumed post-target.
        advance_cap = target_len - pos
        self._collect_subtree(node, advance_cap, result)

        return result

    def graphviz(self):
        """Render the trie as a graphviz Digraph."""
        from graphviz import Digraph
        import html as html_mod
        g = Digraph(
            graph_attr=dict(rankdir='TB'),
            node_attr=dict(
                fontname='Monospace', fontsize='9',
                height='.05', width='.05',
                margin='0.06,0.04', shape='circle', style='filled',
                fillcolor='white',
            ),
            edge_attr=dict(
                arrowsize='0.3', fontname='Monospace', fontsize='9',
            ),
        )
        for nid in range(len(self.children)):
            completions = self.completions[nid]
            if completions:
                labels = ', '.join(
                    html_mod.escape(repr(tid)) for tid, _blen in completions
                )
                g.node(str(nid), label=labels,
                       shape='box', style='filled,rounded',
                       fillcolor='#90EE90')
            else:
                g.node(str(nid), label='' if nid > 0 else 'root',
                       shape='circle')
            for byte_val, child in self.children[nid].items():
                g.edge(str(nid), str(child),
                       label=f' {html_mod.escape(repr(byte_val))} ')
        return g

    def _repr_mimebundle_(self, *args, **kwargs):
        return self.graphviz()._repr_mimebundle_(*args, **kwargs)

    def _collect_subtree(self, node, advance_cap, result):
        """Collect all tokens in the subtree below node."""
        for _byte, child in self.children[node].items():
            for (tid, _byte_len) in self.completions[child]:
                result.append((tid, advance_cap))
            self._collect_subtree(child, advance_cap, result)


def build_trie(fst):
    """Build a ByteTrie from an FST's extracted tokens."""
    token_list = extract_token_bytes(fst)
    trie = ByteTrie()
    for token_id, byte_seq in token_list:
        if byte_seq:
            trie.insert(token_id, byte_seq)
    return trie


class TokenDecompose(DecompositionResult):
    """
    Token-level decomposition for BPE-like FSTs where all_input_universal is True.

    DFA states are frozensets of positions {0..target_len} instead of the
    O(|fst_states| * target_len) NFA state space used by the generic approach.
    """

    def __init__(self, fst, target):
        assert check_all_input_universal(fst), \
            "TokenDecompose requires all_input_universal"

        self.fst = fst
        self.target = target
        target_len = len(target)

        # Extract tokens and build trie
        token_list = extract_token_bytes(fst)
        trie = ByteTrie()
        for (token_id, byte_seq) in token_list:
            if byte_seq:  # non-empty byte sequences go in trie
                trie.insert(token_id, byte_seq)

        # Precompute matches at each target position
        matches = []
        for p in range(target_len):
            matches.append(trie.matches_at(target, p))

        # Collect zero-length token IDs (e.g., delete_b's b->eps)
        zero_len_tokens = [tid for (tid, bs) in token_list if not bs]

        # BFS over position sets
        Q = FSA()
        R = FSA()

        worklist = deque()
        visited = {}  # frozenset -> state_id (we use frozensets directly as states)

        start_state = frozenset({0})
        worklist.append(start_state)
        visited[start_state] = start_state
        Q.add_start(start_state)
        R.add_start(start_state)

        while worklist:
            state = worklist.popleft()

            # States containing target_len are final.
            # Since all_input_universal, all final states are universal -> Q stops.
            if target_len in state:
                Q.add_stop(state)
                continue  # don't expand further

            # Group successor positions by token_id
            by_token = defaultdict(set)
            for p in state:
                if p < target_len:
                    for (tid, advance) in matches[p]:
                        new_pos = p + advance
                        if new_pos <= target_len:
                            by_token[tid].add(new_pos)

            # Zero-length tokens create self-loops
            for tid in zero_len_tokens:
                Q.add_arc(state, tid, state)
                R.add_arc(state, tid, state)

            for token_id, succ_positions in by_token.items():
                if not succ_positions:
                    continue

                succ_state = frozenset(succ_positions)
                Q.add_arc(state, token_id, succ_state)
                R.add_arc(state, token_id, succ_state)

                if succ_state not in visited:
                    visited[succ_state] = succ_state
                    worklist.append(succ_state)

        self.quotient = Q
        self.remainder = R
