"""
Staging ground for vibe-coded methods, which need to be properly vetted
before being incorporated into the library.
"""
import html as _html
from collections import defaultdict

from transduction.fsa import EPSILON


# ---------------------------------------------
# Edge-label compressor (regex-style summaries)
# ---------------------------------------------

def _fmt_symbol(s):
    """Format a single symbol for display.  Strings are shown bare, other
    types use ``repr()``."""
    return str(s) if isinstance(s, str) else repr(s)


def _escape_char(c):
    """Backslash-escape characters that are special inside bracket expressions."""
    specials = {']', '\\', '^', '-'}
    return '\\' + c if c in specials else c


def _char_ranges(chars):
    """Yield ``(lo, hi)`` pairs for runs of consecutive codepoints in *chars*."""
    cps = sorted(ord(c) for c in chars)
    if not cps:
        return
    start = prev = cps[0]
    for k in cps[1:]:
        if k != prev + 1:
            yield chr(start), chr(prev)
            start = k
        prev = k
    yield chr(start), chr(prev)


def _range_tokens(chars):
    """Convert a set of characters into compact regex-style range tokens.

    Runs of 3+ consecutive codepoints become ``a-z`` style ranges; shorter
    runs are listed individually.
    """
    toks = []
    for a, b in _char_ranges(chars):
        if ord(b) - ord(a) + 1 >= 3:
            toks.append(f"{_escape_char(a)}-{_escape_char(b)}")
        elif a == b:
            toks.append(_escape_char(a))
        else:
            for cp in range(ord(a), ord(b) + 1):
                toks.append(_escape_char(chr(cp)))
    return toks


def _format_complement(S, alphabet, max_exclusions, alphabet_name="Σ"):
    """Try to express *S* as ``Σ`` or ``Σ − {exclusions}``.

    Returns a string if the complement is compact, otherwise ``None``.
    *alphabet_name* controls the symbol used (e.g. ``"A"`` for FST input,
    ``"B"`` for FST output).
    """
    if alphabet is None:
        return None
    sigma = set(alphabet)
    if S == sigma:
        return alphabet_name
    excl = sigma - S
    if 0 < len(excl) <= max_exclusions and len(excl) < len(S):
        excl_str = ", ".join(_fmt_symbol(x) for x in sorted(excl, key=repr))
        return f"{alphabet_name} − {{{excl_str}}}"
    return None


def _format_char_class(chars):
    """Format single-character symbols as a bracket expression like ``[a-zA-Z0-9]``."""
    if len(chars) == 1:
        return _escape_char(next(iter(chars)))
    toks = _range_tokens(chars)
    return "[" + "".join(toks) + "]"


def compress_symbols(symbols, alphabet=None, max_exclusions=5,
                     alphabet_name="Σ"):
    """Turn a set of symbols into a compact label string.

    Tries, in order:
    1. Complement notation (e.g. ``Σ − {excl}``) when the excluded set is small
    2. Bracket expression with ranges for single-character symbols: ``[a-z0-9]``
    3. Literal set notation for everything else: ``{'ab', 'cd'}``

    *alphabet_name* controls the symbol used in complement notation
    (default ``"Σ"``; use ``"A"``/``"B"`` for FST input/output).
    """
    S = set(symbols)
    assert S

    # Complement notation (works for any symbol type)
    comp = _format_complement(S, alphabet, max_exclusions, alphabet_name)
    if comp is not None:
        return comp

    # Bracket expression only for sets of single characters
    if all(isinstance(s, str) and len(s) == 1 for s in S):
        return _format_char_class(S)

    # Everything else: literal set notation
    return "{" + ", ".join(_fmt_symbol(x) for x in sorted(S, key=repr)) + "}"


# ---------------------------------------------------------
# FST label compression: (input, output) pair → compact label
# ---------------------------------------------------------

def _factor_pairs(pairs):
    """Decompose ``(a, b)`` pairs into a union of cartesian products.

    Groups by output symbol, then merges groups sharing the same input set.
    Returns a list of ``(input_set, output_set)`` tuples.

    Example: ``{(a,x), (a,y), (b,x), (b,y)}`` → ``[({a,b}, {x,y})]``
    """
    by_output = defaultdict(set)
    for a, b in pairs:
        by_output[b].add(a)

    input_set_to_outputs = defaultdict(set)
    for b, inputs in by_output.items():
        input_set_to_outputs[frozenset(inputs)].add(b)

    return [(inputs, outputs)
            for inputs, outputs in input_set_to_outputs.items()]


def compress_fst_labels(pairs, input_alphabet=None, output_alphabet=None,
                        max_exclusions=5):
    """Compress a set of ``(input, output)`` arc label pairs into a compact
    edge label string.

    - Identity arcs ``a:a`` are displayed without a colon (like FSA arcs)
      and compressed via ``compress_symbols``.
    - Deletion arcs ``a:ε`` and insertion arcs ``ε:b`` are grouped and
      compressed on the non-epsilon side.
    - Relational arcs ``a:b`` (a≠b, neither ε) are factored into a union
      of cartesian products via ``_factor_pairs``.
    """
    # Categorize pairs
    eps_out = set()    # ε:b  (insertion)
    in_eps = set()     # a:ε  (deletion)
    identity = set()   # a:a  (copy)
    relational = set() # a:b  where a ≠ b, neither ε

    for a, b in pairs:
        if a == EPSILON and b == EPSILON:
            pass  # pure ε:ε handled as epsilon edge by caller
        elif a == EPSILON:
            eps_out.add(b)
        elif b == EPSILON:
            in_eps.add(a)
        elif a == b:
            identity.add(a)
        else:
            relational.add((a, b))

    pieces = []

    # Identity arcs: display like FSA (no colon)
    if identity:
        pieces.append(compress_symbols(
            identity, alphabet=input_alphabet,
            max_exclusions=max_exclusions, alphabet_name="A",
        ))

    # Deletion arcs: compress(inputs):ε
    if in_eps:
        label = compress_symbols(
            in_eps, alphabet=input_alphabet,
            max_exclusions=max_exclusions, alphabet_name="A",
        )
        pieces.append(f'{label}:ε')

    # Insertion arcs: ε:compress(outputs)
    if eps_out:
        label = compress_symbols(
            eps_out, alphabet=output_alphabet,
            max_exclusions=max_exclusions, alphabet_name="B",
        )
        pieces.append(f'ε:{label}')

    # Relational arcs: factor into cartesian products
    if relational:
        for inputs, outputs in _factor_pairs(relational):
            in_label = compress_symbols(
                inputs, alphabet=input_alphabet,
                max_exclusions=max_exclusions, alphabet_name="A",
            )
            out_label = compress_symbols(
                outputs, alphabet=output_alphabet,
                max_exclusions=max_exclusions, alphabet_name="B",
            )
            pieces.append(f'{in_label}:{out_label}')

    return ', '.join(pieces)


# ------------------------------------------------
# Graphviz integration: merge & visualize edges
# ------------------------------------------------

def visualize_automaton(
    automaton,
    rankdir="LR",
    max_exclusions=5,
    epsilon_symbol="ε",
    node_attrs=None,
    edge_attrs=None,
    fmt_state=str,
    sty_node=lambda q: {},
):
    """Visualize a materialized FSA or FST using Graphviz.

    Parallel edges between the same pair of states are merged into a single
    edge with a compact label.  For FSAs, labels are compressed via
    ``compress_symbols``; for FSTs, via ``compress_fst_labels`` (identity
    arcs displayed without colons, relational arcs factored into cartesian
    products).

    Args:
        fmt_state: Callable mapping a state to its display label (default ``str``).
        sty_node: Callable mapping a state to a dict of extra Graphviz node
            attributes (e.g. ``{'fillcolor': '#90EE90', 'style': 'filled,rounded'}``).
    """
    from graphviz import Digraph

    is_fst = hasattr(automaton, 'B')   # FSTs have .A and .B alphabets

    # Assign unique DOT node IDs via enumeration.  Using str(q) as a node ID
    # is unsafe: str(1) == str('1'), so mixed-type states would silently merge.
    node_id = {}
    for i, q in enumerate(automaton.states):
        node_id[q] = f"n{i}"

    # Collect and merge parallel edges
    uv_to_eps = set()

    if is_fst:
        uv_to_pairs = defaultdict(set)
        for u in automaton.states:
            for a, b, v in automaton.arcs(u):
                if a == EPSILON and b == EPSILON:
                    uv_to_eps.add((u, v))
                else:
                    uv_to_pairs[(u, v)].add((a, b))
        input_alpha = automaton.A - {EPSILON}
        output_alpha = automaton.B - {EPSILON}
    else:
        uv_to_syms = defaultdict(set)
        for u in automaton.states:
            for a, v in automaton.arcs(u):
                if a == EPSILON:
                    uv_to_eps.add((u, v))
                else:
                    uv_to_syms[(u, v)].add(a)

    # Graphviz setup
    _node_attrs = dict(
        fontname='Monospace',
        fontsize='8',
        height='.05',
        width='.05',
        margin="0.055,0.042",
        shape='box',
        style='rounded',
    )
    _edge_attrs = dict(
        arrowsize='0.3',
        fontname='Monospace',
        fontsize='8',
    )
    if node_attrs: _node_attrs.update(node_attrs)
    if edge_attrs: _edge_attrs.update(edge_attrs)
    dot = Digraph(
        graph_attr=dict(rankdir=rankdir),
        node_attr=_node_attrs,
        edge_attr=_edge_attrs,
    )

    # Invisible entry arrow for each start state
    for q in automaton.start:
        ghost = f"__ghost_{node_id[q]}"
        dot.node(ghost, label="", shape="point", width="0")
        dot.edge(ghost, node_id[q])

    # Nodes
    for q in automaton.states:
        label = _html.escape(str(fmt_state(q)))
        sty = dict(peripheries='2' if automaton.is_final(q) else '1')
        sty.update(sty_node(q))
        dot.node(node_id[q], label=label, **sty)

    # Epsilon edges
    for u, v in uv_to_eps:
        dot.edge(node_id[u], node_id[v], label=epsilon_symbol)

    # Content edges
    if is_fst:
        for (u, v), pairs in uv_to_pairs.items():
            label = compress_fst_labels(
                pairs, input_alpha, output_alpha,
                max_exclusions=max_exclusions,
            )
            dot.edge(node_id[u], node_id[v], label=label)
    else:
        for (u, v), S in uv_to_syms.items():
            label = compress_symbols(
                S, alphabet=automaton.syms - {EPSILON},
                max_exclusions=max_exclusions,
            )
            dot.edge(node_id[u], node_id[v], label=label)

    return dot
