"""
Staging ground for vibe-coded method, which need to be properly vetted
before being incorporated into the libary.
"""
from transduction.fsa import EPSILON

# ---------------------------------------------
# Edge-label compressor (regex-style summaries)
# ---------------------------------------------
import unicodedata

def _escape_char(c):
    specials = {']', '\\', '^', '-', '"'}
    return '\\' + c if c in specials else c

def _char_ranges(chars):
    cps = sorted(ord(c) for c in chars)
    if not cps:
        return
    # group consecutive codepoints
    start = prev = cps[0]
    for k in cps[1:]:
        if k != prev + 1:
            yield chr(start), chr(prev)
            start = k
        prev = k
    yield chr(start), chr(prev)

def _range_tokens(chars):
    toks = []
    for a, b in _char_ranges(chars):
        if ord(b) - ord(a) + 1 >= 3:
            toks.append(f"{_escape_char(a)}-{_escape_char(b)}")
        elif a == b:
            toks.append(_escape_char(a))
        else:
            for cp in range(ord(a), ord(b)+1):
                toks.append(_escape_char(chr(cp)))
    return toks

def _ascii_buckets():
    import string
    return [
        ("DIGIT", set(string.digits), "[0-9]"),
        ("LOWER", set(string.ascii_lowercase), "[a-z]"),
        ("UPPER", set(string.ascii_uppercase), "[A-Z]"),
        ("ALPHA", set(string.ascii_letters), "[A-Za-z]"),
        ("ALNUM", set(string.ascii_letters + string.digits), "[A-Za-z0-9]"),
        ("SPACE", set([' ', '\t']), r"[ \t]"),
        ("PRINTABLE", set(chr(i) for i in range(32, 127)), r"[ -~]"),
        ("PUNCT", set('!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), r"[[:punct:]]"),
    ]

def _unicode_bucket_name(c):
    return unicodedata.category(c)  # e.g., 'Lu', 'Ll', 'Nd'

def compress_symbols(symbols, alphabet=None, max_exclusions=5, use_unicode=False):
    """
    Turn a set/list of single-character symbols into a compact label string.
    If multi-character symbols are present, falls back to a literal union with braces.
    """
    S = set(symbols)
    assert len(S) > 0

    # If any symbol length != 1, fall back to literal listing (range compression is for single chars)
    if any(len(s) != 1 for s in S):
        items = sorted(S)
        if alphabet is not None and len(set(alphabet) - S) <= max_exclusions and len(set(alphabet) - S) < len(S):
            excl = sorted(set(alphabet) - S)
            excl_str = ", ".join(repr(x) for x in excl)
            return f"Σ − {{{excl_str}}}"
        return "{" + ", ".join(repr(x) for x in items) + "}"

    Σ = set(alphabet) if alphabet is not None else set(S)

    # Full set / small complement
    if S == Σ:
        return "Σ"
    if len(Σ - S) <= max_exclusions and len(Σ - S) < len(S):
        excl = sorted(Σ - S)
        excl_str = ", ".join(f'"{_escape_char(c)}"' for c in excl)
        return f"Σ − {{{excl_str}}}"

    candidates = []

    # ASCII buckets fully covered
    for _, bucket, name in _ascii_buckets():
        cov = bucket & Σ
        if cov and cov.issubset(S):
            candidates.append((cov, name, len(name)))

    # Unicode categories fully covered (optional)
    if use_unicode:
        cat_to_syms = {}
        for c in Σ:
            cat = _unicode_bucket_name(c)
            cat_to_syms.setdefault(cat, set()).add(c)
        for cat, bucket in cat_to_syms.items():
            if bucket.issubset(S):
                name = f"[[:{cat}:]]"
                candidates.append((bucket, name, len(name)))

    # ASCII ranges
    ascii_in_S = [c for c in S if ord(c) < 128]
    if ascii_in_S:
        toks = _range_tokens(ascii_in_S)
        if toks:
            name = "[" + "".join(toks) + "]"
            candidates.append((set(ascii_in_S), name, len(name)))

    uncovered = set(S)
    pieces = []
    while uncovered:
        best = None
        best_gain = 0.0
        for cov, name, cost in candidates:
            gain = len(uncovered & cov)
            if gain <= 0:
                continue
            score = gain / (cost + 1e-6)
            if score > best_gain:
                best_gain = score
                best = (cov, name, cost)
        if best is None:
            c = sorted(uncovered)[0]
            pieces.append(f'"{_escape_char(c)}"')
            uncovered.remove(c)
        else:
            cov, name, _ = best
            pieces.append(name)
            uncovered -= cov

    # Regroup many literals back into one bracket if helpful
    literals = [p for p in pieces if p.startswith('"')]
    nonlits = [p for p in pieces if not p.startswith('"')]
    if len(literals) >= 3:
        lits = [p.strip('"') for p in literals]
        lit_toks = _range_tokens(lits)
        nonlits.append("[" + "".join(lit_toks) + "]")
        pieces = nonlits

    # Check complement again vs union length
    if alphabet is not None:
        union_len = sum(len(p) for p in pieces) + max(0, len(pieces)-1)
        if len(Σ - S) <= max_exclusions:
            excl = sorted(Σ - S)
            excl_str = ", ".join(f'"{_escape_char(c)}"' for c in excl)
            comp = f"Σ − {{{excl_str}}}"
            if len(comp) < union_len:
                return comp

    return " ∪ ".join(pieces)

# ------------------------------------------------
# Graphviz integration: merge & visualize edges
# ------------------------------------------------
def visualize_automaton(
    automaton,
    rankdir="LR",
    use_unicode=False,
    max_exclusions=5,
    epsilon_symbol="ε",
    node_attrs=None,
    edge_attrs=None,
):
    from graphviz import Digraph

    # Merge parallel edges (u, v) by symbol set
    from collections import defaultdict
    uv_to_syms = defaultdict(set)
    uv_to_eps = set()  # track presence of epsilon edges separately
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
    for s in automaton.start:
        ghost = f"__start_{s}"
        dot.node(ghost, label="", shape="point", width="0")
        dot.edge(ghost, str(s), label="")

    # Nodes (doublecircle for finals)
    for q in automaton.states:
        is_final = False
        if automaton.is_final(q):
            is_final = automaton.is_final(q)
        if is_final:
            dot.node(str(q), peripheries='2')
        else:
            dot.node(str(q))

    # First, epsilon-only edges
    for (u, v) in sorted(uv_to_eps):
        # If there are both epsilon and symbol edges u→v, we’ll make two edges.
        kwargs = {}
        dot.edge(str(u), str(v), label=epsilon_symbol, **kwargs)

    # Then symbol edges
    for (u, v), S in sorted(uv_to_syms.items()):
        label = compress_symbols(S, alphabet=automaton.syms - {EPSILON}, max_exclusions=max_exclusions, use_unicode=use_unicode)
        dot.edge(str(u), str(v), label=label)

    return dot
