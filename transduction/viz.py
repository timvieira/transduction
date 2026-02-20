"""
Visualization/display utilities for automata (edge-label compression,
interactive Graphviz rendering, chain coalescing, HTML tables).
"""
import re
import html as _html
import json
import base64
import uuid
from collections import defaultdict
from graphviz import Digraph
from IPython.display import HTML, SVG, display

from transduction.fsa import EPSILON
from transduction.util import Integerizer


# ------------------------------------------------
# Simple graphviz renderer (used by FSA/FST.graphviz)
# ------------------------------------------------

def _render_graphviz(states, start, stop, arc_iter, fmt_node, fmt_edge, sty_node):
    """Shared graphviz renderer for FSA and FST.

    Args:
        states: set of states
        start: set of start states
        stop: set of stop/final states
        arc_iter: callable(state) yielding (label_data, dest) pairs
        fmt_edge: callable(src, label_data, dest) -> str
        fmt_node: callable(state) -> str
        sty_node: callable(state) -> dict of extra graphviz node attrs
    """

    g = Digraph(
        graph_attr=dict(rankdir='LR'),
        node_attr=dict(
            fontname='Monospace',
            fontsize='8',
            height='.05', width='.05',
            margin="0.055,0.042",
            shape='box',
            style='rounded',
        ),
        edge_attr=dict(
            arrowsize='0.3',
            fontname='Monospace',
            fontsize='8',
        ),
    )

    f = Integerizer()

    # Start pointers
    for i in start:
        start_id = f'<start_{i}>'
        g.node(start_id, label='', shape='point', height='0', width='0')
        g.edge(start_id, str(f(i)), label='')

    # Nodes
    for i in states:
        sty = dict(peripheries='2' if i in stop else '1')
        sty.update(sty_node(i))
        g.node(str(f(i)), label=_html.escape(str(fmt_node(i))), **sty)

    # Collect parallel-edge labels by (src, dst)
    by_pair = defaultdict(list)
    for i in states:
        for label_data, j in arc_iter(i):
            lbl = _html.escape(str(fmt_edge(i, label_data, j)))
            by_pair[(str(f(i)), str(f(j)))].append(lbl)

    # Emit one edge per (src, dst) with stacked labels
    for (u, v), labels in by_pair.items():
        g.edge(u, v, label='\n'.join(sorted(labels)))

    return g


# ------------------------------------------------
# HTML table utilities (display_table, format_table)
# ------------------------------------------------

def _pick_from_mimebundle(obj, prefer=(
    "text/html",
    "image/svg+xml",
    "image/png",
    "image/jpeg",
    "text/latex",
    "text/markdown",
    "application/json",
    "text/plain",
)):
    """Extract the best HTML-embeddable representation from an object's mimebundle.

    Queries ``obj._repr_mimebundle_()`` and returns an HTML fragment for the
    highest-priority MIME type found.  Returns ``None`` if no mimebundle is
    available.
    """
    bundle = None

    if hasattr(obj, "_repr_mimebundle_"):
        out = obj._repr_mimebundle_()
        if isinstance(out, tuple) and out and isinstance(out[0], dict):
            bundle = out[0]
        elif isinstance(out, dict):
            bundle = out

    if not isinstance(bundle, dict):
        return None

    for mime in prefer:
        if mime not in bundle:
            continue
        data = bundle[mime]

        if mime == "text/html":
            return str(data)
        if mime == "image/svg+xml":
            return str(data)
        if mime in ("image/png", "image/jpeg"):
            if isinstance(data, str):
                try:
                    base64.b64decode(data, validate=True)
                    b64 = data
                except Exception:  # pylint: disable=W0718
                    b64 = base64.b64encode(data.encode("utf-8")).decode("ascii")
            elif isinstance(data, (bytes, bytearray, memoryview)):
                b64 = base64.b64encode(bytes(data)).decode("ascii")
            else:
                b64 = base64.b64encode(str(data).encode("utf-8")).decode("ascii")
            return f'<img alt="" src="data:{mime};base64,{b64}" />'
        if mime == "text/latex":
            return f'\\[{str(data)}\\]'
        if mime == "text/markdown":
            return f'<pre>{_html.escape(str(data))}</pre>'
        if mime == "application/json":
            try:
                obj_json = json.loads(data) if isinstance(data, str) else data
                pretty = json.dumps(obj_json, indent=2, ensure_ascii=False)
            except Exception:       # pylint: disable=W0718
                pretty = str(data)
            return f'<pre>{_html.escape(pretty)}</pre>'
        if mime == "text/plain":
            return f'<pre>{_html.escape(str(data))}</pre>'
    return None


def _as_html_cell(x):
    """Convert an arbitrary object to an HTML fragment for embedding in a table cell.

    Tries, in order: unwrapping IPython ``HTML``/``SVG`` objects, mimebundle
    extraction, ``_repr_html_``/``_repr_svg_``/image repr methods, LaTeX repr,
    and finally falls back to ``html.escape(str(x))`` inside ``<pre>`` tags.
    """

    if isinstance(x, HTML):
        return x.data
    if isinstance(x, SVG):
        return x.data

    html_frag = _pick_from_mimebundle(x)
    if html_frag is not None:
        return html_frag

    for meth in ("_repr_html_", "_repr_svg_", "_repr_image_svg_xml"):
        if hasattr(x, meth):
            try:
                return getattr(x, meth)()
            except Exception:  # pylint: disable=W0718
                pass

    for meth, tag in (("_repr_png_", "image/png"), ("_repr_jpeg_", "image/jpeg")):
        if hasattr(x, meth):
            try:
                data = getattr(x, meth)()
                if data is not None:
                    if isinstance(data, str):
                        try:
                            base64.b64decode(data, validate=True)
                            b64 = data
                        except Exception:   # pylint: disable=W0718
                            b64 = base64.b64encode(data.encode("utf-8")).decode("ascii")
                    else:
                        b64 = base64.b64encode(data).decode("ascii")
                    return f'<img alt="" src="data:{tag};base64,{b64}" />'
            except Exception:   # pylint: disable=W0718
                pass

    if hasattr(x, "_repr_latex_"):
        try:
            return f'\\[{x._repr_latex_()}\\]'
        except Exception:   # pylint: disable=W0718
            pass

    return f'<pre>{_html.escape(str(x))}</pre>'


def format_table(rows, headings=None, column_styles=None, max_rows=None,
                 total=None):
    """Build an HTML ``<table>`` string from a list of rows.

    Each element in a row is converted to HTML via ``_as_html_cell``, so cells
    can contain IPython display objects (FSA/FST graphviz, images, HTML
    fragments, etc.) alongside plain values.

    When *max_rows* is set and the row count exceeds it, the overflow rows are
    rendered but hidden; a clickable "▸ N more" toggle reveals them in-place
    (no kernel round-trip).

    Args:
        rows: Iterable of row iterables.  Each element becomes one ``<td>``.
        headings: Optional column header labels.
        column_styles: Optional dict mapping column index to a CSS style string
            (e.g. ``{0: "text-align:left"}``).  Styles are merged with the
            default ``vertical-align:top`` on each ``<td>``.
        max_rows: If given, show at most this many rows initially; overflow
            rows are hidden behind a clickable toggle.
        total: Total item count for the toggle label.  Inferred from
            *rows* (which is materialized to a list) when not provided.
    """
    if column_styles is None:
        column_styles = {}

    rows = list(rows)
    if total is None:
        total = len(rows)
    truncated = max_rows is not None and len(rows) > max_rows

    def _render_row(row):
        cells = []
        for i, x in enumerate(row):
            style = "vertical-align:top"
            extra = column_styles.get(i)
            if extra:
                style += ";" + extra
            cells.append(f'<td style="{style}">{_as_html_cell(x)}</td>')
        return f'<tr>{"".join(cells)}</tr>', len(cells)

    head_html = ""
    ncols = 1
    if headings:
        ncols = len(headings)
        head_cells = "".join(f"<th>{h}</th>" for h in headings)
        head_html = f"<thead><tr>{head_cells}</tr></thead>"

    visible_html = []
    for row in rows[:max_rows] if truncated else rows:
        tr, nc = _render_row(row)
        ncols = max(ncols, nc)
        visible_html.append(tr)
    body_html = "<tbody>" + "".join(visible_html) + "</tbody>"

    overflow_html = ""
    if truncated:
        remaining = total - max_rows
        uid = uuid.uuid4().hex[:8]

        overflow_rows = []
        for row in rows[max_rows:]:
            tr, nc = _render_row(row)
            ncols = max(ncols, nc)
            overflow_rows.append(tr)
        overflow_html = (
            f'<tbody id="ft_o_{uid}" style="display:none">'
            + "".join(overflow_rows)
            + '</tbody>'
        )

        toggle_html = (
            f'<tbody><tr><td colspan="{ncols}" style="text-align:center;'
            f'padding:2px 6px">'
            f'<a id="ft_t_{uid}" style="color:#888;cursor:pointer;'
            f'text-decoration:none"'
            f' onclick="'
            f"var o=document.getElementById('ft_o_{uid}'),"
            f"t=document.getElementById('ft_t_{uid}');"
            f"if(o.style.display==='none'){{"
            f"o.style.display='';"
            f"t.textContent='\\u25be collapse'"
            f"}}else{{"
            f"o.style.display='none';"
            f"t.textContent='\\u25b8 {remaining} more'"
            f"}}"
            f'">&#x25b8; {remaining} more</a>'
            f'</td></tr></tbody>'
        )
        overflow_html += toggle_html

    return (
        '<table class="fmt-table" style="border-collapse:collapse;">'
        f"{head_html}{body_html}{overflow_html}"
        "</table>"
    )


def display_table(rows, **kwargs):
    """Render a table in a Jupyter notebook.

    Convenience wrapper that calls ``format_table`` and displays the result
    as an IPython ``HTML`` object.  Accepts the same keyword arguments as
    ``format_table`` (e.g. ``headings``).
    """
    display(HTML(format_table(rows, **kwargs)))


def render_logp_next_html(class_name, target, logp, logp_next, top_k=20):
    """Render an LM state's logp_next distribution as an HTML header + table.

    Shared by TransducedState and ReferenceTransducedState ``_repr_html_``.

    Args:
        class_name: Display name for the state class.
        target: Sequence of target symbols consumed so far.
        logp: Cumulative log probability.
        logp_next: Dict-like mapping token -> logp (must support ``.items()``).
        top_k: Maximum number of entries to show (sorted by descending probability).
    """
    import numpy as np
    target_str = ''.join(str(y) for y in target) if target else 'ε'
    header = (f'<b>{class_name}</b> '
              f'target={target_str!r}, logp={logp:.4f}<br>')
    items = sorted(logp_next.items(), key=lambda kv: -kv[1])
    rows = [[repr(y), f'{lp:.4f}', f'{np.exp(lp):.4f}'] for y, lp in items]
    return header + format_table(rows, headings=['Token', 'logp', 'p'],
                                 max_rows=top_k)


# ---------------------------------------------------------------------------
# Particle table rendering (shared by TransducedState / FusedTransducedState)
# ---------------------------------------------------------------------------

def _format_source_path(lm_state):
    """Format an LM state's source path for display."""
    path = lm_state.path()
    if not path:
        return 'ε'
    try:
        return bytes(path).decode('utf-8', errors='replace')
    except TypeError:
        return ''.join(str(s) for s in path)


def _format_nfa_element(fst_state, buf, truncated):
    """Format a single NFA element (fst_state, buffer, truncated) compactly."""
    if not buf:
        buf_str = 'ε'
    elif all(isinstance(b, str) and len(b) == 1 for b in buf):
        s = ''.join(buf)
        buf_str = repr(s) if len(s) <= 6 else repr(s[:5]) + '…'
    else:
        items = [str(b) for b in buf[:4]]
        buf_str = '(' + ','.join(items) + ('…' if len(buf) > 4 else '') + ')'
    trunc = '†' if truncated else ''
    return f'({fst_state}, {buf_str}{trunc})'


def _format_nfa_set(decoded_set):
    """Format a decoded NFA state set for compact display."""
    elements = sorted(decoded_set, key=lambda x: (repr(x[0]), x[1]))
    parts = [_format_nfa_element(*e) for e in elements]
    MAX = 4
    if len(parts) <= MAX:
        return '{' + ', '.join(parts) + '}'
    return '{' + ', '.join(parts[:MAX]) + f', …+{len(parts)-MAX}' + '}'


def render_particles_html(
    class_name,       # 'TransducedState' or 'FusedTransducedState'
    items,            # Particle list with .dfa_state, .lm_state, .log_weight
    target,           # list of target symbols
    logp,             # float
    *,
    decode_fn=None,             # callable(dfa_state) -> str (formatted NFA set)
    q_states=frozenset(),       # set of quotient DFA states
    r_states=frozenset(),       # set of remainder DFA states
    qr_builder=None,            # callable(y) -> (q_fsa, r_fsa)
    decomp=None,                # decomp dict for Q/R filter
    max_rows=50,                # max particle groups to show in the table
):
    """Render an HTML table for particles.

    Shared by TransducedState and FusedTransducedState _repr_html_.
    """
    import numpy as np
    from transduction.util import logsumexp

    target_str = ''.join(str(y) for y in target) if target else 'ε'
    header = (f'<b>{class_name}</b> '
              f'target={target_str!r}, K={len(items)}, '
              f'logp={logp:.4f}<br>')
    if not items:
        return header

    log_weights = np.array([p.log_weight for p in items])
    log_Z = logsumexp(log_weights)

    show_role = bool(q_states or r_states)

    # Group by (source_path, dfa_state)
    groups = {}
    for p in items:
        source = _format_source_path(p.lm_state)
        key = (source, p.dfa_state)
        groups.setdefault(key, []).append(p.log_weight)

    table_rows = []
    for (source, dfa_state), lws in groups.items():
        group_log_w = logsumexp(np.array(lws))
        posterior = np.exp(group_log_w - log_Z)
        if show_role:
            is_q = dfa_state in q_states
            is_r = dfa_state in r_states
            role = ('Q+R' if is_q and is_r else
                    'Q' if is_q else
                    'R' if is_r else 'frontier')
        else:
            role = None
        table_rows.append((source, dfa_state, role, len(lws),
                           group_log_w, posterior))
    table_rows.sort(key=lambda r: -r[5])

    rows = []
    for source, dfa_state, role, count, log_w, posterior in table_rows:
        dfa_label = decode_fn(dfa_state) if decode_fn else str(dfa_state)
        row = [repr(source), dfa_label]
        if show_role:
            row.append(role)
        row.extend([str(count), f'{log_w:.2f}', f'{posterior:.4f}'])
        rows.append(row)

    headings = ['Source prefix', 'DFA state']
    if show_role:
        headings.append('Role')
    headings.extend(['Count', 'log w', 'p(x|y)'])

    result = header + format_table(
        rows,
        headings=headings,
        column_styles={0: 'text-align:left'},
        max_rows=max_rows,
    )

    # Add collapsible Q/R FSA visualizations
    if decode_fn and qr_builder and decomp:
        particle_states = {p.dfa_state for p in items}

        def sty_node(state):
            if state in particle_states:
                return {'fillcolor': '#ADD8E6',
                        'style': 'filled,rounded'}
            return {}

        def fmt_node(state):
            return decode_fn(state)

        relevant_syms = set()
        for y, d in decomp.items():
            if d.quotient & particle_states or \
               d.remainder & particle_states:
                relevant_syms.add(y)

        MAX_QR = 10
        shown = 0
        for y in sorted(relevant_syms, key=repr):
            if shown >= MAX_QR:
                rest = len(relevant_syms) - shown
                result += (f'<details><summary>… and {rest} more '
                           f'Q/R sections</summary></details>')
                break
            try:
                q_fsa, r_fsa = qr_builder(y)
            except Exception:
                continue
            if not q_fsa.states and not r_fsa.states:
                continue

            parts = []
            y_label = _html.escape(repr(y))
            if q_fsa.states:
                try:
                    g = q_fsa.graphviz(fmt_node=fmt_node,
                                       sty_node=sty_node)
                    svg = g._repr_image_svg_xml()
                    parts.append(f'<b>Q({y_label})</b><br>{svg}')
                except Exception:
                    pass
            if r_fsa.states:
                try:
                    g = r_fsa.graphviz(fmt_node=fmt_node,
                                       sty_node=sty_node)
                    svg = g._repr_image_svg_xml()
                    parts.append(f'<b>R({y_label})</b><br>{svg}')
                except Exception:
                    pass
            if parts:
                content = '<br>'.join(parts)
                result += (f'<details><summary>Q/R for y={y_label}'
                           f'</summary>{content}</details>')
                shown += 1

    return result


# ---------------------------------------------
# Edge-label compressor (regex-style summaries)
# ---------------------------------------------

# Sentinels wrapping meta-brackets ([ ] { }) so the visualizer can color them
# differently from literal bracket characters in the alphabet.
_MB_OPEN = '\ue000'
_MB_CLOSE = '\ue001'

def _m(bracket):
    """Wrap a meta-bracket character in sentinels."""
    return f"{_MB_OPEN}{bracket}{_MB_CLOSE}"

def _fmt_symbol(s):
    """Format a single symbol for display.  Strings are shown bare (with
    space shown as ``␣``), length-1 bytes use the ``_fmt_byte`` style,
    other types use ``repr()``."""
    if isinstance(s, str):
        return s.replace(' ', '␣')
    if isinstance(s, bytes) and len(s) == 1:
        return _fmt_byte(s[0])
    if isinstance(s, int) and 0 <= s <= 255:
        return _fmt_byte(s)
    return repr(s)


def _escape_char(c):
    """Backslash-escape characters that are special inside bracket expressions.
    Space is rendered as ``␣``."""
    if c == ' ':
        return '␣'
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
        return _m(alphabet_name)
    excl = sigma - S
    if 0 < len(excl) <= max_exclusions and len(excl) < len(S):
        excl_str = ", ".join(_fmt_symbol(x) for x in sorted(excl, key=repr))
        return f"{_m(alphabet_name)} {_m('−')} {_m('{')}{excl_str}{_m('}')}"
    return None


def _format_char_class(chars):
    """Format single-character symbols as a bracket expression like ``[a-zA-Z0-9]``."""
    if len(chars) == 1:
        return _escape_char(next(iter(chars)))
    toks = _range_tokens(chars)
    return _m("[") + "".join(toks) + _m("]")


# -------------------------------------------------
# Integer-label range compression (analogous to char ranges)
# -------------------------------------------------

def _int_ranges(ints):
    """Yield ``(lo, hi)`` pairs for runs of consecutive integers in *ints*."""
    vals = sorted(ints)
    if not vals:
        return
    start = prev = vals[0]
    for k in vals[1:]:
        if k != prev + 1:
            yield start, prev
            start = k
        prev = k
    yield start, prev


def _int_range_tokens(ints):
    """Convert a set of integers into compact range tokens.

    Runs of 3+ consecutive integers become ``a-b`` style ranges; shorter
    runs are listed individually.
    """
    toks = []
    for a, b in _int_ranges(ints):
        if b - a + 1 >= 3:
            toks.append(f"{a}-{b}")
        else:
            for v in range(a, b + 1):
                toks.append(str(v))
    return toks


def _format_int_class(ints):
    """Format integer symbols as a bracket expression like ``[0-9, 32-126]``."""
    if len(ints) == 1:
        return str(next(iter(ints)))
    toks = _int_range_tokens(ints)
    return _m("[") + ", ".join(toks) + _m("]")


def _fmt_byte(b):
    """Format a byte value (0-255) for display in bracket expressions.

    Printable ASCII (0x20-0x7e) is shown as the character itself (with
    bracket-expression specials escaped, space as ``␣``); everything else
    uses ``\\xNN``.
    """
    if 0x20 <= b <= 0x7e:
        return _escape_char(chr(b))
    return f"\\x{b:02x}"


def _format_bytes_class(byte_syms):
    """Format a set of length-1 ``bytes`` as a bracket expression like ``[\\x00-\\x1f, a-z]``."""
    vals = sorted(b[0] for b in byte_syms)
    if len(vals) == 1:
        return _fmt_byte(vals[0])
    toks = []
    for a, b in _int_ranges(vals):
        if b - a + 1 >= 3:
            toks.append(f"{_fmt_byte(a)}-{_fmt_byte(b)}")
        else:
            for v in range(a, b + 1):
                toks.append(_fmt_byte(v))
    return _m("[") + "".join(toks) + _m("]")


def compress_symbols(symbols, alphabet=None, max_exclusions=5,
                     alphabet_name="Σ"):
    """Turn a set of symbols into a compact label string.

    Tries, in order:
    1. Complement notation (e.g. ``Σ − {excl}``) when the excluded set is small
    2. Bracket expression with ranges for single-character symbols: ``[a-z0-9]``
    3. Integer range compression: ``[0-9, 32-126]``
    4. Byte range compression for length-1 ``bytes``: ``[\\x00-\\x1f, a-z]``
    5. Literal set notation for everything else: ``{'ab', 'cd'}``

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

    # Integer range compression
    if all(isinstance(s, int) for s in S):
        return _format_int_class(S)

    # Byte range compression for length-1 bytes objects
    if all(isinstance(s, bytes) and len(s) == 1 for s in S):
        return _format_bytes_class(S)

    # Everything else: literal set notation
    return _m("{") + ", ".join(_fmt_symbol(x) for x in sorted(S, key=repr)) + _m("}")


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
# Expanded (uncompressed) label formatting
# ------------------------------------------------

def _expand_symbols(symbols, max_shown=50):
    """Format a full set of symbols as a comma-separated list (no range compression)."""
    syms = sorted(symbols, key=repr)
    parts = [_fmt_symbol(s) for s in syms[:max_shown]]
    text = ', '.join(parts)
    if len(syms) > max_shown:
        text += f', \u2026 ({len(syms)} total)'
    return text


def _expand_fst_pairs(pairs, max_shown=50):
    """Format FST label pairs as a comma-separated list (no factoring)."""
    sorted_pairs = sorted(pairs, key=repr)
    parts = []
    for a, b in sorted_pairs[:max_shown]:
        fa = 'ε' if a is EPSILON else _fmt_symbol(a)
        fb = 'ε' if b is EPSILON else _fmt_symbol(b)
        if a == b:
            parts.append(fa)
        else:
            parts.append(f'{fa}:{fb}')
    text = ', '.join(parts)
    if len(sorted_pairs) > max_shown:
        text += f', \u2026 ({len(sorted_pairs)} total)'
    return text


# ------------------------------------------------
# Graphviz integration: merge & visualize edges
# ------------------------------------------------

_META_BRACKET_COLOR = "#4a86c8"

def _gv_html_escape(text):
    """HTML-escape *text* for use inside a Graphviz HTML label.

    In addition to the standard ``&<>"`` escapes, ``]`` must be
    entity-encoded because Graphviz's HTML parser treats it as
    a DOT attribute-list terminator inside ``<font>`` tags.
    """
    return _html.escape(text).replace("]", "&#93;")


def _label_to_html(label, meta_color=_META_BRACKET_COLOR):
    """Convert a label with meta-bracket sentinels to a Graphviz HTML label.

    Characters between ``_MB_OPEN`` / ``_MB_CLOSE`` sentinels are rendered
    in *meta_color*; everything else is HTML-escaped normally.
    """
    parts = []
    i = 0
    while i < len(label):
        if label[i] == _MB_OPEN:
            j = label.index(_MB_CLOSE, i + 1)
            bracket = _gv_html_escape(label[i + 1 : j])
            parts.append(
                f'<font color="{meta_color}">{bracket}</font>'
            )
            i = j + 1
        else:
            parts.append(_gv_html_escape(label[i]))
            i += 1
    return "<" + "".join(parts) + ">"


def strip_meta_brackets(label):
    """Remove meta-bracket sentinels, returning a plain-text label."""
    return label.replace(_MB_OPEN, "").replace(_MB_CLOSE, "")


class InteractiveGraph:
    """Wrapper around a Graphviz ``Digraph`` that adds click-to-expand edge
    labels when displayed in a Jupyter notebook.

    - In Jupyter: ``_repr_html_`` renders SVG with JavaScript popups.
    - Elsewhere: ``_repr_svg_`` falls back to plain SVG.
    - Attribute access delegates to the underlying ``dot`` Digraph
      (e.g. ``.render()``, ``.pipe()``, ``.source``).
    """

    def __init__(self, dot, edge_labels):
        self.dot = dot
        self._edge_labels = edge_labels   # {edge_id: expanded_text}

    def __getattr__(self, name):
        return getattr(self.dot, name)

    def __repr__(self):
        return repr(self.dot)

    def _repr_image_svg_xml(self):
        return self.dot._repr_image_svg_xml()

    def _repr_html_(self):
        svg_raw = self.dot.pipe(format='svg').decode('utf-8')
        cid = f"ig_{uuid.uuid4().hex[:8]}"

        # Strip Graphviz's fixed width/height (in pt) so the SVG scales
        # naturally inside the viewport.  Keep the viewBox for aspect ratio.
        svg = re.sub(r'(<svg[^>]*?)\s+width="[^"]*"', r'\1', svg_raw, count=1)
        svg = re.sub(r'(<svg[^>]*?)\s+height="[^"]*"', r'\1', svg, count=1)

        labels_json = json.dumps(self._edge_labels)
        return (
            # -- container: clipped viewport -----------------------------------
            f'<div id="{cid}" style="position:relative;overflow:hidden;'
            f'  border:1px solid #ddd;border-radius:4px;'
            f'  width:100%;height:500px;cursor:grab">'
            f'<div class="ig-inner" style="transform-origin:0 0">'
            f'{svg}'
            f'</div></div>'

            # -- CSS -----------------------------------------------------------
            f'<style>'
            f'#{cid} .edge {{cursor:pointer}}'
            f'#{cid} .edge-popup {{'
            f'  position:absolute;background:#fffff8;border:1px solid #bbb;'
            f'  border-radius:4px;padding:4px 8px;font:11px/1.4 monospace;'
            f'  white-space:pre-wrap;max-width:500px;max-height:200px;'
            f'  overflow:auto;z-index:1000;box-shadow:2px 2px 6px rgba(0,0,0,.15);'
            f'}}'
            f'#{cid} .ig-hint {{'
            f'  position:absolute;bottom:6px;right:8px;font:10px sans-serif;'
            f'  color:#999;pointer-events:none'
            f'}}'
            f'</style>'

            # -- JavaScript: zoom, pan, click-to-expand ------------------------
            f'<script>'
            f'(function(){{'

            # --- refs ---------------------------------------------------------
            f'  var c=document.getElementById("{cid}"),'
            f'      inner=c.querySelector(".ig-inner"),'
            f'      svg=c.querySelector("svg"),'
            f'      L={labels_json};'

            # --- zoom / pan state ---------------------------------------------
            f'  var scale=1,tx=0,ty=0;'
            f'  function apply(){{inner.style.transform='
            f'    "translate("+tx+"px,"+ty+"px) scale("+scale+")"}}'

            # --- fit SVG to viewport on load ----------------------------------
            f'  svg.style.width="100%";svg.style.height="auto";'

            # --- wheel zoom toward cursor -------------------------------------
            f'  c.addEventListener("wheel",function(e){{'
            f'    e.preventDefault();'
            f'    var f=e.deltaY<0?1.1:1/1.1,'
            f'        r=c.getBoundingClientRect(),'
            f'        mx=e.clientX-r.left,my=e.clientY-r.top;'
            f'    tx=mx-f*(mx-tx);ty=my-f*(my-ty);'
            f'    scale*=f;apply()'
            f'  }},{{passive:false}});'

            # --- drag to pan (distinguish from click via movement) ------------
            f'  var dragging=false,dx,dy,moved;'
            f'  c.addEventListener("mousedown",function(e){{'
            f'    if(e.target.closest(".edge-popup"))return;'
            f'    dragging=true;moved=false;dx=e.clientX-tx;dy=e.clientY-ty;'
            f'    c.style.cursor="grabbing"'
            f'  }});'
            f'  document.addEventListener("mousemove",function(e){{'
            f'    if(!dragging)return;moved=true;'
            f'    tx=e.clientX-dx;ty=e.clientY-dy;apply()'
            f'  }});'
            f'  document.addEventListener("mouseup",function(){{'
            f'    dragging=false;c.style.cursor="grab"'
            f'  }});'

            # --- click-to-expand edge labels ----------------------------------
            f'  Object.keys(L).forEach(function(eid){{'
            f'    var g=svg.getElementById(eid);'
            f'    if(!g)return;'
            f'    g.addEventListener("click",function(e){{'
            f'      if(moved)return;'
            f'      e.stopPropagation();'
            f'      var old=c.querySelector(".edge-popup");'
            f'      if(old){{old.remove();if(old.dataset.eid===eid)return}}'
            f'      var r=c.getBoundingClientRect(),'
            f'          p=document.createElement("div");'
            f'      p.className="edge-popup";p.dataset.eid=eid;'
            f'      p.textContent=L[eid];'
            f'      p.style.left=(e.clientX-r.left+10)+"px";'
            f'      p.style.top=(e.clientY-r.top-10)+"px";'
            f'      c.appendChild(p)'
            f'    }})'
            f'  }});'

            # --- dismiss popup on outside click -------------------------------
            f'  document.addEventListener("click",function(e){{'
            f'    if(moved)return;'
            f'    if(!e.target.closest("#{cid} .edge")'
            f'       && !e.target.closest("#{cid} .edge-popup")){{'
            f'      var p=c.querySelector(".edge-popup");'
            f'      if(p)p.remove()'
            f'    }}'
            f'  }});'

            # --- hint ---------------------------------------------------------
            f'  var h=document.createElement("div");h.className="ig-hint";'
            f'  h.textContent="scroll to zoom \u00b7 drag to pan";'
            f'  c.appendChild(h)'

            f'}})();'
            f'</script>'
        )


def _coalesce_unbranching(uv_to_data, uv_to_eps, automaton):
    """Detect unbranching chains and return coalescing info.

    A state is *collapsible* if it is not start/final, has exactly one
    non-epsilon predecessor and one non-epsilon successor, and has no
    epsilon edges.  Maximal chains of collapsible states are merged into
    single edges.

    Returns ``(removed_states, removed_edges, chains)`` where *chains*
    is a list of ``((head, tail), [(u1,v1), ...])`` entries.
    """
    out_nbrs = defaultdict(set)
    in_nbrs = defaultdict(set)
    for u, v in uv_to_data:
        out_nbrs[u].add(v)
        in_nbrs[v].add(u)

    eps_out_count = defaultdict(int)
    eps_in_count = defaultdict(int)
    for u, v in uv_to_eps:
        eps_out_count[u] += 1
        eps_in_count[v] += 1

    start = set(automaton.start)
    collapsible = set()
    for q in automaton.states:
        if (q not in start
                and not automaton.is_final(q)
                and len(out_nbrs.get(q, ())) == 1
                and len(in_nbrs.get(q, ())) == 1
                and not eps_out_count.get(q)
                and not eps_in_count.get(q)):
            collapsible.add(q)

    if not collapsible:
        return set(), set(), {}

    removed_states = set()
    removed_edges = set()
    chains = []
    visited = set()

    for q in automaton.states:
        if q in collapsible:
            continue
        for v in out_nbrs.get(q, ()):
            if v not in collapsible or v in visited:
                continue
            chain = [q, v]
            visited.add(v)
            cur = v
            while True:
                (nxt,) = out_nbrs[cur]
                if nxt in collapsible and nxt not in visited:
                    chain.append(nxt)
                    visited.add(nxt)
                    cur = nxt
                else:
                    chain.append(nxt)
                    break
            if len(chain) <= 2:
                continue
            steps = []
            for i in range(len(chain) - 1):
                steps.append((chain[i], chain[i + 1]))
                removed_edges.add((chain[i], chain[i + 1]))
            for s in chain[1:-1]:
                removed_states.add(s)
            chains.append(((chain[0], chain[-1]), steps))

    return removed_states, removed_edges, chains


def visualize_automaton(
    automaton,
    rankdir="LR",
    max_exclusions=5,
    epsilon_symbol="ε",
    node_attrs=None,
    edge_attrs=None,
    fmt_state=str,
    sty_node=lambda q: {},
    coalesce_chains=True,
):
    """Visualize a materialized FSA or FST using Graphviz.

    Parallel edges between the same pair of states are merged into a single
    edge with a compact label.  For FSAs, labels are compressed via
    ``compress_symbols``; for FSTs, via ``compress_fst_labels`` (identity
    arcs displayed without colons, relational arcs factored into cartesian
    products).

    Returns an ``InteractiveGraph`` that renders as interactive SVG in Jupyter
    (click edge labels to expand) and delegates to the underlying
    ``graphviz.Digraph`` for ``.render()``, ``.pipe()``, etc.

    Args:
        fmt_state: Callable mapping a state to its display label (default ``str``).
        sty_node: Callable mapping a state to a dict of extra Graphviz node
            attributes (e.g. ``{'fillcolor': '#90EE90', 'style': 'filled,rounded'}``).
        coalesce_chains: Collapse unbranching chains into single edges with
            dot-separated labels (default ``True``).
    """

    is_fst = hasattr(automaton, 'B')   # FSTs have .A and .B alphabets
    prefix = uuid.uuid4().hex[:8]

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

    # --- Chain coalescing (visualization-only) ---
    removed_states = set()
    removed_edges = set()
    chain_edges = []
    if coalesce_chains:
        uv_data = uv_to_pairs if is_fst else uv_to_syms
        removed_states, removed_edges, chain_edges = _coalesce_unbranching(
            uv_data, uv_to_eps, automaton,
        )

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
        if q in removed_states:
            continue
        label = _html.escape(str(fmt_state(q)))
        sty = dict(peripheries='2' if automaton.is_final(q) else '1')
        sty.update(sty_node(q))
        dot.node(node_id[q], label=label, **sty)

    # Epsilon edges (HTML label so it matches the styled content edges)
    eps_html = f'<{_html.escape(epsilon_symbol)}>'
    for u, v in uv_to_eps:
        dot.edge(node_id[u], node_id[v], label=eps_html)

    # Content edges (HTML labels with colored meta-brackets)
    edge_labels = {}  # edge_id → expanded label (for click-to-expand)
    edge_idx = 0

    if is_fst:
        for (u, v), pairs in uv_to_pairs.items():
            if (u, v) in removed_edges:
                continue
            label = compress_fst_labels(
                pairs, input_alpha, output_alpha,
                max_exclusions=max_exclusions,
            )
            expanded = _expand_fst_pairs(pairs)
            compressed_plain = strip_meta_brackets(label)
            eid = f"e_{prefix}_{edge_idx}"
            edge_idx += 1
            dot.edge(node_id[u], node_id[v],
                     label=_label_to_html(label), id=eid,
                     tooltip=expanded)
            if expanded != compressed_plain:
                edge_labels[eid] = expanded
    else:
        for (u, v), S in uv_to_syms.items():
            if (u, v) in removed_edges:
                continue
            label = compress_symbols(
                S, alphabet=automaton.syms - {EPSILON},
                max_exclusions=max_exclusions,
            )
            expanded = _expand_symbols(S)
            compressed_plain = strip_meta_brackets(label)
            eid = f"e_{prefix}_{edge_idx}"
            edge_idx += 1
            dot.edge(node_id[u], node_id[v],
                     label=_label_to_html(label), id=eid,
                     tooltip=expanded)
            if expanded != compressed_plain:
                edge_labels[eid] = expanded

    # Coalesced chain edges
    for (head, tail), steps in chain_edges:
        step_labels = []
        step_expanded = []
        if is_fst:
            for u, v in steps:
                step_labels.append(compress_fst_labels(
                    uv_to_pairs[(u, v)], input_alpha, output_alpha,
                    max_exclusions=max_exclusions,
                ))
                step_expanded.append(_expand_fst_pairs(uv_to_pairs[(u, v)]))
        else:
            for u, v in steps:
                step_labels.append(compress_symbols(
                    uv_to_syms[(u, v)],
                    alphabet=automaton.syms - {EPSILON},
                    max_exclusions=max_exclusions,
                ))
                step_expanded.append(_expand_symbols(uv_to_syms[(u, v)]))
        merged_label = _m('\u00b7').join(step_labels)
        expanded = '\n'.join(step_expanded)
        compressed_plain = strip_meta_brackets(merged_label)
        eid = f"e_{prefix}_{edge_idx}"
        edge_idx += 1
        dot.edge(node_id[head], node_id[tail],
                 label=_label_to_html(merged_label), id=eid,
                 tooltip=expanded)
        if expanded != compressed_plain:
            edge_labels[eid] = expanded

    return InteractiveGraph(dot, edge_labels)
