import html
import json
import base64
from IPython.display import SVG, HTML, display


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
            return f'<pre>{html.escape(str(data))}</pre>'
        if mime == "application/json":
            try:
                obj_json = json.loads(data) if isinstance(data, str) else data
                pretty = json.dumps(obj_json, indent=2, ensure_ascii=False)
            except Exception:       # pylint: disable=W0718
                pretty = str(data)
            return f'<pre>{html.escape(pretty)}</pre>'
        if mime == "text/plain":
            return f'<pre>{html.escape(str(data))}</pre>'
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

    return f'<pre>{html.escape(str(x))}</pre>'


def format_table(rows, headings=None):
    """Build an HTML ``<table>`` string from a list of rows.

    Each element in a row is converted to HTML via ``_as_html_cell``, so cells
    can contain IPython display objects (FSA/FST graphviz, images, HTML
    fragments, etc.) alongside plain values.

    Args:
        rows: Iterable of row iterables.  Each element becomes one ``<td>``.
        headings: Optional column header labels.
    """
    head_html = ""
    if headings:
        head_cells = "".join(f"<th>{h}</th>" for h in headings)
        head_html = f"<thead><tr>{head_cells}</tr></thead>"

    body_rows = []
    for row in rows:
        cells = "".join(f'<td style="vertical-align:top">{_as_html_cell(x)}</td>' for x in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = "<tbody>" + "".join(body_rows) + "</tbody>"

    return (
        '<table class="fmt-table" style="border-collapse:collapse;">'
        f"{head_html}{body_html}"
        "</table>"
    )


def display_table(rows, **kwargs):
    """Render a table in a Jupyter notebook.

    Convenience wrapper that calls ``format_table`` and displays the result
    as an IPython ``HTML`` object.  Accepts the same keyword arguments as
    ``format_table`` (e.g. ``headings``).
    """
    display(HTML(format_table(rows, **kwargs)))
