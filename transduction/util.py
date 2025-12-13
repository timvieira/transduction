import html
import json
import base64
from IPython.display import SVG, HTML, display

#
#def format_table(rows, headings=None):
#    def fmt(x):
#        if isinstance(x, (SVG, HTML)):
#            return x.data
#        elif hasattr(x, '_repr_html_'):
#            return x._repr_html_()
#        elif hasattr(x, '_repr_svg_'):
#            return x._repr_svg_()
#        elif hasattr(x, '_repr_image_svg_xml'):
#            return x._repr_image_svg_xml()
#        else:
#            return f'<pre>{html.escape(str(x))}</pre>'
#
#    return (
#        '<table>'
#        + (
#            '<tr style="font-weight: bold;">'
#            + ''.join(f'<td>{x}</td>' for x in headings)
#            + '</tr>'
#            if headings
#            else ''
#        )
#        + ''.join(
#            '<tr>' + ''.join(f'<td>{fmt(x)}</td>' for x in row) + ' </tr>' for row in rows
#        )
#        + '</table>'
#    )
#
#
#def display_table(*args, **kwargs):
#    return display(HTML(format_table(*args, **kwargs)))
#

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
    display(HTML(format_table(rows, **kwargs)))
