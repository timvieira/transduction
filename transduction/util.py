import html
import json
import base64
import signal
from functools import partial
from contextlib import contextmanager
from IPython.display import SVG, HTML, display


#_______________________________________________________________________________
# Integerizer (formerly from arsenal)

class Integerizer:
    """Maintain a perfect hash (bijection) between keys and contiguous ints."""

    def __init__(self, data=None):
        self._map = {}
        self._list = []
        self._frozen = False
        if data: self.add(data)

    def _add(self, k):
        try:
            return self._map[k]
        except KeyError:
            if self._frozen:
                raise ValueError(f'Alphabet is frozen. Key "{k}" not found.')
            x = self._map[k] = len(self._list)
            self._list.append(k)
            return x

    def __contains__(self, k):
        return k in self._map

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)

    def __eq__(self, other):
        return self._list == other._list

    def __repr__(self):
        return f'Integerizer(size={len(self)}, frozen={self._frozen})'

    def __getitem__(self, i):
        if isinstance(i, list):
            return [self._list[ii] for ii in i]
        return self._list[i]

    def __call__(self, k):
        if isinstance(k, list):
            return [self._add(kk) for kk in k]
        return self._add(k)

    encode = __call__
    decode = __getitem__
    lookup = __getitem__
    add = __call__

    def freeze(self):
        self._frozen = True
        return self

    def keys(self):
        return self._list

    def items(self):
        return self._map.items()


#_______________________________________________________________________________
# colors (formerly from arsenal)

class _AnsiStyle:
    """ANSI escape code wrapper supporting ``style % string`` formatting."""
    def __init__(self, code):
        self._code = code
    def __mod__(self, text):
        return f'\033[{self._code}m{text}\033[0m'

class _LightColors:
    red = _AnsiStyle('1;31')
    green = _AnsiStyle('1;32')

class _DarkColors:
    white = _AnsiStyle('2;37')

class colors:
    light = _LightColors()
    dark = _DarkColors()
    @staticmethod
    def mark(ok):
        return '\033[0;32m\u2714\033[0m' if ok else '\033[2;31m\u2718\033[0m'


#_______________________________________________________________________________
# memoize (formerly from arsenal.cache)

class memoize:
    """Cache a function's return value to avoid recalculation."""
    def __init__(self, func):
        self.func = func
        self.cache = {}
        try:
            self.__name__ = func.__name__
            self.__doc__ = func.__doc__
        except AttributeError:
            pass

    def __get__(self, obj, objtype=None):
        if obj is None: return self.func
        return partial(self, obj)

    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        return f'<memoize({self.func!r})>'


#_______________________________________________________________________________
# timelimit (formerly from arsenal)

class Timeout(Exception):
    pass

@contextmanager
def timelimit(seconds, sig=signal.SIGALRM):
    """Context manager that raises Timeout after `seconds`."""
    if seconds is None:
        yield
        return
    def signal_handler(signum, frame):
        raise Timeout(f'Call took longer than {seconds} seconds.')
    signal.signal(sig, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    yield
    signal.setitimer(signal.ITIMER_REAL, 0)


@contextmanager
def timeit(name, fmt='{name} ({htime})', header=None):
    """Context manager that prints elapsed wall-clock time."""
    import sys
    from time import time
    if header is not None:
        print(header)
    b4 = time()
    yield
    sec = time() - b4
    ht = f'{sec:.4f} sec' if sec < 60 else f'{sec/60:.1f} min'
    print(fmt.format(name=name, htime=ht, sec=sec), file=sys.stderr)


#_______________________________________________________________________________
# sample (formerly from arsenal.maths)

def sample(w, size=None, u=None):
    """Draw samples from an (unnormalized) discrete distribution via inverse CDF."""
    import numpy as np
    c = np.cumsum(w)
    if u is None:
        u = np.random.uniform(0, 1, size=size)
    return c.searchsorted(u * c[-1])


#_______________________________________________________________________________
# HTML table utilities

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
