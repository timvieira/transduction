#!/usr/bin/env python3
"""Generate SVG diagrams for README.md using the library's own visualization tools.

Usage:
    python scripts/generate_readme_assets.py

Outputs SVG files to images/ directory.
"""
import re
import resource
from pathlib import Path

# Memory limit per CLAUDE.md guidelines
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))

from graphviz import Digraph

from transduction import examples
from transduction.eager_nonrecursive import Precover
from transduction.viz import visualize_automaton

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def strip_svg_dimensions(svg: str) -> str:
    """Strip width/height from SVG for responsive rendering (pattern from viz.py:411)."""
    svg = re.sub(r'(<svg[^>]*?)\s+width="[^"]*"', r'\1', svg, count=1)
    svg = re.sub(r'(<svg[^>]*?)\s+height="[^"]*"', r'\1', svg, count=1)
    return svg


def save_svg(graph, path: Path):
    """Render a graphviz graph/InteractiveGraph to SVG file with stripped dimensions."""
    # InteractiveGraph wraps a Digraph in .dot
    dot = getattr(graph, 'dot', graph)
    svg = dot.pipe(format='svg').decode('utf-8')
    svg = strip_svg_dimensions(svg)
    path.write_text(svg)
    print(f"  {path.name}: {path.stat().st_size:,} bytes")


def generate_pushforward():
    """Conceptual diagram: cloud of source strings mapped to cloud of target strings.

    Uses the delete_b FST (deletes b, maps a→A) with a fictional source
    distribution to illustrate many-to-one transduction and mass aggregation.
    """
    # Source strings with fictional probabilities and their delete_b images
    source_strings = [
        ("a",   0.30, "A"),
        ("b",   0.20, "ε"),
        ("ab",  0.15, "A"),
        ("aa",  0.10, "AA"),
        ("ba",  0.08, "A"),
        ("bb",  0.07, "ε"),
        ("aab", 0.05, "AA"),
        ("aba", 0.05, "AA"),
    ]

    # Aggregate target probabilities
    target_mass = {}
    for _, p, y in source_strings:
        target_mass[y] = target_mass.get(y, 0.0) + p
    # Sort targets by mass descending
    target_strings = sorted(target_mass.items(), key=lambda t: -t[1])

    def fontsize(p, lo=10, hi=20):
        """Scale fontsize linearly with probability."""
        return lo + (hi - lo) * (p / 0.53)  # 0.53 is max target mass

    def bar(p, max_width=1.8):
        """Node width scaled by probability."""
        return 0.4 + max_width * (p / 0.53)

    dot = Digraph(
        graph_attr=dict(
            rankdir='LR',
            bgcolor='transparent',
            pad='0.3',
            nodesep='0.15',
            ranksep='1.5',
            compound='true',
        ),
        node_attr=dict(fontname='Monospace', fontsize='11', shape='box',
                       style='filled,rounded', height='0.3', margin='0.08,0.04'),
        edge_attr=dict(arrowsize='0.4', color='#bbbbbb', penwidth='0.7'),
    )

    # Source cloud
    with dot.subgraph(name='cluster_source') as s:
        s.attr(label='<<b>p(x)</b>  over X*>', labelloc='t',
               fontname='Helvetica', fontsize='12', fontcolor='#555555',
               style='rounded,dashed', color='#d4a84b', bgcolor='#fffdf5',
               penwidth='1.2')
        for x, p, _ in source_strings:
            fs = f'{fontsize(p):.0f}'
            w = f'{bar(p):.2f}'
            lbl = f'<<font point-size="{fs}"><b>{x}</b></font>  <font point-size="9" color="#999999">.{p*100:.0f}</font>>'
            s.node(f'x_{x}', label=lbl, fillcolor='#fff3e0', color='#e6a123',
                   width=w, penwidth='1.0')

    # Target cloud
    with dot.subgraph(name='cluster_target') as t:
        t.attr(label='<<b>p(y)</b>  over Y*>', labelloc='t',
               fontname='Helvetica', fontsize='12', fontcolor='#555555',
               style='rounded,dashed', color='#4caf50', bgcolor='#f5fff5',
               penwidth='1.2')
        for y, p in target_strings:
            fs = f'{fontsize(p):.0f}'
            w = f'{bar(p):.2f}'
            display = y if y != 'ε' else 'ε'
            lbl = f'<<font point-size="{fs}"><b>{display}</b></font>  <font point-size="9" color="#999999">.{p*100:.0f}</font>>'
            t.node(f'y_{y}', label=lbl, fillcolor='#e8f5e9', color='#4caf50',
                   width=w, penwidth='1.0')

    # Mapping edges
    for x, p, y in source_strings:
        dot.edge(f'x_{x}', f'y_{y}')

    # FST label between clusters (invisible node)
    dot.node('fst_label',
             label='<<font point-size="13"><b><i>f</i></b></font><br/>'
                   '<font point-size="9">delete_b</font>>',
             shape='none', fontname='Helvetica', fontcolor='#4a86c8')
    # Position it between clusters via invisible edges
    dot.edge('x_ab', 'fst_label', style='invis')
    dot.edge('fst_label', 'y_A', style='invis')

    save_svg(dot, IMAGES_DIR / "pushforward.svg")


def generate_delete_b():
    """Hero image: the delete_b FST."""
    fst = examples.delete_b()
    graph = visualize_automaton(fst)
    save_svg(graph, IMAGES_DIR / "delete_b.svg")


def generate_decomposition():
    """Colored precover DFA for delete_b with target='A'."""
    fst = examples.delete_b()
    p = Precover(fst, 'A')
    graph = p.graphviz()
    save_svg(graph, IMAGES_DIR / "decomposition.svg")


def generate_newspeak2():
    """The newspeak2 FST (Orwellian replacement)."""
    fst = examples.newspeak2()
    graph = visualize_automaton(fst)
    save_svg(graph, IMAGES_DIR / "newspeak2.svg")


def generate_parity():
    """The parity FST over {a,b}."""
    fst = examples.parity({'a', 'b'})
    graph = visualize_automaton(fst)
    save_svg(graph, IMAGES_DIR / "parity.svg")


if __name__ == '__main__':
    print("Generating README assets...")
    generate_pushforward()
    generate_delete_b()
    generate_decomposition()
    generate_newspeak2()
    generate_parity()
    print("Done.")
