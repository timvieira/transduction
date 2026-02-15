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
    """Conceptual diagram: source distribution mapped through a lowercasing FST.

    Shows realistic English strings with plausible LM probabilities being
    transduced through a case-normalization FST, illustrating many-to-one
    mass aggregation (e.g., "The", "the", "THE" all map to "the").
    """
    # Source strings: plausible LM probabilities for cased English words.
    # FST: lowercase normalization (every character mapped to its lowercase).
    source_strings = [
        ("the",    0.218,  "the"),
        ("The",    0.147,  "the"),
        ("of",     0.109,  "of"),
        ("a",      0.094,  "a"),
        ("and",    0.086,  "and"),
        ("is",     0.077,  "is"),
        ("A",      0.064,  "a"),
        ("And",    0.038,  "and"),
        ("Is",     0.031,  "is"),
        ("THE",    0.019,  "the"),
        ("OF",     0.017,  "of"),
    ]

    # Aggregate target probabilities
    target_mass = {}
    for _, p, y in source_strings:
        target_mass[y] = target_mass.get(y, 0.0) + p
    target_strings = sorted(target_mass.items(), key=lambda t: -t[1])

    max_p = max(p for _, p in target_strings)

    def fontsize(p, lo=9.5, hi=14):
        return lo + (hi - lo) * (p / max_p)

    def node_width(p, base=0.45, scale=1.4):
        return base + scale * (p / max_p)

    dot = Digraph(
        graph_attr=dict(
            rankdir='LR',
            bgcolor='transparent',
            pad='0.25',
            nodesep='0.12',
            ranksep='1.8',
            compound='true',
        ),
        node_attr=dict(fontname='Helvetica', fontsize='10', shape='box',
                       style='filled,rounded', height='0.28',
                       margin='0.1,0.035'),
        edge_attr=dict(arrowsize='0.35', color='#cccccc', penwidth='0.6'),
    )

    # Source cloud
    with dot.subgraph(name='cluster_source') as s:
        s.attr(label='<<font point-size="11"><b>p(x)</b></font>>',
               labelloc='t', fontname='Helvetica', fontcolor='#666666',
               style='rounded', color='#dddddd', bgcolor='#fafafa',
               penwidth='0.8')
        for x, p, _ in source_strings:
            fs = f'{fontsize(p):.1f}'
            w = f'{node_width(p):.2f}'
            prob = f'{p:.3f}'[1:]   # ".218" style
            lbl = (f'<<font point-size="{fs}">{x}</font>'
                   f'<font point-size="8" color="#aaaaaa">  {prob}</font>>')
            s.node(f'x_{x}', label=lbl,
                   fillcolor='#fff8f0', color='#d9c5a0', penwidth='0.6')

    # Target cloud
    with dot.subgraph(name='cluster_target') as t:
        t.attr(label='<<font point-size="11"><b>p(y)</b></font>>',
               labelloc='t', fontname='Helvetica', fontcolor='#666666',
               style='rounded', color='#dddddd', bgcolor='#fafafa',
               penwidth='0.8')
        for y, p in target_strings:
            fs = f'{fontsize(p):.1f}'
            w = f'{node_width(p):.2f}'
            prob = f'{p:.3f}'[1:]
            lbl = (f'<<font point-size="{fs}">{y}</font>'
                   f'<font point-size="8" color="#aaaaaa">  {prob}</font>>')
            t.node(f'y_{y}', label=lbl,
                   fillcolor='#f0f7f0', color='#a0c9a0', penwidth='0.6',
                   width=w)

    # Mapping edges
    for x, _, y in source_strings:
        dot.edge(f'x_{x}', f'y_{y}')

    # FST label between clusters
    dot.node('fst_label',
             label='<<font point-size="11" color="#555555"><i>f</i>  = lowercase</font>>',
             shape='none', fontname='Helvetica')
    dot.edge('x_and', 'fst_label', style='invis')
    dot.edge('fst_label', 'y_and', style='invis')

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
