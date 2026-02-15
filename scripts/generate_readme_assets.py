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


def generate_logo():
    """Conceptual diagram: LM over X* -> FST f -> transduced LM over Y*."""
    dot = Digraph(
        graph_attr=dict(
            rankdir='LR',
            bgcolor='transparent',
            pad='0.4',
            nodesep='0.6',
            ranksep='0.8',
        ),
        node_attr=dict(fontname='Helvetica', fontsize='13'),
        edge_attr=dict(arrowsize='0.6', color='#555555', penwidth='1.2',
                       fontname='Helvetica', fontsize='10'),
    )
    # Source LM
    dot.node('px', label='<<b>p</b>(<i>x</i>)>',
             shape='ellipse', style='filled',
             fillcolor='#fff3e0', color='#e6a123', penwidth='1.5',
             fontcolor='#333333', width='0.8', height='0.45')
    # FST
    dot.node('fst',
             label='<<b><i>f</i></b><br/><font point-size="9">X* → Y*</font>>',
             shape='box', style='filled,rounded',
             fillcolor='#4a86c8', color='#35628f', penwidth='1.8',
             fontcolor='white', width='0.9', height='0.55')
    # Target LM
    dot.node('py', label='<<b>p</b>(<i>y</i>)>',
             shape='ellipse', style='filled',
             fillcolor='#e8f5e9', color='#4caf50', penwidth='1.5',
             fontcolor='#333333', width='0.8', height='0.45')

    dot.edge('px', 'fst', label='  X*  ', fontcolor='#888888')
    dot.edge('fst', 'py', label='  Y*  ', fontcolor='#888888')
    save_svg(dot, IMAGES_DIR / "logo.svg")


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
    generate_logo()
    generate_delete_b()
    generate_decomposition()
    generate_newspeak2()
    generate_parity()
    print("Done.")
