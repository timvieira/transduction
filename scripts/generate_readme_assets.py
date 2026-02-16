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
from transduction.precover import Precover
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
    """Conceptual diagram: cloud of source strings mapped through an FST to a cloud of targets.

    Uses neato with pinned positions for a scattered cloud layout.
    Ellipsis nodes indicate the distributions are partial views.
    Multi-word phrases with plausible probabilities; lowercasing FST.
    """
    from graphviz import Graph

    # Source strings: complete independent sentences with different casings.
    # These must be obviously unrelated topics so readers don't mistake them
    # for sequential fragments of one text.
    source_strings = [
        ("it is cold outside",   0.148, "it is cold outside"),
        ("It is cold outside",   0.117, "it is cold outside"),
        ("IT IS COLD OUTSIDE",   0.009, "it is cold outside"),
        ("she likes coffee",     0.106, "she likes coffee"),
        ("She likes coffee",     0.079, "she likes coffee"),
        ("the bus is late",      0.094, "the bus is late"),
        ("The bus is late",      0.047, "the bus is late"),
        ("we need more time",    0.072, "we need more time"),
        ("We Need More Time",    0.028, "we need more time"),
        ("that was unexpected",  0.063, "that was unexpected"),
        ("That Was Unexpected",  0.022, "that was unexpected"),
    ]

    # Aggregate target probabilities
    target_mass = {}
    for _, p, y in source_strings:
        target_mass[y] = target_mass.get(y, 0.0) + p
    target_strings = sorted(target_mass.items(), key=lambda t: -t[1])

    max_src = max(p for _, p, _ in source_strings)
    max_tgt = max(p for _, p in target_strings)

    # Use neato with pinned positions for organic cloud layout
    dot = Digraph(
        engine='neato',
        graph_attr=dict(
            bgcolor='transparent',
            pad='0.3',
            overlap='false',
            splines='true',
        ),
        node_attr=dict(fontname='Helvetica', fontsize='10', shape='box',
                       style='filled,rounded', height='0.26',
                       margin='0.08,0.03'),
        edge_attr=dict(arrowsize='0.3', color='#cccccc', penwidth='0.5'),
    )

    # Source cloud — scattered positions on the left (x ~ 0-2.5)
    src_positions = [
        (0.3, 5.8), (1.8, 5.2), (0.1, 4.3),   # brown fox variants
        (1.5, 3.6), (0.4, 2.9),                  # lazy dog variants
        (1.7, 2.1), (0.2, 1.4),                  # jumped over it
        (1.4, 0.7), (0.6, 0.0),                  # and ran away
        (1.8, -0.7), (0.3, -1.3),                # is not so bad
    ]

    for i, ((x_str, p, _), (px, py)) in enumerate(zip(source_strings, src_positions)):
        fs = 8.5 + 3.0 * (p / max_src)
        lbl = (f'<<font point-size="{fs:.1f}">{x_str}</font>'
               f'<font point-size="7" color="#aaaaaa"> {p:.3f}</font>>')
        dot.node(f'x{i}', label=lbl, pos=f'{px},{py}!',
                 fillcolor='#fff8f0', color='#d9c5a0', penwidth='0.6')

    # Source ellipsis nodes
    dot.node('xdots1', label='<<font point-size="12" color="#bbbbbb">...</font>>',
             shape='none', pos='0.9,6.5!')
    dot.node('xdots2', label='<<font point-size="12" color="#bbbbbb">...</font>>',
             shape='none', pos='1.1,-2.0!')

    # Target cloud — scattered positions on the right (x ~ 6-8)
    tgt_positions = [
        (7.0, 4.8),   # the brown fox
        (6.3, 3.1),   # a lazy dog
        (7.2, 1.5),   # jumped over it
        (6.5, -0.1),  # and ran away
        (7.0, -1.2),  # is not so bad
    ]

    tgt_ids = {}
    for j, ((y, p), (px, py)) in enumerate(zip(target_strings, tgt_positions)):
        tgt_ids[y] = f'y{j}'
        fs = 9.0 + 3.0 * (p / max_tgt)
        w = 1.1 + 0.9 * (p / max_tgt)
        lbl = (f'<<font point-size="{fs:.1f}">{y}</font>'
               f'<font point-size="7" color="#aaaaaa"> {p:.3f}</font>>')
        dot.node(f'y{j}', label=lbl, pos=f'{px},{py}!',
                 fillcolor='#f0f7f0', color='#a0c9a0', penwidth='0.6',
                 width=f'{w:.2f}')

    # Target ellipsis
    dot.node('ydots1', label='<<font point-size="12" color="#bbbbbb">...</font>>',
             shape='none', pos='6.8,5.8!')
    dot.node('ydots2', label='<<font point-size="12" color="#bbbbbb">...</font>>',
             shape='none', pos='6.7,-2.0!')

    # Mapping edges
    for i, (_, _, y) in enumerate(source_strings):
        dot.edge(f'x{i}', tgt_ids[y])

    # FST label in the center
    dot.node('fst_label',
             label='<<font point-size="11" color="#555555"><i>f</i> = lowercase</font>>',
             shape='none', fontname='Helvetica', pos='3.8,2.5!')

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
