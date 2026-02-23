from IPython.display import HTML, display, Javascript
from tokenization.util import prefixes, escape, format_table, flatten

def format_key(key):
    return ('|'.join(map(escape, flatten(key))))


def bundle_about_html(self, limit=5):
    items = sorted(self.items(), key = lambda xy: -xy[1])
    lines = []
    partial = self.trie.node2prefix[self.node]

    EOT = '☐'
    def color_partial(x):
        assert x.startswith(partial)
        return f'<font style="color: #bbb;">{escape(partial)}</font><font>{escape(x[len(partial):])}</font>'


    lines.append("""
    <style>
    .panel__content {
        max-height: 0;                 /* start collapsed        */
        overflow: hidden;              /* clip while collapsing  */
        transition: max-height .3s ease;
    }
    .panel.open .panel__content {
        max-height: 500px;             /* any value > real height */
    }
    </style>
    """)

    lines.append("""\
    <div class="panel" style="font-family: monospace !important;">
       <a href="#" onclick="this.parentNode.classList.toggle('open'); return false;">
        Expand ▸
       </a>
    """)

    lines.append(
        f'<strong style="">{format_key(self.key)} {escape(self.trie.node2prefix[self.node])} {self.weight:.2f}</strong>'
    )

    lines.append("""\
    <div class="panel__content">
    """)

    tmp = self.logp_next.top(5)
    if None in tmp:
        tmp[EOT] = tmp.pop(None)
        tmp = tmp.sort_descending()

    lines.append(format_table([[escape(x) for x in tmp.keys()],
                               [HTML(f'<font style="font-size: 5pt;">{x:.2f}</font>') for x in tmp.values()]]))

    lines.append('<table style="white-space: nowrap !important; table-layout: fixed; border: thin solid black; width: 150px; overflow-x: hidden;">')
    for node, mass in items[:limit]:
        lines.append(f'<tr><td>{mass:6.2f}</td><td style="text-align: left;">{color_partial(self.trie.leaf2word[node])}</td></tr>')
    for _ in range(limit - len((items[:limit]))):
        lines.append('<tr><td colspan=2 style="color: #bbb; text-align: center;">n/a</td></tr>')
    n_missing = max(0, len(items) - limit)
    #if n_missing > 0:
    lines.append(f'<tr><td colspan=2 style="color: #bbb; text-align: center;">({n_missing} more)</td></tr>')
    eot = self.trie.children[self.node].get(None)
    if eot is None:
        lines.append(f'<tr><td>n/a</td><td style="text-align: left;">{EOT}</td></tr>')
    else:
        #token_string = self.trie.leaf2word[eot]
        lines.append(f'<tr><td>{self.mass[eot]:.2f}</td><td style="text-align: left;">{EOT}</td></tr>')

    lines.append('</table>')

    lines.append("""
    </div>
    </div>
    """)

    return HTML('\n'.join(lines))


def show_beam_state(self, limit=5):
    css = f"""
    <style>
    .beam-container {{
        position: relative;
    }}
    .beam-cell {{
        position: relative;
        padding: 0px;
        z-index: 3;
        border: 1px solid lightgray;
    }}
    .link {{
        stroke: darkgray;
        stroke-width: 2;
        fill: none;
        z-index: 2;
    }}
    </style>
    """

    table = []

    from arsenal import Integerizer
    m = Integerizer()
    M = 1 + len(self)

    column = []
    for bundle in self:
        p = bundle.parent
        html = f'<div class="beam-cell">'
        html += bundle.about_html(limit=limit)._repr_html_()
        html += '</div>'
        column.append(HTML(html))
    column.extend(['']*(M - len(column)))
    table.append(column)

    return HTML(format_table(zip(*table)))


def character_beam_show_html(self, context, limit=3, links=True, link_x_space=20, filename='/tmp/tmp.html'):

    css = f"""
    <style>
    .beam-container {{
        position: relative;
    }}
    .beam-cell {{
        position: relative;
        margin-left: {link_x_space}px;
        padding: 0px;
        z-index: 3;
        border: 1px solid lightgray;
    }}
    .link {{
        stroke: darkgray;
        stroke-width: 2;
        fill: none;
        z-index: 2;
    }}
    </style>
    """

    table = []

    from arsenal import Integerizer
    m = Integerizer()
    M = 1 + max(len(self.beam(xs)) for xs in prefixes(context))

    for xs in prefixes(context):
        column = []
        for bundle in self.candidates(xs):
            p = bundle.parent
            html = f'<div class="beam-cell" data-n={m(bundle)} data-pn={m(p)}>'
            html += bundle.about_html(limit=limit)._repr_html_()
            html += '</div>'
            column.append(HTML(html))
        column.extend(['']*(M - len(column)))
        table.append(column)

    formatted_table = format_table(zip(*table), headings=list(map(repr, prefixes(context))))

    jsbody = """
    // Create SVG overlay
    const container = document.querySelector('.beam-container');
    const displayedTable = container.querySelector('table');
    if (!displayedTable) {
        console.error("Could not find table element. Can't draw links.");
        return;
    }
    const overlayRect = displayedTable.getBoundingClientRect();

    const svg = d3.select(container)
        .append('svg')
        .style('position', 'absolute')
        .style('top', 0)
        .style('left', 0)
        .style('pointer-events', 'none')
        .style('z-index', 3)
        .attr('width', overlayRect.width)
        .attr('height', overlayRect.height);

    // Draw connections between cells and their parents
    const cells = Array.from(container.getElementsByClassName('beam-cell'));
    cells.forEach(cell => {
        const cellNum = parseInt(cell.dataset.n);
        const parentNum = parseInt(cell.dataset.pn);

        // Skip if no valid parent
        if (parentNum < 0) return;

        const parentCell = cells.find(c => parseInt(c.dataset.n) === parentNum);
        if (!parentCell) return;

        const cellRect = cell.getBoundingClientRect();
        const parentRect = parentCell.getBoundingClientRect();

        const source = {
            x: parentRect.right - overlayRect.left,
            y: parentRect.top + parentRect.height/2 - overlayRect.top
        };
        const target = {
            x: cellRect.left - overlayRect.left,
            y: cellRect.top + cellRect.height/2 - overlayRect.top
        };

        const linkGenerator = d3.linkHorizontal()
            .x(d => d.x)
            .y(d => d.y);

        svg.append('path')
            .attr('class', 'link')
            .attr('d', linkGenerator({source, target}))
            .style('pointer-events', 'none');
    });

    // Add window resize handler to update SVG size if needed
    window.addEventListener('resize', () => {
        const newRect = container.getBoundingClientRect();
        svg
            .attr('width', newRect.width)
            .attr('height', newRect.height);
    });
    """

    # Add D3 code to draw links
    js = """
    if (typeof(require)=='function') {
        // if requireJS is available (eg VSCode notebook), require D3
        const D3 = 'https://d3js.org/d3.v7.min';
        require.config({ paths: { d3: D3 } });
        require(['d3'], function (d3) {
        """ + jsbody + """
        });
    } else {
        // if not (Jupyter Notebook), use script tag to load D3
        var script = document.createElement('script');
        script.src = 'https://d3js.org/d3.v7.min.js';
        script.onload = function() {
        """ + jsbody + """
        };
        document.head.appendChild(script);
    }
    """

    display(HTML(css + f'<div class="beam-container">{formatted_table}</div>'))
    if links:
        display(Javascript(js))

    if filename:
        open(filename,'w',encoding='utf-8').write(
            (css + f'<div class="beam-container">{formatted_table}</div>')
            + ('<script>' + js + '</script>' if links else '')
        )
