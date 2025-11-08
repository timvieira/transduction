import html
from IPython.display import SVG, HTML, display

def format_table(rows, headings=None):
    def fmt(x):
        if isinstance(x, (SVG, HTML)):
            return x.data
        elif hasattr(x, '_repr_html_'):
            return x._repr_html_()
        elif hasattr(x, '_repr_svg_'):
            return x._repr_svg_()
        elif hasattr(x, '_repr_image_svg_xml'):
            return x._repr_image_svg_xml()
        else:
            return f'<pre>{html.escape(str(x))}</pre>'

    return (
        '<table>'
        + (
            '<tr style="font-weight: bold;">'
            + ''.join(f'<td>{x}</td>' for x in headings)
            + '</tr>'
            if headings
            else ''
        )
        + ''.join(
            '<tr>' + ''.join(f'<td>{fmt(x)}</td>' for x in row) + ' </tr>' for row in rows
        )
        + '</table>'
    )


def display_table(*args, **kwargs):
    return display(HTML(format_table(*args, **kwargs)))


#def drop_weights(x):
#    if isinstance(x, FSA):
#        return x
#    m = FSA()
#    for i,a,j,_ in x.arcs():
#        m.add(i,a,j)
#    for i,_ in x.I:
#        m.add_start(i)
#    for i,_ in x.F:
#        m.add_stop(i)
#    return m




#_ord = ord
#def newspeak(small=False):
#    T = FST(Float)
#
#    if small:
#        alphabet = 'abcde'
#        ord = lambda x: x
#        outs  = "ungood"
#    else:
#        alphabet = range(256)
#        outs  = b"ungood"
#        ord = _ord
#
#    T.add_I(b'', 1)
#    T.add_F(b'', 1)
#    T.add_F(b'dead', 1)
#
#    # ---------- state 0 : copy everything except ‘b’ ----------
#    B = ord("b")
#    for b in alphabet:
#        if b == B:
#            T.add_arc(b'', (B, EPSILON), b'b', 1)            # potential start of “bad”
#            T.add_arc(b'b', (EPSILON, B), b'dead', 1)
#        else:
#            T.add_arc(b'', (b, b), b'', 1)     # pass-through
#
#    # ---------- state 1 : saw “b” ----------
#    A = ord("a")
#    for b in alphabet:
#        if b == A:
#            T.add_arc(b'b', (A, EPSILON), b'ba', 1)    # maybe “ba…”
#            T.add_arc(b'ba', (EPSILON, B), b'a-dead', 1)    # maybe “ba…”
#            T.add_arc(b'a-dead', (EPSILON, A), b'dead', 1)    # maybe “ba…”
#        else:
#            # flush the buffered ‘b’ and continue copying
#            T.add_arc(b'b', (b, B), b'', 1)      # output ‘b’ first
#
#    # ---------- state 2 : saw “ba” ----------
#    D = ord("d")
#    for b in alphabet:
#        if b == D:
#            T.add_arc(b'ba', (D, EPSILON), b'bad', 1)    # exactly “bad”
#        else:
#            # flush “ba” then current byte
#            T.add_arc(b'ba', (b, B), b'a', 1)
#            T.add_arc(b'a', (EPSILON, A), b'', 1)
#
#    # ---------- state 3 : matched “bad” – output “ungood” ----------
#    prev = b'bad'
#    for o in outs:
#        y = f'{prev}:{o}'
#        T.add_arc(prev, (EPSILON, o), y, 1)     # ε → u,n,g,o,o,d
#        prev = y
#    T.add_arc(prev, (EPSILON, EPSILON), b'', 1)       # back to copy mode
#
#    return T



#class MyFST(FST):
#
#    def is_universal(self, q, source_alphabet):
#        q_fsa = FSA()
#        assert q in self.states
#        q_fsa.add_start(q)
#        for i, (a, b), j, _ in self.arcs():
#            q_fsa.add(i, a, j)
#        for i,_ in self.F:
#            q_fsa.add_stop(i)
#        q_fsa = q_fsa.trim().min()
#        #if verbosity > 0:
#        #display(q_fsa.graphviz())
#        if len(q_fsa.nodes) != 1:
#            #if verbosity > 0: print(len(q_fsa.nodes), 'nodes')
#            return False
#        [i] = q_fsa.nodes
#        for a in source_alphabet:
#            if set(q_fsa.arcs(i, a)) != {i}:
#                return False
#        return True
#
#    def graphviz(self, source_alphabet=None, **kwargs):
#        attr_node = {}
#        for x in self.states:
#            attr_node[x] = {'shape': 'box', 'style': 'rounded,filled', 'margin': '.05'}
#            if self.is_universal(x, (source_alphabet or self.A) - {EPSILON}):   # TODO: why is epsilon in there?
#                attr_node[x]['fillcolor'] = 'yellow'
#            else:
#                attr_node[x]['fillcolor'] = 'red'
#        return super().graphviz(fmt_node=lambda x: str(x), attr_node=attr_node, **kwargs)
#
