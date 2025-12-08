"""
The module contains some experimental visualization stuff that I used to
derive the latest `peekaboo_recursive.py` method.
"""
from transduction.eager_nonrecursive import is_universal, EPSILON
from transduction.fsa import frozenset
from graphviz import Digraph
#from transduction.lazy_nonrecursive import LazyPrecoverNFA, Lazy
from transduction.lazy import Lazy
from transduction import Precover
from arsenal import colors
from IPython.display import display


class Analyze:
    def __init__(self, fst, target):
        source_alphabet = fst.A - {EPSILON}
        tmp = LazyPrecoverNFA(fst, target).det()
        self.dfa = tmp.materialize()
        self.arcs = set(self.dfa.arcs())
        self.states = set(self.dfa.states)

        N = len(target)
        def refine(state):
            return frozenset((i, ys[:N]) for i, ys in state)
        refined = self.dfa.rename(refine)

        self.universal = {i for i in self.dfa.states if is_universal(refined, refine(i), source_alphabet)}
        self.remainder = {i for i in self.dfa.states if self.dfa.is_final(i)} - self.universal

    def compare(self, prev):
        curr = self

        print('# nodes', colors.line(80))
        for i in sorted(prev.states - curr.states):   # removed
            print(colors.light.red % '├─ remove:', i)
        for i in sorted(curr.states & prev.states):   # kept
            print(colors.light.yellow % '├─ copy:  ', i)
        for i in sorted(curr.states - prev.states):   # addded
            print(colors.light.green % '├─ added: ', i)

        print('# edges', colors.line(80))
        for (i,x,j) in sorted(prev.arcs - curr.arcs):
            if i in prev.universal: continue
            print(colors.light.red % '├─ remove:', (i,x,j))
        for (i,x,j) in sorted(curr.arcs - prev.arcs):
            print(colors.light.green % '├─ added: ', (i,x,j))
        for (i,x,j) in sorted(curr.arcs & prev.arcs):
            if i in prev.universal:
                continue
            print(colors.light.yellow % '├─ copy:  ', (i,x,j))


class LazyPrecoverNFA(Lazy):

    def __init__(self, f, target):
        self.f = f
        self.target = target
        self.N = len(target)

    def arcs(self, state):
        (i, ys) = state
        if ys.startswith(self.target):
            for x, y, j in self.f.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                else:

                    if 1:                         # if true use peekaboo
                        was = (ys + y)
                        now = was[:self.N+1]
                        if was == now:
                            yield (x, (j, was))
                        else:
                            # mark truncated nodes because they need to be removed in the next iteration
                            yield (x, (j, now + '#'))
                    else:
                        was = (ys + y)
                        now = was[:self.N]
                        if was == now:
                            yield (x, (j, was))
                        else:
                            # mark truncated nodes because they need to be removed in the next iteration
                            yield (x, (j, now + '#'))

        else:
            n = len(ys)
            for x, y, j in self.f.arcs(i):
                if y == EPSILON:
                    yield (x, (j, ys))
                elif y == self.target[n]:
                    yield (x, (j, ys + y))

    def start(self):
        for i in self.f.I:
            yield (i, '')

    def is_final(self, state):
        (i, ys) = state
        return self.f.is_final(i) and ys.startswith(self.target)


def visualization(fst, Target):
    dups = set()

    N = len(Target)

    prev = None

    for n in range(N+1):

        target = Target[:n]
        print(target or 'ε')

        dot = Digraph(
            graph_attr=dict(rankdir='LR'),
            node_attr=dict(
                fontname='Monospace',
                fontsize='8',
                height='.05',
                width='.05',
                margin="0.055,0.042",
                shape='box',
                style='rounded, filled',
            ),
            edge_attr=dict(
                arrowsize='0.3',
                fontname='Monospace',
                fontsize='8'
            ),
        )

        curr = Analyze(fst, target)
        if prev is not None:
            curr.compare(prev)

        for i in curr.states:
            if i in curr.universal:
                dot.node(str(i), fillcolor='red')
                #continue
            elif i in curr.remainder:
                dot.node(str(i), fillcolor='magenta')

            for x, j in curr.dfa.arcs(i):
                dups.add((i,x,j))
                dot.edge(str(i), str(j), label=x)

        display(dot)

        display(Precover(fst, target))

        prev = curr
