from transduction.fsa import FSA
from transduction.lazy import Lazy


class LazyWrapper(Lazy):

    def __init__(self, base):
        self.base = base

    def start(self):
        return self.base.start

    def is_final(self, i):
        return self.base.is_final(i)

    def arcs(self, i):
        return self.base.arcs(i)


def test_foo():

    m = FSA()
    m.add_start(1)
    m.add_stop(3)
    m.add_arc(1, '', 2)
    m.add_arc(2, 'x', 3)
    m.add_arc(3, '', 2)

    lazy = LazyWrapper(m)

    # sanity check for lazy wrapper
    assert lazy.materialize().equal(m)

    E = lazy.epsremove()

    assert E.materialize().equal(m)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
