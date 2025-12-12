from transduction.fst import FST, EPSILON


def infinite_quotient(alphabet=('a',), separators=('#',)):
    fst = FST()

    fst.add_I(0)
    fst.add_F(0)

    # From start (state 0)
    for x in alphabet:
        fst.add_arc(0, x, EPSILON, 0)

    # Exit first region after seeing the first separator
    for x in separators:
        fst.add_arc(0, x, EPSILON, 1)

    # Absorbing region:
    for x in alphabet:
        fst.add_arc(1, x, EPSILON, 1)
    for x in separators:
        fst.add_arc(1, x, EPSILON, 1)

    fst.add_arc(1, EPSILON, '1', 3)

    fst.add_F(3)

    return fst


def weird_copy():
    m = FST()
    m.add_start(0)
    m.add_stop(0)
    m.add_arc(0, 'b', 'b', 1)
    m.add_arc(0, 'a', 'a', 2)
    m.add_arc(1, '', '', 0)
    m.add_arc(2, '', '', 0)
    return m


def small():
    fst = FST()
    fst.add_I(0)
    fst.add_F(0)

    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'x', 2)

    fst.add_arc(2, 'a', 'a', 3)
    fst.add_arc(2, 'b', 'b', 3)

    fst.add_arc(3, 'a', 'a', 3)
    fst.add_arc(3, 'b', 'b', 3)

    fst.add_F(1)
    fst.add_F(3)
    return fst


def lookahead():
    fst = FST()
    fst.add_I(0)
    fst.add_F(0)

    fst.add_arc(0, 'a', '', 1)
    fst.add_arc(1, 'a', 'x', 11)
    fst.add_arc(11, 'a', 'x', 111)

    fst.add_arc(0, 'b', 'x', 2)

    fst.add_arc(2, 'a', 'a', 3)
    fst.add_arc(2, 'b', 'b', 3)

    fst.add_arc(3, 'a', 'a', 4)
    fst.add_arc(3, 'b', 'b', 4)

    fst.add_arc(4, 'a', 'a', 4)
    fst.add_arc(4, 'b', 'b', 4)

    fst.add_F(111)
    fst.add_F(4)
    return fst


def triplets_of_doom():
    m = FST()
    m.add_start(0)
    m.add_stop(0)
    m.add_arc(0, 'a', 'a', 1)
    m.add_arc(0, 'b', 'b', 2)
    m.add_arc(1, 'a', 'a', 3)
    m.add_arc(2, 'b', 'b', 4)
    m.add_arc(3, 'a', 'a', 0)
    m.add_arc(4, 'b', 'b', 0)
    return m


def samuel_example():
    sam = FST()

    # state 0
    sam.add_I(0)
    sam.add_F(0)
    sam.add_arc(0, 'a', EPSILON, 1)
    sam.add_arc(0, 'a', 'c', 2)
    sam.add_arc(0, 'b', 'y', 4)

    # state 1
    sam.add_arc(1, 'b', 'c', 3)

    # state 2
    sam.add_F(2)
    sam.add_arc(2, 'a', 'x', 4)

    # state 3
    sam.add_F(3)
    sam.add_arc(3, EPSILON, 'x', 4)

    # state 4
    sam.add_F(4)
    sam.add_arc(4, 'a', 'x', 4)
    sam.add_arc(4, 'b', 'x', 4)
    return sam


def number_comma_separator(
    Domain,
    SEP = '|',
    Digit = frozenset({str(i) for i in range(10)}),
    COMMA = frozenset({','}),
):
    """
    Example input-output pairs:
      ''            ==> ''
      '1,1,,'       ==> '1,1,|,'
      '1'           ==> '1'
      ',,,'         ==> ',|,|,'
      '100,'        ==> '100,'
      '100, '       ==> '100,| '
      '3,50'        ==> '3,50'
      ',00,'        ==> ',00,'
      '123'         ==> '123'
      '1,2,3'       ==> '1,2,3'
      '1, 2, 3'     ==> '1,| 2,| 3'
      '1, 2, and 3' ==> '1,| 2,| and 3'
    """
    m = FST()
    assert (Digit | COMMA) <= Domain

    # state 0 out arcs
    m.add_I(0)
    m.add_F(0)
    for x in Domain - COMMA:
        m.add_arc(0, x, x, 0)

    for x in COMMA:
        m.add_arc(0, x, x, 1)

    # state 1, out arcs
    m.add_F(1)
    for x in Digit:   # if we see another digit, then go back as this is still a numeric expression like 12,3
        m.add_arc(1, x, x, 0)
    m.add_arc(1, EPSILON, SEP, 2)

    # state 2, out arcs
    for x in COMMA:
        m.add_arc(2, x, x, 1)
    for x in Domain - (Digit | COMMA):
        m.add_arc(2, x, x, 0)

    return m


def delete_b():
    "This FST is over the alphabet `{a,b}`.  It deletes every instance of `b`."
    fst = FST()
    fst.add_I(0)
    fst.add_arc(0, 'a', 'A', 0)
    fst.add_arc(0, 'b', EPSILON, 0)
    fst.add_F(0)
    return fst


def sdd1_fst():
    fst = FST()
    fst.add_I(0)

    fst.add_arc(0, EPSILON, 'a', 1)
    fst.add_arc(1, 'a', EPSILON, 2)
    fst.add_arc(1, EPSILON, EPSILON, 1)

    fst.add_arc(2, 'a', 'a', 2)
    fst.add_arc(2, 'b', 'b', 2)

    fst.add_arc(0, 'a', 'a', 2)

    fst.add_F(2)

    return fst



def parity(alphabet):
    m = FST()

    E, O, F = 0, 1, 2   # Even, Odd, Final
    m.add_I(E)

    # Toggle parity on every consumed input symbol; output is epsilon while reading
    for x in alphabet:
        m.add_arc(E, x, EPSILON, O)
        m.add_arc(O, x, EPSILON, E)

    # Emit a single bit at the end via an epsilon-input arc, then accept
    m.add_arc(E, EPSILON, '1', F)   # even length → output 1
    m.add_arc(O, EPSILON, '0', F)   # odd length  → output 0
    m.add_F(F)
    return m


def duplicate(V):
    dup = FST()
    dup.add_I(0)
    for b in V:
        dup.add_arc(0, b, b, (1, b))
        dup.add_arc((1, b), EPSILON, b, 0)
    dup.add_F(0)
    return dup


def replace(mapping):
    fst = FST()
    fst.add_I(0)
    for x,y in mapping:
        fst.add_arc(0, x, y, 0)
    fst.add_F(0)
    return fst


def togglecase():
    T = FST()
    T.add_I(0)
    for b in range(256):
        if bytes([b]).isupper():
            T.add_arc(0, b, bytes([b]).lower()[0], 0)
        else:
            T.add_arc(0, b, bytes([b]).upper()[0], 0)
    T.add_F(0)
    return T


# # TODO: dump the machine to python code and create it here so that `pynini` is
# # no longer a dependency.
# def newspeak():
#     import pynini
#
#     # Symtab
#     symtab = pynini.SymbolTable()
#     symtab.add_symbol("eps")
#     #for c in range(256):
#     for c in "abcdefghijklmnopqrstuvwxyz":
#         symtab.add_symbol(c)
#
#     # Add chars
#     sigma_chars = [pynini.accep(symtab.find(idx), token_type=symtab)
#                    for idx in range(1, symtab.num_symbols())]
#
#     sigma = pynini.union(*sigma_chars)
#     sigma_star = sigma.closure().optimize()
#
#     def seq(s):
#         f = pynini.accep(str(s[0]), token_type=symtab)
#         for ch in s[1:]:
#             f += pynini.accep(str(ch), token_type=symtab)
#         return f
#
#     bad = seq("bad")
#     ungood = seq("ungood")
#
#     bad_to_ungood = pynini.cross(bad, ungood)
#
#     replace_bad = pynini.cdrewrite(bad_to_ungood, "", "", sigma_star).optimize()
#     replace_bad.set_input_symbols(symtab)
#     replace_bad.set_output_symbols(symtab)
#
#     m = FST()
#     m.add_I(replace_bad.start())
#     for s in replace_bad.states():
#         if 'Infinity' not in str(replace_bad.final(s)):   # TODO: absolutely hideous hack
#             m.add_F(s)
#         for a in replace_bad.arcs(s):
#             x, y = symtab.find(a.ilabel), symtab.find(a.olabel)
#             x = EPSILON if x == 'eps' else x
#             y = EPSILON if y == 'eps' else y
#             #print((x, y), a.nextstate)
#             m.add_arc(s, x, y, a.nextstate)
#
#     return m


def newspeak2():
    "Same as `newspeak` except over `str` rather than `bytes`."
    m = FST()
    m.add_start(0)
    m.add_stop(0)
    m.add_stop(2)
    m.add_stop(4)
    m.add_arc(0, 'a', 'a', 0)
    m.add_arc(0, 'b', 'b', 2)
    m.add_arc(0, 'b', 'u', 1)
    m.add_arc(0, 'c', 'c', 0)
    m.add_arc(0, 'd', 'd', 0)
    m.add_arc(0, 'e', 'e', 0)
    m.add_arc(0, 'f', 'f', 0)
    m.add_arc(0, 'g', 'g', 0)
    m.add_arc(0, 'h', 'h', 0)
    m.add_arc(0, 'i', 'i', 0)
    m.add_arc(0, 'j', 'j', 0)
    m.add_arc(0, 'k', 'k', 0)
    m.add_arc(0, 'l', 'l', 0)
    m.add_arc(0, 'm', 'm', 0)
    m.add_arc(0, 'n', 'n', 0)
    m.add_arc(0, 'o', 'o', 0)
    m.add_arc(0, 'p', 'p', 0)
    m.add_arc(0, 'q', 'q', 0)
    m.add_arc(0, 'r', 'r', 0)
    m.add_arc(0, 's', 's', 0)
    m.add_arc(0, 't', 't', 0)
    m.add_arc(0, 'u', 'u', 0)
    m.add_arc(0, 'v', 'v', 0)
    m.add_arc(0, 'w', 'w', 0)
    m.add_arc(0, 'x', 'x', 0)
    m.add_arc(0, 'y', 'y', 0)
    m.add_arc(0, 'z', 'z', 0)
    m.add_arc(1, 'a', 'n', 3)
    m.add_arc(2, 'a', 'a', 4)
    m.add_arc(2, 'b', 'b', 2)
    m.add_arc(2, 'b', 'u', 1)
    m.add_arc(2, 'c', 'c', 0)
    m.add_arc(2, 'd', 'd', 0)
    m.add_arc(2, 'e', 'e', 0)
    m.add_arc(2, 'f', 'f', 0)
    m.add_arc(2, 'g', 'g', 0)
    m.add_arc(2, 'h', 'h', 0)
    m.add_arc(2, 'i', 'i', 0)
    m.add_arc(2, 'j', 'j', 0)
    m.add_arc(2, 'k', 'k', 0)
    m.add_arc(2, 'l', 'l', 0)
    m.add_arc(2, 'm', 'm', 0)
    m.add_arc(2, 'n', 'n', 0)
    m.add_arc(2, 'o', 'o', 0)
    m.add_arc(2, 'p', 'p', 0)
    m.add_arc(2, 'q', 'q', 0)
    m.add_arc(2, 'r', 'r', 0)
    m.add_arc(2, 's', 's', 0)
    m.add_arc(2, 't', 't', 0)
    m.add_arc(2, 'u', 'u', 0)
    m.add_arc(2, 'v', 'v', 0)
    m.add_arc(2, 'w', 'w', 0)
    m.add_arc(2, 'x', 'x', 0)
    m.add_arc(2, 'y', 'y', 0)
    m.add_arc(2, 'z', 'z', 0)
    m.add_arc(3, 'd', 'g', 5)
    m.add_arc(4, 'a', 'a', 0)
    m.add_arc(4, 'b', 'b', 2)
    m.add_arc(4, 'b', 'u', 1)
    m.add_arc(4, 'c', 'c', 0)
    m.add_arc(4, 'e', 'e', 0)
    m.add_arc(4, 'f', 'f', 0)
    m.add_arc(4, 'g', 'g', 0)
    m.add_arc(4, 'h', 'h', 0)
    m.add_arc(4, 'i', 'i', 0)
    m.add_arc(4, 'j', 'j', 0)
    m.add_arc(4, 'k', 'k', 0)
    m.add_arc(4, 'l', 'l', 0)
    m.add_arc(4, 'm', 'm', 0)
    m.add_arc(4, 'n', 'n', 0)
    m.add_arc(4, 'o', 'o', 0)
    m.add_arc(4, 'p', 'p', 0)
    m.add_arc(4, 'q', 'q', 0)
    m.add_arc(4, 'r', 'r', 0)
    m.add_arc(4, 's', 's', 0)
    m.add_arc(4, 't', 't', 0)
    m.add_arc(4, 'u', 'u', 0)
    m.add_arc(4, 'v', 'v', 0)
    m.add_arc(4, 'w', 'w', 0)
    m.add_arc(4, 'x', 'x', 0)
    m.add_arc(4, 'y', 'y', 0)
    m.add_arc(4, 'z', 'z', 0)
    m.add_arc(5, '', 'o', 6)
    m.add_arc(6, '', 'o', 7)
    m.add_arc(7, '', 'd', 0)
    return m
