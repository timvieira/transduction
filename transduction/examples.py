from transduction.fst import FST, EPSILON


def infinite_quotient(alphabet=('a',), separators=('#',)):
    fst = FST()

    fst.add_start(0)
    fst.add_stop(0)

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

    fst.add_stop(3)

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
    fst.add_start(0)
    fst.add_stop(0)

    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'x', 2)

    fst.add_arc(2, 'a', 'a', 3)
    fst.add_arc(2, 'b', 'b', 3)

    fst.add_arc(3, 'a', 'a', 3)
    fst.add_arc(3, 'b', 'b', 3)

    fst.add_stop(1)
    fst.add_stop(3)
    return fst


def lookahead():
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)

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

    fst.add_stop(111)
    fst.add_stop(4)
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
    sam.add_start(0)
    sam.add_stop(0)
    sam.add_arc(0, 'a', EPSILON, 1)
    sam.add_arc(0, 'a', 'c', 2)
    sam.add_arc(0, 'b', 'y', 4)

    # state 1
    sam.add_arc(1, 'b', 'c', 3)

    # state 2
    sam.add_stop(2)
    sam.add_arc(2, 'a', 'x', 4)

    # state 3
    sam.add_stop(3)
    sam.add_arc(3, EPSILON, 'x', 4)

    # state 4
    sam.add_stop(4)
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
    m.add_start(0)
    m.add_stop(0)
    for x in Domain - COMMA:
        m.add_arc(0, x, x, 0)

    for x in COMMA:
        m.add_arc(0, x, x, 1)

    # state 1, out arcs
    m.add_stop(1)
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
    """
    This FST is over the alphabet `{a,b}`: it deletes every instance of `b` and replaces a -> A
    This example has infinite quotients.
    """
    fst = FST()
    fst.add_start(0)
    fst.add_arc(0, 'a', 'A', 0)
    fst.add_arc(0, 'b', EPSILON, 0)
    fst.add_stop(0)
    fst.B.add('b')     # avoid OOV errors.
    return fst


def sdd1_fst():
    fst = FST()
    fst.add_start(0)

    fst.add_arc(0, EPSILON, 'a', 1)
    fst.add_arc(1, 'a', EPSILON, 2)
    fst.add_arc(1, EPSILON, EPSILON, 1)

    fst.add_arc(2, 'a', 'a', 2)
    fst.add_arc(2, 'b', 'b', 2)

    fst.add_arc(0, 'a', 'a', 2)

    fst.add_stop(2)

    return fst



def parity(alphabet):
    m = FST()

    E, O, F = 0, 1, 2   # Even, Odd, Final
    m.add_start(E)

    # Toggle parity on every consumed input symbol; output is epsilon while reading
    for x in alphabet:
        m.add_arc(E, x, EPSILON, O)
        m.add_arc(O, x, EPSILON, E)

    # Emit a single bit at the end via an epsilon-input arc, then accept
    m.add_arc(E, EPSILON, '1', F)   # even length → output 1
    m.add_arc(O, EPSILON, '0', F)   # odd length  → output 0
    m.add_stop(F)
    return m


#def duplicate(V):
#    dup = FST()
#    dup.add_start(0)
#    for b in V:
#        dup.add_arc(0, b, b, (1, b))
#        dup.add_arc((1, b), EPSILON, b, 0)
#    dup.add_stop(0)
#    return dup


def duplicate(V, K=2):
    "Duplicate (by K > 1) each symbol in the input string, e.g., `abc -> a^K b^K c^K`."
    assert K > 1
    dup = FST()
    dup.add_start(0)
    for b in V:
        dup.add_arc(0, b, b, (0, b))
        for k in range(K-2):
            dup.add_arc((k, b), EPSILON, b, (k+1, b))
        dup.add_arc((K-2, b), EPSILON, b, 0)
    dup.add_stop(0)
    return dup.renumber()


def replace(mapping):
    fst = FST()
    fst.add_start(0)
    for x,y in mapping:
        fst.add_arc(0, x, y, 0)
    fst.add_stop(0)
    return fst


def doom(V, K):   # k-tuples of doom
    """doom(V={'a', 'b'}, K) creates a copy transducer for (a^K|b^K)^*.

    The example `triplets_of_doom` is a special case of this method.  This
    method generates some tricky examples with infinite remainders.

    Historically, this example is what helped us surface the issues related to
    target buffer truncation (without it the unbounded buffering construction of
    the precover results in infinitely many states, causing the DFA-factoring
    method to run forever unnecessarily).

    """
    assert K > 1
    dup = FST()
    dup.add_start(0)
    for b in V:
        dup.add_arc(0, b, b, (0, b))
        for k in range(K-2):
            dup.add_arc((k, b), b, b, (k+1, b))
        dup.add_arc((K-2, b), EPSILON, EPSILON, 0)
    dup.add_stop(0)
    return dup.renumber()


#def togglecase():
#    T = FST()
#    T.add_start(0)
#    for b in range(256):
#        if bytes([b]).isupper():
#            T.add_arc(0, b, bytes([b]).lower()[0], 0)
#        else:
#            T.add_arc(0, b, bytes([b]).upper()[0], 0)
#    T.add_stop(0)
#    return T


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
#     m.add_start(replace_bad.start())
#     for s in replace_bad.states():
#         if 'Infinity' not in str(replace_bad.final(s)):   # TODO: absolutely hideous hack
#             m.add_stop(s)
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


def togglecase():
    T = FST()
    T.add_start(0)
    for x in 'abcdefghijklmnopqrstuvwxyz ':
        T.add_arc(0, x.lower(), x.upper(), 0)
        T.add_arc(0, x.upper(), x.lower(), 0)
    T.add_stop(0)
    return T


def lowercase():
    T = FST()
    T.add_start(0)
    for x in 'abcdefghijklmnopqrstuvwxyz ':
        T.add_arc(0, x.lower(), x.lower(), 0)
        T.add_arc(0, x.upper(), x.lower(), 0)
    T.add_stop(0)
    return T



#____________________________
#


def mystery1():
    fst = FST()

    fst.add_start(0)

    # Path A: 'a' -> 'c'
    fst.add_arc(0, 'a', 'c', 1)

    # Path B: 'b' (ε) then 'a' -> 'c'
    fst.add_arc(0, 'b', EPSILON, 2)
    fst.add_arc(2, 'a', 'c', 3)

    # After 'c', just loop with 'x'
    for q in (1, 3):
        fst.add_stop(q)
        fst.add_arc(q, 'a', 'x', q)
        fst.add_arc(q, 'b', 'x', q)

    return fst


def mystery2():

    fst = FST()

    fst.add_start(0)

    # Path 1: 'a' -> 'c'
    fst.add_arc(0, 'a', 'c', 1)

    # Path 2: 'b' (ε) then 'a' -> 'c'  i.e., "ba"
    fst.add_arc(0, 'b', EPSILON, 5)
    fst.add_arc(5, 'a', 'c', 2)

    # Path 3: 'b' (ε), then 'b' (ε), then 'b' -> 'c'  i.e., "bbb"
    fst.add_arc(5, 'b', EPSILON, 6)
    fst.add_arc(6, 'b', 'c', 3)

    # After 'c', loop 'x' on both symbols
    for q in (1, 2, 3):
        fst.add_stop(q)
        fst.add_arc(q, 'a', 'x', q)
        fst.add_arc(q, 'b', 'x', q)

    return fst


def mystery3():
    fst = FST()

    fst.add_start(0)

    # Transition: track last symbol, no output yet.
    fst.add_arc(0, 'a', EPSILON, 1)
    fst.add_arc(0, 'b', EPSILON, 2)

    fst.add_arc(1, 'a', EPSILON, 1)
    fst.add_arc(1, 'b', EPSILON, 2)

    fst.add_arc(2, 'a', EPSILON, 1)
    fst.add_arc(2, 'b', EPSILON, 2)

    # Final ε-output arcs
    fst.add_arc(1, EPSILON, 'A', 3)   # last was a
    fst.add_arc(2, EPSILON, 'B', 4)   # last was b

    fst.add_stop(3)
    fst.add_stop(4)

    return fst


def mystery4():
    fst = FST()

    fst.add_start(0)

    # From state 0
    fst.add_arc(0, 'a', EPSILON, 1)  # first a
    fst.add_arc(0, 'b', EPSILON, 0)  # still zero a's

    # From state 1
    fst.add_arc(1, 'a', EPSILON, 2)  # second a
    fst.add_arc(1, 'b', EPSILON, 1)  # still exactly one a

    # From state 2
    fst.add_arc(2, 'a', EPSILON, 2)
    fst.add_arc(2, 'b', EPSILON, 2)

    # Final ε-output arcs
    fst.add_arc(0, EPSILON, '0', 3)  # zero a's
    fst.add_arc(1, EPSILON, '1', 4)  # exactly one a
    fst.add_arc(2, EPSILON, '0', 3)  # at least two a's

    fst.add_stop(3)
    fst.add_stop(4)

    return fst


def mystery5():

    fst = FST()

    fst.add_start(0)

    # Cycle mod 3 on any input symbol, no output
    for s_from, s_to in [(0, 1), (1, 2), (2, 0)]:
        fst.add_arc(s_from, 'a', EPSILON, s_to)
        fst.add_arc(s_from, 'b', EPSILON, s_to)

    # Final ε-output arcs
    fst.add_arc(0, EPSILON, '0', 3)
    fst.add_arc(1, EPSILON, '1', 4)
    fst.add_arc(2, EPSILON, '2', 5)

    fst.add_stop(3)
    fst.add_stop(4)
    fst.add_stop(5)

    return fst


def mystery6():
    fst = FST()

    fst.add_start(0)

    fst.add_arc(0, 'a', '', 1)
    fst.add_arc(1, 'a', '', 2)
    fst.add_arc(2, 'a', '', 3)

    fst.add_arc(3, 'a', 'a', 3)
    fst.add_arc(3, 'b', 'b', 3)
    fst.add_arc(3, 'c', 'c', 3)

    fst.add_arc(0, '', 'b', 4)

    fst.add_arc(4, '', 'a', 4)
    fst.add_arc(4, '', 'b', 4)
    fst.add_arc(4, '', 'c', 5)
    fst.add_stop(5)

    fst.add_stop(3)

    return fst


def infinite_quotient2():
    fst = FST()

    fst.add_start(0)

    # From start (state 0)
    fst.add_arc(0, 'a', EPSILON, 1)   # odd a-count
    fst.add_arc(0, 'b', EPSILON, 0)   # still even

    # Parity-tracking region (no '#' allowed anymore)
    fst.add_arc(1, 'a', EPSILON, 0)
    fst.add_arc(1, 'b', EPSILON, 1)

    # Exit parity region
    fst.add_arc(0, '#', EPSILON, 2)
    fst.add_arc(1, '#', EPSILON, 3)

    # Absorbing region:
    fst.add_arc(2, 'a', EPSILON, 2)
    fst.add_arc(2, 'b', EPSILON, 2)
    fst.add_arc(2, '#', EPSILON, 2)

    # Absorbing region:
    fst.add_arc(3, 'a', EPSILON, 3)
    fst.add_arc(3, 'b', EPSILON, 3)
    fst.add_arc(3, '#', EPSILON, 3)

    # even → '0', odd → '1', absorbing → '0'
    #fst.add_arc(0, '', '0', 'done')
    fst.add_stop('done')
    fst.add_arc(3, '', '1', 'done')
    fst.add_arc(2, '', '0', 'done')

    fst.add_stop(0)
    fst.add_stop(1)

    return fst


def mystery7():
    fst = FST()

    fst.add_start(0)

    # Path 1: 0 -b|c-> 3
    fst.add_arc(0, 'b', 'c', 3)

    # Path 2: 0 -a|ε-> 1; 1 -b|c-> 2
    fst.add_arc(0, 'a', EPSILON, 1)
    fst.add_arc(1, 'b', 'c', 2)

    # After we have emitted 'c', we only emit 'x' forever.
    for q in (2, 3):
        fst.add_stop(q)
        fst.add_arc(q, 'a', 'x', q)
        fst.add_arc(q, 'b', 'x', q)

    return fst


def mystery8():
    fst = FST()

    fst.add_start(0)
    fst.add_stop(0)

    # Direct C, like your original
    fst.add_arc(0, 'a', 'c', 1)
    fst.add_stop(1)
    fst.add_arc(1, 'a', 'x', 3)

    # Indirect C via epsilon-output
    fst.add_arc(0, 'b', EPSILON, 2)
    fst.add_arc(2, 'b', 'c', 3)

    fst.add_stop(3)
    fst.add_arc(3, 'a', 'x', 3)
    fst.add_arc(3, 'b', 'x', 3)

    return fst


# ---- UniversalityFilter test FSTs ----
# These exercise specific levels of the UniversalityFilter optimization hierarchy.
# See notes/UniversalityFilter-Tests.ipynb for detailed descriptions.


def gated_universal():
    """ip-universal witness check (level 2). Partial function.

    State 0 is a non-universal gate (not final, arcs on {a,b,c}).
    State 1 has complete self-loops on {a,b,c} (ip-universal).
    State 2 only has 'a' arcs (not ip-universal).
    After reading one symbol from state 0, the powerset DFA reaches states
    containing state 1, so the witness check fires immediately.
    """
    fst = FST()
    fst.add_start(0); fst.add_stop(1); fst.add_stop(2)
    y = 'y'

    # State 0: gate (not final — avoids ε output ambiguity)
    fst.add_arc(0, 'a', y, 1); fst.add_arc(0, 'a', y, 2)
    fst.add_arc(0, 'b', y, 1); fst.add_arc(0, 'b', y, 2)
    fst.add_arc(0, 'c', y, 1)

    # State 1: ip-universal (complete self-loops, single output per input)
    for x in 'abc':
        fst.add_arc(1, x, y, 1)

    # State 2: NOT ip-universal (only 'a' arcs)
    fst.add_arc(2, 'a', y, 2)

    return fst


def complementary_halves():
    """BFS fallback and positive cache (levels 5 and 3). Partial function.

    Alphabet {a,b,c,d} with 4 states each covering only half the alphabet.
    No state is ip-universal, so universality must be discovered via BFS.
    Complementary pairs ({1,2} and {3,4}) cover all symbols.
    """
    fst = FST()
    fst.add_start(0)
    for s in [1, 2, 3, 4]:
        fst.add_stop(s)
    y = 'y'
    symbols = ['a', 'b', 'c', 'd']

    # State 0: dispatcher (not final)
    for x in symbols:
        for dest in [1, 2, 3, 4]:
            fst.add_arc(0, x, y, dest)

    # State 1 ({a,b}) -> {1,2}
    for x in ['a', 'b']:
        fst.add_arc(1, x, y, 1); fst.add_arc(1, x, y, 2)

    # State 2 ({c,d}) -> {1,2}
    for x in ['c', 'd']:
        fst.add_arc(2, x, y, 1); fst.add_arc(2, x, y, 2)

    # State 3 ({a,c}) -> {3,4}
    for x in ['a', 'c']:
        fst.add_arc(3, x, y, 3); fst.add_arc(3, x, y, 4)

    # State 4 ({b,d}) -> {3,4}
    for x in ['b', 'd']:
        fst.add_arc(4, x, y, 3); fst.add_arc(4, x, y, 4)

    return fst


def shrinking_nonuniversal():
    """Negative cache / subset monotonicity (level 4). Partial function.

    Alphabet {a,b,c}. Three states, none covering all symbols (missing 'c').
    The set {1,2,3} is NOT universal. Subsets should be recognized as
    non-universal via the negative cache without running BFS again.
    """
    fst = FST()
    fst.add_start(0)
    for s in [1, 2, 3]:
        fst.add_stop(s)
    y = 'y'

    # State 0: dispatcher (not final, no ε self-loops)
    for x in 'abc':
        for dest in [1, 2, 3]:
            fst.add_arc(0, x, y, dest)

    # State 1: only 'a'
    fst.add_arc(1, 'a', y, 1)

    # State 2: only 'b'
    fst.add_arc(2, 'b', y, 2)

    # State 3: {a, b} -> splits on transitions
    fst.add_arc(3, 'a', y, 1)
    fst.add_arc(3, 'b', y, 2)

    return fst


def scaled_newspeak(n_patterns=5, alpha_size=10, n_partial=0):
    """Multi-pattern replacement FST. Function.

    State 0 passes through non-trigger symbols with identity output. Trigger
    symbols deterministically enter a pattern mid-state (ε output). Each
    mid-state handles all symbols and routes back to state 0 on its completion
    symbol. Requires n_patterns <= alpha_size for unique triggers.

    Note: all states are ip-universal (each individually accepts Σ*), but
    check_all_input_universal returns False because trigger symbols break the
    start-set containment invariant.

    Setting n_partial>0 removes self-loop arcs from the first n_partial
    mid-states so they only accept their completion symbol, creating dead
    paths from those states.
    """
    assert n_patterns <= alpha_size, "need unique triggers"
    assert n_partial <= n_patterns
    fst = FST()
    fst.add_start(0); fst.add_stop(0)
    symbols = [str(i) for i in range(alpha_size)]
    out_symbols = [chr(ord('A') + i) for i in range(n_patterns)]

    triggers = {symbols[i % alpha_size] for i in range(n_patterns)}

    # State 0: passthrough for non-triggers only
    for x in symbols:
        if x not in triggers:
            fst.add_arc(0, x, x, 0)

    # Patterns: trigger enters mid-state with ε output (no passthrough overlap)
    for i in range(n_patterns):
        trigger = symbols[i % alpha_size]
        mid_state = i + 1
        fst.add_stop(mid_state)
        fst.add_arc(0, trigger, EPSILON, mid_state)
        fst.add_arc(mid_state, symbols[(i+1) % alpha_size], out_symbols[i], 0)
        if i < n_partial:
            # Partial mid-state: only accepts completion symbol (non-AUI)
            pass
        else:
            for x in symbols:
                if x != symbols[(i+1) % alpha_size]:
                    fst.add_arc(mid_state, x, x, mid_state)

    return fst


def layered_witnesses(n_layers=3):
    """Witness check at scale (level 2). Partial function.

    A chain of n layers, each with 3 states: gate (non-final dispatcher),
    univ (ip-universal with complete self-loops), and partial (only 'a' arcs).
    Every universal conclusion should be resolved by the witness check with
    zero BFS work.
    """
    fst = FST()
    y = 'y'

    for i in range(n_layers):
        g = 3 * i       # gate
        u = 3 * i + 1   # universal
        p = 3 * i + 2   # partial

        fst.add_stop(u); fst.add_stop(p)

        # Universal state: complete self-loops (single output per input)
        for x in 'ab':
            fst.add_arc(u, x, y, u)

        # Partial state: only 'a'
        fst.add_arc(p, 'a', y, p)

        # Gate: fans out (no ε self-loop)
        for x in 'ab':
            fst.add_arc(g, x, y, u)
            fst.add_arc(g, x, y, p)
            if i + 1 < n_layers:
                fst.add_arc(g, x, y, 3 * (i + 1))  # next gate

    fst.add_start(0)
    return fst


def anbn():
    """Ambiguous a^n b^n FST from the paper.

    Maps: a→b, aa→c, aaa→bbb, aaaa→bbbb, ...
    Exercises ambiguity in decomposition: reading target symbol 'b' requires
    lookahead to determine which source path is active.
    """
    fst = FST()
    fst.add_start("START")
    for final in ["2ea", "4ec", "7ab"]:
        fst.add_stop(final)
    fst.add_arc("START", "a", EPSILON, "1ae")
    fst.add_arc("1ae", EPSILON, "b", "2ea")
    fst.add_arc("1ae", "a", EPSILON, "3ae")
    fst.add_arc("3ae", EPSILON, "c", "4ec")
    fst.add_arc("3ae", "a", "b", "5eb")
    fst.add_arc("5eb", EPSILON, "b", "6eb")
    fst.add_arc("6eb", EPSILON, "b", "7ab")
    fst.add_arc("7ab", "a", "b", "7ab")
    return fst


def backticks_to_quote():
    """Backtick-to-quote transducer: single ` passes through, `` collapses to ".

    Also maps a→b as passthrough. Exercises lookahead and epsilon-output
    buffering: after reading one backtick, the decomposition must wait to see
    whether a second backtick follows before committing output.
    """
    fst = FST()
    fst.add_start("START")
    fst.add_stop("START")

    # a → b passthrough
    fst.add_arc("START", "a", EPSILON, "CHAR_a")
    fst.add_arc("CHAR_a", EPSILON, "b", "START")

    # backtick logic
    fst.add_arc("START", "`", EPSILON, "Quote")
    fst.add_arc("Quote", EPSILON, "`", "1_Quote")
    fst.add_stop("1_Quote")

    fst.add_arc("Quote", "`", '"', "2_quotes")
    fst.add_arc("2_quotes", EPSILON, EPSILON, "START")
    fst.add_stop("2_quotes")

    # continue from single-backtick state
    fst.add_arc("1_Quote", "a", EPSILON, "CHAR_a")

    return fst


def parity_copy():
    """Parity-dependent copy: even-length inputs produce b^n, odd produce c^n.

    Unlike `parity()` which outputs a single bit, this transducer's output
    length grows with input length, creating interesting decomposition behavior.
    """
    fst = FST()
    fst.add_start("START")
    fst.add_stop("END")

    # Even branch
    fst.add_arc("START", EPSILON, EPSILON, "E0")
    fst.add_arc("E0", "a", "b", "E1")
    fst.add_arc("E1", "a", "b", "E0")
    fst.add_arc("E0", EPSILON, EPSILON, "END")

    # Odd branch
    fst.add_arc("START", EPSILON, EPSILON, "O0")
    fst.add_arc("O0", "a", "c", "O1")
    fst.add_arc("O1", "a", "c", "O0")
    fst.add_arc("O1", EPSILON, EPSILON, "END")

    return fst
