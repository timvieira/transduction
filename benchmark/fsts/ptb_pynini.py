"""
PTB Tokenizer FST built using pynini.

This uses pynini's cdrewrite rules to create a compact FST that matches
NLTK's TreebankWordTokenizer behavior. The pynini FST is then converted to the native
FST format for use with the Rust decomposition backend.

Period handling uses pynini's built-in [EOS] boundary symbol in cdrewrite rules
to match NLTK's TreebankWordTokenizer behavior (periods only separated at end of string).

Based on the reference implementation in ptb.py.
"""
import pynini
from pynini import cdrewrite, cross, union

from transduction.fst import FST
from transduction.fsa import EPSILON as NATIVE_EPSILON


EPS = "<eps>"
MARKER = '257'    # Out-of-band word boundary marker (not a real byte) Removed at the end
SEP = '258'       # Output separator symbol (word boundary in final output)


def char_to_byte_str(ch):
    """Convert a single ASCII character to its byte value as string."""
    return str(ord(ch))


def string_to_byte_strs(s):
    """Convert a string to tuple of byte value strings."""
    return tuple(str(b) for b in s.encode('utf-8'))


def decode_ptb_output(output_tuple):
    """Decode PTB FST output back to readable string."""
    tokens = []
    current_token = []

    for sym in output_tuple:
        if sym == SEP:
            if current_token:
                byte_vals = [int(b) for b in current_token]
                tokens.append(bytes(byte_vals).decode('utf-8', errors='replace'))
                current_token = []
        elif sym != MARKER and sym != NATIVE_EPSILON:
            current_token.append(sym)

    if current_token:
        byte_vals = [int(b) for b in current_token]
        tokens.append(bytes(byte_vals).decode('utf-8', errors='replace'))

    return ' '.join(tokens)


def _char_to_byte(ch, symbols):
    """Convert a single ASCII character to pynini acceptor."""
    return pynini.accep(str(ord(ch)), token_type=symbols)


def _chars_to_bytes(s, symbols):
    """Convert a string to concatenation of byte acceptors."""
    result = pynini.accep("", token_type=symbols)
    for ch in s:
        result = result + pynini.accep(str(ord(ch)), token_type=symbols)
    return result


def _build_separator_inserter(symbols, ext_symbols):
    """
    Build an FST that converts MARKER bytes to SEP and collapses consecutive markers.
    """
    fst = pynini.Fst()
    start = fst.add_state()
    in_markers = fst.add_state()
    fst.set_start(start)
    fst.set_final(start)
    fst.set_final(in_markers)

    marker_id = ext_symbols.find(MARKER)
    sep_id = ext_symbols.find(SEP)

    # From start: first marker -> output SEP, go to in_markers
    fst.add_arc(start, pynini.Arc(marker_id, sep_id, 0, in_markers))

    # From in_markers: more markers -> consume without output
    fst.add_arc(in_markers, pynini.Arc(marker_id, 0, 0, in_markers))

    # From both states: non-marker -> pass through
    for idx in range(1, symbols.num_symbols()):
        sym = symbols.find(idx)
        if sym is not None and idx != marker_id:
            fst.add_arc(start, pynini.Arc(idx, idx, 0, start))
            fst.add_arc(in_markers, pynini.Arc(idx, idx, 0, start))

    fst.set_input_symbols(ext_symbols)
    fst.set_output_symbols(ext_symbols)
    return fst.optimize()


def build_ptb_fst_pynini():
    """
    Build PTB tokenizer FST using pynini, then convert to native format.

    Returns a native FST with ~130 states that implements Penn Treebank tokenization.
    """
    # Build symbol table: bytes 0-255 + out-of-band MARKER
    symbols = pynini.SymbolTable()
    symbols.add_symbol(EPS, 0)
    for bt in range(256):
        symbols.add_symbol(str(bt), bt + 1)
    symbols.add_symbol(MARKER, 257)

    # Extended symbols: adds SEP for output
    ext_symbols = pynini.SymbolTable()
    ext_symbols.add_symbol(EPS, 0)
    for bt in range(256):
        ext_symbols.add_symbol(str(bt), bt + 1)
    ext_symbols.add_symbol(MARKER, 257)
    ext_symbols.add_symbol(SEP, 258)

    # Helper function shortcuts
    def cb(ch):
        return _char_to_byte(ch, symbols)

    def cs(s):
        return _chars_to_bytes(s, symbols)

    # Common acceptors
    MARKER_ACC = pynini.accep(MARKER, token_type=symbols)
    SPACE = cb(" ")
    APOS = cb("'")
    QUOTE = cb('"')
    DOT = cb(".")
    DASH = cb("-")
    BACKTICK = cb("`")
    DOUBLE_BACKTICK = BACKTICK + BACKTICK

    # Build sigma (all bytes 0-255 + MARKER)
    sigma = pynini.union(
        *[pynini.accep(str(i), token_type=symbols) for i in range(256)],
        pynini.accep(MARKER, token_type=symbols),
    )
    sigma_star = pynini.closure(sigma)

    # Identity FST as base (passes through all bytes)
    identity_fst = pynini.Fst()
    s = identity_fst.add_state()
    identity_fst.set_start(s)
    identity_fst.set_final(s)
    for idx in range(symbols.num_symbols()):
        if idx != 0:
            identity_fst.add_arc(s, pynini.Arc(idx, idx, 0, s))
    identity_fst.set_input_symbols(symbols)
    identity_fst.set_output_symbols(symbols)
    identity_fst = identity_fst.closure()

    # === Quote handling ===
    # Matches NLTK's STARTING_QUOTES then ENDING_QUOTES order:
    # 1. '' after space/brackets -> `` (STARTING_QUOTES rule 3 for '{2})
    # 2. Remaining '' -> separated '' (ENDING_QUOTES rule 1)
    # 3. " after word chars -> '' (closing double-quote)
    # 4. " after space/brackets -> `` (opening double-quote)
    # 5. Remaining " -> `` (e.g., at start of string)
    # 6. Wrap existing `` with markers

    # Characters that can precede a CLOSING quote
    # Include UTF-8 continuation bytes (128-191 / 0x80-0xBF) for accented characters
    UTF8_CONTINUATION_BYTES = union(*[pynini.accep(str(i), token_type=symbols) for i in range(128, 192)])
    WORD_CHARS = union(
        *[cb(chr(i)) for i in range(ord('a'), ord('z')+1)],  # lowercase
        *[cb(chr(i)) for i in range(ord('A'), ord('Z')+1)],  # uppercase
        *[cb(chr(i)) for i in range(ord('0'), ord('9')+1)],  # digits
        APOS, cb("!"), cb("?"), cb("."), cb(")"), cb(","),  # punctuation
        cb(";"), cb(":"),  # semicolon and colon can precede closing quote
        cb("]"),  # closing bracket can precede closing quote
        UTF8_CONTINUATION_BYTES,  # UTF-8 continuation bytes for accented chars
    )

    # Characters that can precede an OPENING quote
    opening_context = union(cb("("), cb("["), cb("{"), cb("<"), SPACE)
    DOUBLE_APOS = APOS + APOS

    # Rule 1: '' after space/brackets -> `` (opening double-single-quotes)
    # NLTK STARTING_QUOTES rule 3: ([ \(\[{<])(\"{2}|\'{2}) -> \1 ``
    opening_double_apos_rule = cdrewrite(
        cross(DOUBLE_APOS, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        opening_context.plus,
        "", sigma_star
    )

    # Rule 2: Remaining '' -> MARKER '' MARKER (closing/separating)
    # NLTK ENDING_QUOTES rule 1: '' -> ' '' '
    closing_double_apos_rule = cdrewrite(
        cross(DOUBLE_APOS, MARKER_ACC + APOS + APOS + MARKER_ACC),
        "", "", sigma_star
    )

    # Rule 3: " after word chars -> '' (closing quote) - must run FIRST among "-rules
    closing_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + APOS + APOS + MARKER_ACC),
        WORD_CHARS,
        "", sigma_star
    )

    # Rule 4: " after space/brackets -> `` (opening quote)
    opening_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        opening_context.plus,
        "", sigma_star
    )

    # Rule 5a: " at beginning of string -> `` (NLTK STARTING_QUOTES rule 1: ^")
    bos_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        "[BOS]", "", sigma_star
    )

    # Rule 5b: Remaining " -> '' (NLTK ENDING_QUOTES rule 2: all other " become '')
    remaining_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + APOS + APOS + MARKER_ACC),
        "", "", sigma_star
    )

    # Rule 6: `` -> MARKER `` MARKER (for manually entered backticks)
    backtick_rule = cdrewrite(
        cross(DOUBLE_BACKTICK, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        "", "", sigma_star
    )

    # Compose: '' rules first, then " rules, then backtick wrapping
    quotes_fst = (opening_double_apos_rule @ closing_double_apos_rule
                  @ closing_quote_rule @ opening_quote_rule
                  @ bos_quote_rule @ remaining_quote_rule
                  @ backtick_rule).optimize()

    # === Punctuation ===
    DIGIT = union(*[cb(str(i)) for i in range(10)])
    NON_DIGIT = pynini.difference(sigma, DIGIT).optimize()

    # , and : before non-digit
    # NLTK regex: ([:,])([^\d]) - only separate when followed by non-digit
    punct_1 = cdrewrite(
        cross(cb(","), MARKER_ACC + cb(",") + MARKER_ACC) |
        cross(cb(":"), MARKER_ACC + cb(":") + MARKER_ACC),
        "", NON_DIGIT, sigma_star
    )

    # , and : at end of string only (before EOS)
    # NLTK regex: ([:,])$ - only at end of string
    punct_2 = cdrewrite(
        cross(cb(","), MARKER_ACC + cb(",") + MARKER_ACC) |
        cross(cb(":"), MARKER_ACC + cb(":") + MARKER_ACC),
        "", "[EOS]", sigma_star
    )

    # Ellipsis
    ellipsis_rule = cdrewrite(
        cross(DOT + DOT + DOT, MARKER_ACC + DOT + DOT + DOT + MARKER_ACC),
        "", "", sigma_star
    )

    # Special punct: ; @ # % & $
    special_punct = [cb(c) for c in ";@#%&$"]
    punct_4 = cdrewrite(
        union(*[cross(sym, MARKER_ACC + sym + MARKER_ACC) for sym in special_punct]),
        "", "", sigma_star
    )

    # Period at end - ONLY separate when followed by EOS (end of string)
    # This matches NLTK's behavior which uses $ regex anchor
    # NLTK regex: ([^\.])(\.)([\]\)}>"']*)\s*$
    # Period is separated when followed by optional closing punct, optional whitespace, then EOS
    NON_DOT = pynini.difference(sigma, DOT).optimize()

    # Optional closing punctuation that can appear after period before EOS
    # Includes MARKER_ACC since quotes are wrapped with markers at this point
    CLOSING_PUNCT = union(
        cb("]"), cb(")"), cb("}"), cb(">"),
        APOS, MARKER_ACC,
    )
    CLOSING_PUNCT_STAR = pynini.closure(CLOSING_PUNCT)
    WHITESPACE_STAR = pynini.closure(SPACE)

    punct_5 = cdrewrite(
        cross(DOT, MARKER_ACC + DOT),
        NON_DOT,
        CLOSING_PUNCT_STAR + WHITESPACE_STAR + pynini.accep("[EOS]"),
        sigma_star
    )

    # ? and !
    punct_6 = cdrewrite(
        cross(cb("?"), MARKER_ACC + cb("?") + MARKER_ACC) |
        cross(cb("!"), MARKER_ACC + cb("!") + MARKER_ACC),
        "", "", sigma_star
    )

    punct_fst = (punct_1 @ punct_2 @ ellipsis_rule @ punct_4 @ punct_5 @ punct_6).optimize()

    # === Brackets and parens ===
    parens_chars = [cb(c) for c in "[](){}><"]
    parens_brackets_fst = cdrewrite(
        union(*[cross(sym, MARKER_ACC + sym + MARKER_ACC) for sym in parens_chars]),
        "", "", sigma_star
    ).optimize()

    # === Double dashes ===
    double_dashes_fst = cdrewrite(
        cross(DASH + DASH, MARKER_ACC + DASH + DASH + MARKER_ACC),
        "", "", sigma_star
    ).optimize()

    # === Clitics and contractions ===
    NON_APOS_OR_SPACE_OR_MARKER = pynini.difference(sigma, union(APOS, SPACE, MARKER_ACC)).optimize()
    # Right context for clitics: space, marker, apostrophe (for 's' followed by quote), or EOS
    SEP_CHARS = union(MARKER_ACC, SPACE, APOS, "[EOS]")

    # Clitics: 's 'm 'd
    clitics_1 = [cs(c) for c in ["'s", "'m", "'d", "'S", "'M", "'D"]]
    endq_3 = cdrewrite(
        union(*[cross(clit, MARKER_ACC + clit) for clit in clitics_1]),
        NON_APOS_OR_SPACE_OR_MARKER, SEP_CHARS, sigma_star
    )

    # Double clitics and n't - must come before apos_rule
    clitics_2 = [
        cs("'ll"), cs("'LL"),
        cs("'re"), cs("'RE"),
        cs("'ve"), cs("'VE"),
        cs("n't"), cs("N'T"),
    ]
    endq_4 = cdrewrite(
        union(*[cross(clit, MARKER_ACC + clit) for clit in clitics_2]),
        NON_APOS_OR_SPACE_OR_MARKER, SEP_CHARS, sigma_star
    )

    # Standalone apostrophe at word end (before space or EOS)
    # NLTK rule: ([^' ])(' ) - separates trailing ' before space
    SEP_CHARS_OR_EOS = union(MARKER_ACC, SPACE, "[EOS]")
    apos_rule = cdrewrite(
        cross(APOS, MARKER_ACC + APOS),
        NON_APOS_OR_SPACE_OR_MARKER, SEP_CHARS_OR_EOS, sigma_star
    )

    clitics_fst = (endq_3 @ endq_4 @ apos_rule).optimize()

    # === Contractions ===
    SEP_OR_BOS = union(MARKER_ACC, SPACE, "[BOS]")

    # NLTK uses (?i) so all contractions are case-insensitive.
    # We enumerate lowercase, Title, and UPPER; mixed-case is rare enough to skip.
    contractions_base = [
        ("cannot", "can", "not"),
        ("gonna", "gon", "na"),
        ("gotta", "got", "ta"),
        ("lemme", "lem", "me"),
        ("wanna", "wan", "na"),
        ("gimme", "gim", "me"),
        ("d'ye", "d", "'ye"),
        ("more'n", "more", "'n"),
    ]
    contractions_raw = []
    for orig, p1, p2 in contractions_base:
        contractions_raw.append((orig, p1, p2))                          # lowercase
        contractions_raw.append((orig.capitalize(), p1.capitalize(), p2)) # Title
        contractions_raw.append((orig.upper(), p1.upper(), p2.upper()))   # UPPER

    contractions_fsts = []
    for orig, part1, part2 in contractions_raw:
        rule = cdrewrite(
            cross(cs(orig), MARKER_ACC + cs(part1) + MARKER_ACC + MARKER_ACC + cs(part2) + MARKER_ACC),
            SEP_OR_BOS, SEP_CHARS, sigma_star
        )
        contractions_fsts.append(rule)

    contractions_fst = contractions_fsts[0]
    for c in contractions_fsts[1:]:
        contractions_fst = contractions_fst @ c
    contractions_fst = contractions_fst.optimize()

    # === CONTRACTIONS3: 'tis, 'twas ===
    # NLTK regex: ('t)(is)\b and ('t)(was)\b — preceded by space (text is padded)
    contractions3_raw = [("'tis", "'t", "is"), ("'twas", "'t", "was"),
                         ("'Tis", "'T", "is"), ("'Twas", "'T", "was"),
                         ("'TIS", "'T", "IS"), ("'TWAS", "'T", "WAS")]
    contractions3_fsts = []
    for orig, p1, p2 in contractions3_raw:
        rule = cdrewrite(
            cross(cs(orig), MARKER_ACC + cs(p1) + MARKER_ACC + MARKER_ACC + cs(p2) + MARKER_ACC),
            SEP_OR_BOS, SEP_CHARS, sigma_star
        )
        contractions3_fsts.append(rule)
    contractions3_fst = contractions3_fsts[0]
    for c in contractions3_fsts[1:]:
        contractions3_fst = contractions3_fst @ c
    contractions3_fst = contractions3_fst.optimize()

    # === Space to marker ===
    space_to_marker = cdrewrite(
        cross(SPACE, MARKER_ACC),
        "", "", sigma_star
    ).optimize()

    # === Compose all rules ===
    print("Composing PTB rules...")
    core_fst = identity_fst
    core_fst = (core_fst @ quotes_fst).optimize()  # Handle all quote types
    core_fst = (core_fst @ punct_fst).optimize()
    core_fst = (core_fst @ parens_brackets_fst).optimize()
    core_fst = (core_fst @ double_dashes_fst).optimize()
    core_fst = (core_fst @ clitics_fst).optimize()  # Handle clitics ('s, 'll, n't, etc.)
    core_fst = (core_fst @ contractions_fst).optimize()
    core_fst = (core_fst @ contractions3_fst).optimize()
    core_fst = (core_fst @ space_to_marker).optimize()

    print(f"Core PTB FST: {core_fst.num_states()} states")

    # === Separator inserter ===
    core_fst.set_input_symbols(ext_symbols)
    core_fst.set_output_symbols(ext_symbols)

    sep_fst = _build_separator_inserter(symbols, ext_symbols)

    final_fst = (core_fst @ sep_fst).optimize()
    print(f"Final pynini FST: {final_fst.num_states()} states")

    # === Convert to native FST ===
    print("Converting to native FST...")
    native_fst = FST()

    for state in final_fst.states():
        native_fst.states.add(state)

    native_fst.add_start(final_fst.start())

    for state in final_fst.states():
        final_weight = final_fst.final(state)
        if final_weight != pynini.Weight.zero(final_fst.weight_type()):
            native_fst.add_stop(state)

    marker_id = ext_symbols.find(MARKER)
    for state in final_fst.states():
        for arc in final_fst.arcs(state):
            # Skip arcs with MARKER as input (unreachable — input never contains MARKER)
            if arc.ilabel == marker_id:
                continue
            input_sym = NATIVE_EPSILON if arc.ilabel == 0 else ext_symbols.find(arc.ilabel)
            output_sym = NATIVE_EPSILON if arc.olabel in (0, marker_id) else ext_symbols.find(arc.olabel)
            native_fst.add_arc(state, input_sym, output_sym, arc.nextstate)

    # Summary statistics
    total_arcs = 0
    eps_in = eps_out = marker_in = marker_out = eos_in = eos_out = 0
    for s in native_fst.states:
        for (i, o, t) in native_fst.arcs(s):
            total_arcs += 1
            if i == NATIVE_EPSILON: eps_in += 1
            if o == NATIVE_EPSILON: eps_out += 1
            if i == MARKER: marker_in += 1
            if o == MARKER: marker_out += 1
            if i == "[EOS]": eos_in += 1
            if o == "[EOS]": eos_out += 1
    print(f"Native FST: {len(native_fst.states)} states, {total_arcs} arcs")
    print(f"  eps: {eps_in} in, {eps_out} out")
    print(f"  MARKER: {marker_in} in, {marker_out} out")
    print(f"  [EOS]: {eos_in} in, {eos_out} out")

    return native_fst


def test_ptb_pynini():
    """Test the pynini-based PTB FST against NLTK."""
    try:
        from nltk.tokenize import TreebankWordTokenizer
    except ImportError:
        print("NLTK not available for testing")
        return

    nltk_tokenizer = TreebankWordTokenizer()

    test_cases = [
        "Hello, world!",
        "I can't do it.",
        'She said "hello" to me.',
        "It's a test -- really!",
        "Don't you think?",
        "I'll go there.",
        "We've been here.",
        "CAN'T STOP WON'T STOP",
    ]

    print("Building FST...")
    fst = build_ptb_fst_pynini()
    print(f"FST: {len(fst.states)} states\n")

    print("Testing:\n")
    matches = 0
    for text in test_cases:
        nltk_tokens = nltk_tokenizer.tokenize(text)
        nltk_result = ' '.join(nltk_tokens)

        byte_strs = string_to_byte_strs(text)
        try:
            from transduction.benchmarking.fst_utils import fst_output_language
            output = next(fst_output_language(fst, byte_strs))
            fst_result = decode_ptb_output(output)
        except StopIteration:
            fst_result = "<REJECTED>"
        except Exception as e:
            fst_result = f"<ERROR: {e}>"

        match = "✓" if fst_result == nltk_result else "✗"
        if fst_result == nltk_result:
            matches += 1

        print(f"{match} Input: {text}")
        if fst_result != nltk_result:
            print(f"  NLTK: {nltk_result}")
            print(f"  FST:  {fst_result}")
        print()

    print(f"Matches: {matches}/{len(test_cases)}")
    return fst


if __name__ == "__main__":
    test_ptb_pynini()
