"""
PTB Tokenizer FST built using pynini.

This uses pynini's cdrewrite rules to create a compact FST that matches
NLTK's TreebankWordTokenizer behavior. The pynini FST is then converted to the native
FST format for use with the Rust decomposition backend.

To match NLTK's behavior for period handling, input text must be terminated
with an end-of-string marker (EOS = '\x03'). NLTK's TreebankWordTokenizer only separates periods when
followed by EOS. Use string_to_byte_strs() which automatically appends EOS.

Based on the reference implementation in ptb.py.
"""
import pynini
from pynini import cdrewrite, cross, union

from transduction.fst import FST
from transduction.fsa import EPSILON as NATIVE_EPSILON


# TODO better handling of these symbols, need to change EOS
EPS = "<eps>"
MARKER = '0'  # Null byte used as internal word boundary marker
SEP = '258'   # Output separator symbol (word boundary in final output)
EOS = '3'     # End-of-string marker (ETX byte) - used to detect string end for period handling


def char_to_byte_str(ch):
    """Convert a single ASCII character to its byte value as string."""
    return str(ord(ch))


def string_to_byte_strs(s, add_eos=True):
    """Convert a string to tuple of byte value strings.

    Args:
        s: Input string
        add_eos: If True, append end-of-string marker (required for NLTK-compatible period handling)
    """
    byte_strs = tuple(str(b) for b in s.encode('utf-8'))
    if add_eos:
        byte_strs = byte_strs + (EOS,)
    return byte_strs


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
        elif sym != MARKER and sym != NATIVE_EPSILON and sym != EOS:
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

    marker_id = ext_symbols.find("0")
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
    # Build symbol table for bytes 0-255
    symbols = pynini.SymbolTable()
    symbols.add_symbol(EPS, 0)
    for bt in range(256):
        symbols.add_symbol(str(bt), bt + 1)

    # Extended symbols including SEP for output
    ext_symbols = pynini.SymbolTable()
    ext_symbols.add_symbol(EPS, 0)
    for bt in range(256):
        ext_symbols.add_symbol(str(bt), bt + 1)
    ext_symbols.add_symbol(SEP, 259)

    # Helper function shortcuts
    def cb(ch):
        return _char_to_byte(ch, symbols)

    def cs(s):
        return _chars_to_bytes(s, symbols)

    # Common acceptors
    MARKER_ACC = pynini.accep("0", token_type=symbols)
    EOS_ACC = pynini.accep(EOS, token_type=symbols)  # End-of-string marker
    SPACE = cb(" ")
    APOS = cb("'")
    QUOTE = cb('"')
    DOT = cb(".")
    DASH = cb("-")
    BACKTICK = cb("`")
    DOUBLE_BACKTICK = BACKTICK + BACKTICK

    # Build sigma (all bytes 0-255 including EOS)
    sigma = pynini.union(*[pynini.accep(str(i), token_type=symbols) for i in range(256)])
    sigma_star = pynini.closure(sigma)

    # Identity FST as base (passes through all bytes including EOS)
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

    # EOS stripper - removes the end-of-string marker from output
    eos_strip = cdrewrite(
        cross(EOS_ACC, pynini.accep("", token_type=symbols)),  # Delete EOS
        "", "", sigma_star
    ).optimize()

    # === Quote handling ===
    # We process quotes in a specific order:
    # 1. Convert closing quotes (after word chars) to ''
    # 2. Convert opening quotes (after space/brackets) to ``
    # 3. Convert remaining quotes (e.g., at start of string) to ``
    # 4. Wrap existing `` with markers

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

    # Rule 1: " after word chars -> '' (closing quote) - must run FIRST
    closing_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + APOS + APOS + MARKER_ACC),
        WORD_CHARS,
        "", sigma_star
    )

    # Rule 2: " after space/brackets -> `` (opening quote)
    opening_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        opening_context.plus,
        "", sigma_star
    )

    # Rule 3: Remaining " -> `` (for quotes at beginning of string)
    remaining_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        "", "", sigma_star
    )

    # Rule 4: `` -> MARKER `` MARKER (for manually entered backticks)
    backtick_rule = cdrewrite(
        cross(DOUBLE_BACKTICK, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        "", "", sigma_star
    )

    # Compose: closing first, then opening, then remaining, then backticks
    quotes_fst = (closing_quote_rule @ opening_quote_rule @ remaining_quote_rule @ backtick_rule).optimize()

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
        "", EOS_ACC, sigma_star
    )

    # Ellipsis
    ellipsis_rule = cdrewrite(
        cross(DOT + DOT + DOT, MARKER_ACC + DOT + DOT + DOT + MARKER_ACC),
        "", "", sigma_star
    )

    # Special punct: ; @ % & $
    special_punct = [cb(c) for c in ";@%&$"]
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
    # Note: quotes are already converted to '' by quotes_fst at this point
    # The marker bytes around '' are from the quote conversion
    CLOSING_PUNCT = union(
        cb("]"), cb(")"), cb("}"), cb(">"),
        APOS,  # Single apostrophe
        MARKER_ACC,  # Marker bytes from quote conversion
    )
    CLOSING_PUNCT_STAR = pynini.closure(CLOSING_PUNCT)

    # Optional whitespace (spaces) before EOS
    WHITESPACE_STAR = pynini.closure(SPACE)

    punct_5 = cdrewrite(
        cross(DOT, MARKER_ACC + DOT),
        NON_DOT,
        CLOSING_PUNCT_STAR + WHITESPACE_STAR + EOS_ACC,  # Period followed by optional closing punct, optional spaces, then EOS
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
    SEP_CHARS = union(MARKER_ACC, SPACE, APOS, EOS_ACC)

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
    SEP_CHARS_OR_EOS = union(MARKER_ACC, SPACE, EOS_ACC)
    apos_rule = cdrewrite(
        cross(APOS, MARKER_ACC + APOS),
        NON_APOS_OR_SPACE_OR_MARKER, SEP_CHARS_OR_EOS, sigma_star
    )

    clitics_fst = (endq_3 @ endq_4 @ apos_rule).optimize()

    # === Contractions ===
    SEP_OR_BOS = union(MARKER_ACC, SPACE)

    contractions_raw = [
        ("cannot", "can", "not"),
        ("gonna", "gon", "na"),
        ("gotta", "got", "ta"),
        ("lemme", "lem", "me"),
        ("wanna", "wan", "na"),
        ("gimme", "gim", "me"),
        # Capitalized
        ("Cannot", "Can", "not"),
        ("Gonna", "Gon", "na"),
        ("Gotta", "Got", "ta"),
        ("Lemme", "Lem", "me"),
        ("Wanna", "Wan", "na"),
        ("Gimme", "Gim", "me"),
    ]

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
    core_fst = (core_fst @ space_to_marker).optimize()
    core_fst = (core_fst @ eos_strip).optimize()  # Strip EOS marker from output

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

    native_fst.add_I(final_fst.start())

    for state in final_fst.states():
        final_weight = final_fst.final(state)
        if final_weight != pynini.Weight.zero(final_fst.weight_type()):
            native_fst.add_F(state)

    for state in final_fst.states():
        for arc in final_fst.arcs(state):
            input_sym = NATIVE_EPSILON if arc.ilabel == 0 else ext_symbols.find(arc.ilabel)
            output_sym = NATIVE_EPSILON if arc.olabel == 0 else ext_symbols.find(arc.olabel)
            native_fst.add_arc(state, input_sym, output_sym, arc.nextstate)

    print(f"Native FST: {len(native_fst.states)} states")

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
            output_fsa = fst(byte_strs, None)
            output = next(output_fsa.language(tuple=True))
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
