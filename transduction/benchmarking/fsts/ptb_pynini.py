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
    Build an FST that converts every MARKER byte to SEP.

    Uses a simple 1-state transducer (MARKER→SEP, everything else passes through).
    Consecutive MARKERs produce consecutive SEPs, which is fine because
    decode_ptb_output skips empty tokens between consecutive separators.
    """
    fst = pynini.Fst()
    s = fst.add_state()
    fst.set_start(s)
    fst.set_final(s)

    marker_id = ext_symbols.find(MARKER)
    sep_id = ext_symbols.find(SEP)

    # MARKER -> SEP
    fst.add_arc(s, pynini.Arc(marker_id, sep_id, 0, s))

    # All other symbols pass through
    for idx in range(1, symbols.num_symbols()):
        sym = symbols.find(idx)
        if sym is not None and idx != marker_id:
            fst.add_arc(s, pynini.Arc(idx, idx, 0, s))

    fst.set_input_symbols(ext_symbols)
    fst.set_output_symbols(ext_symbols)
    return fst.optimize()


def build_ptb_fst_pynini():
    """
    Build PTB tokenizer FST using pynini, then convert to native format.

    Returns a native FST with ~228 states that implements Penn Treebank tokenization.
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

    DOUBLE_APOS = APOS + APOS  # '' (two apostrophes)

    # Rule 1: " or '' after word chars -> '' (closing quote) - must run FIRST
    closing_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + APOS + APOS + MARKER_ACC) |
        cross(DOUBLE_APOS, MARKER_ACC + APOS + APOS + MARKER_ACC),
        WORD_CHARS,
        "", sigma_star
    )

    # Rule 2: " or '' after space/brackets -> `` (opening quote)
    opening_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC) |
        cross(DOUBLE_APOS, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        opening_context.plus,
        "", sigma_star
    )

    # Rule 3: Remaining " -> `` (for double quotes at beginning of string)
    remaining_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        "", "", sigma_star
    )

    # Rule 4: Remaining '' -> separated '' (for double apostrophes at beginning of string)
    # Unlike ", remaining '' stays as '' (not converted to ``)
    remaining_double_apos_rule = cdrewrite(
        cross(DOUBLE_APOS, MARKER_ACC + APOS + APOS + MARKER_ACC),
        "", "", sigma_star
    )

    # Rule 5: `` -> MARKER `` MARKER (for manually entered backticks)
    backtick_rule = cdrewrite(
        cross(DOUBLE_BACKTICK, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        "", "", sigma_star
    )

    # Compose: closing first, then opening, then remaining quotes, then remaining double apos, then backticks
    quotes_fst = (closing_quote_rule @ opening_quote_rule @ remaining_quote_rule @ remaining_double_apos_rule @ backtick_rule).optimize()

    # === Punctuation ===
    DIGIT = union(*[cb(str(i)) for i in range(10)])
    NON_DIGIT = pynini.difference(sigma, DIGIT).optimize()

    # , and : before non-digit or at end of string
    # NLTK regex: ([:,])([^\d]) and ([:,])$
    punct_comma_colon = cdrewrite(
        cross(cb(","), MARKER_ACC + cb(",") + MARKER_ACC) |
        cross(cb(":"), MARKER_ACC + cb(":") + MARKER_ACC),
        "", union(NON_DIGIT, pynini.accep("[EOS]")), sigma_star
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

    punct_fst = (punct_comma_colon @ ellipsis_rule @ punct_4 @ punct_5 @ punct_6).optimize()

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
    # NLTK uses (?i) case-insensitive matching for all contractions.
    # We build case-preserving transducers that match any case and output
    # the same case as input, using a single entry per contraction.
    eps = pynini.accep("", token_type=symbols)
    eps_to_marker = cross(eps, MARKER_ACC)

    def _ci_char(ch):
        """Case-preserving character transducer: matches upper or lower, outputs same."""
        if ch.isalpha():
            lo = pynini.accep(str(ord(ch.lower())), token_type=symbols)
            hi = pynini.accep(str(ord(ch.upper())), token_type=symbols)
            return union(cross(lo, lo), cross(hi, hi))
        else:
            a = pynini.accep(str(ord(ch)), token_type=symbols)
            return cross(a, a)

    def _ci_contraction(word, split_pos):
        """Case-insensitive, case-preserving contraction transducer."""
        result = eps_to_marker
        for ch in word[:split_pos]:
            result = result + _ci_char(ch)
        result = result + eps_to_marker + eps_to_marker
        for ch in word[split_pos:]:
            result = result + _ci_char(ch)
        return result + eps_to_marker

    # (word, split_position) — split_pos divides word into left/right parts
    contractions_bases = [
        ("cannot", 3), ("gonna", 3), ("gotta", 3),
        ("lemme", 3), ("wanna", 3), ("gimme", 3),
        # NLTK CONTRACTIONS2
        ("d'ye", 1), ("more'n", 4),
        # NLTK CONTRACTIONS3
        ("'tis", 2), ("'twas", 2),
    ]

    all_contraction_tau = union(*[_ci_contraction(w, s) for w, s in contractions_bases])
    # Single cdrewrite with empty left context handles all positions including string start.
    # Right context SEP_CHARS ensures word boundary (prevents matching inside "wannabe" etc.)
    contractions_fst = cdrewrite(
        all_contraction_tau, "", SEP_CHARS, sigma_star
    ).optimize()

    # === Space to marker ===
    space_to_marker = cdrewrite(
        cross(SPACE, MARKER_ACC),
        "", "", sigma_star
    ).optimize()

    # === Compose all rules ===
    print("Composing PTB rules...")
    core_fst = quotes_fst
    core_fst = (core_fst @ punct_fst).optimize()
    core_fst = (core_fst @ parens_brackets_fst).optimize()
    core_fst = (core_fst @ double_dashes_fst).optimize()
    core_fst = (core_fst @ clitics_fst).optimize()  # Handle clitics ('s, 'll, n't, etc.)
    core_fst = (core_fst @ contractions_fst).optimize()
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

    native_fst.add_I(final_fst.start())

    for state in final_fst.states():
        final_weight = final_fst.final(state)
        if final_weight != pynini.Weight.zero(final_fst.weight_type()):
            native_fst.add_F(state)

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
