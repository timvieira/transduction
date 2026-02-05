import os
from typing import List, Dict, Any

import pynini
from pynini import cross, cdrewrite, union

from tokenizer_conversion import compute_universal_states, TransducedLM


from tokenizer_conversion.transducedLM.utils.utils import _log
from tokenizer_conversion.transducedLM.utils.constants import EPS, SEP

from tokenizer_conversion.transducedLM.logp_next.precompute import precompute_next_pset
from tokenizer_conversion.transducers.utils.construction import (
    char_to_byte,
    chars_to_bytes,
)
from tokenizer_conversion.transducers.utils.properties import calculate_num_states_arcs


def aggregate_ptb_words(
    transduced_tokens: List[
        str
    ],  # transduced tokens as strings (e.g., "97", "258", "259")
    log_probs: List[float],  # same length, aligned to chars
    bow_token: str = "258",
    eow_token: str = "259",
) -> List[Dict[str, Any]]:
    if len(transduced_tokens) != len(log_probs):
        raise ValueError("chars and log_probs must be the same length")

    words: List[Dict[str, Any]] = []
    cur_chars: List[str] = []
    cur_lps: List[float] = []
    cur_start_tr: int | None = None  # start index in *transduced* stream
    in_word = False
    has_content = False  # saw any non-(BOW/EOW) token
    content_bytes: List[int] = []  # byte ints for decoding (excludes BOW/EOW)

    def _decode_content(bs: List[int]) -> str:
        try:
            return bytes(bs).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return bytes(bs).decode("utf-8", errors="ignore")

    def flush(end_tr: int | None = None) -> None:
        nonlocal cur_chars, cur_lps, in_word, has_content, cur_start_tr, content_bytes
        if not cur_chars or cur_start_tr is None:
            # nothing open
            cur_chars.clear()
            cur_lps.clear()
            in_word = False
            has_content = False
            cur_start_tr = None
            content_bytes.clear()
            return
        end_idx = (len(cur_chars) - 1) if end_tr is None else (end_tr - cur_start_tr)
        # Compose result
        words.append(
            dict(
                unit=_decode_content(content_bytes),  # excludes BOW/EOW
                logprob=float(sum(cur_lps)),  # includes BOW & EOW
                chars=cur_chars.copy(),  # span incl. BOW/EOW
                char_logprobs=cur_lps.copy(),
                start_tr=cur_start_tr,  # NEW: transduced start index
                end_tr=cur_start_tr + end_idx,  # NEW: transduced end index (inclusive)
                n_tr_tokens=len(cur_chars),  # NEW: count incl. separators
                n_bytes=len(content_bytes),  # NEW: content bytes only
                bytes=[
                    str(b) for b in content_bytes
                ],  # NEW: content byte IDs as strings
            )
        )
        # reset
        cur_chars.clear()
        cur_lps.clear()
        in_word = False
        has_content = False
        cur_start_tr = None
        content_bytes.clear()

    for i, (ch, lp) in enumerate(zip(transduced_tokens, log_probs)):
        lp = float(lp)
        if ch == bow_token:
            if not in_word:
                in_word = True
                cur_start_tr = i
            elif has_content:
                # start new word — close previous first
                flush(end_tr=i - 1)
                in_word = True
                cur_start_tr = i
            # include BOW token prob/span
            cur_chars.append(ch)
            cur_lps.append(lp)

        elif ch == eow_token:
            if not in_word:
                raise ValueError("EOW encountered without an open word")
            # include EOW token then close the word
            cur_chars.append(ch)
            cur_lps.append(lp)
            flush(end_tr=i)

        else:
            # ordinary byte (stringified int 0..255)
            if not in_word:
                in_word = True
                cur_start_tr = i
            cur_chars.append(ch)
            cur_lps.append(lp)
            try:
                content_bytes.append(int(ch))
                has_content = True
            except ValueError:
                # ignore non-numeric tokens as content
                pass

    # If stream ends mid-word, close it
    flush()

    # Sanity: total probs accounted for
    total_stream_lp = float(sum(float(x) for x in log_probs))
    total_words_lp = float(sum(w["logprob"] for w in words))
    if abs(total_stream_lp - total_words_lp) > 1e-10:
        raise ValueError(f"Probability leakage: {total_stream_lp} vs. {total_words_lp}")

    return words


def detokenise_ptb(
    stream: list[str],
    keep_space: bool = True,
) -> str:
    out_parts, buf = [], []

    def _flush():
        if buf:
            out_parts.append(bytes(buf).decode("utf-8"))
            buf.clear()

    for tok in stream:
        if tok == SEP:
            _flush()
            out_parts.append(tok)
            continue
        buf.append(int(tok))
    _flush()
    if keep_space:
        return "".join(out_parts)
    else:
        raise ValueError("Not implemented")


def load_ptb(
    directory,
    llm=None,
    model_name=None,
    verbose=True,
) -> TransducedLM:
    if os.path.isdir(directory):
        _log(verbose, "Loading PTB")
        ptb_wrapped = TransducedLM.load(
            path=directory, use_genlm_bytes=True, llm_name=model_name
        )
        compute_universal_states(ptb_wrapped)
        # ptb_wrapped.get_all_fst_arcs() # recompute with special symbols
        ptb = TransducedLM(
            ptb_wrapped.fst, llm=llm, use_genlm_bytes=True, llm_name=model_name
        )
        ptb._universal_set_cache = ptb_wrapped._universal_set_cache
        compute_universal_states(ptb)
        # Separator symbols are in the OUTPUT symbol table only

    else:
        ptb_fst = build_ptb_fst_bytes()
        ptb = TransducedLM(ptb_fst, llm_name=model_name, use_genlm_bytes=True)
        compute_universal_states(ptb)
        # ptb_wrapped.precompute_universal_set(verbose=True)
        ptb.save(directory)
        ptb = load_ptb(directory, llm=llm, model_name=model_name)
    precompute_next_pset(ptb)
    ptb.name = "ptb"
    return ptb


def _build_separator_inserter(
    input_symbols: pynini.SymbolTable,
) -> pynini.Fst:
    """
    Builds an FST that converts marker bytes (byte 0) to END_SEP_CHAR on the output side.
    """
    # Create output symbol table with separators
    output_symbols = pynini.SymbolTable()
    for idx in range(input_symbols.num_symbols()):
        sym = input_symbols.find(idx)
        if sym is not None:
            output_symbols.add_symbol(sym, idx)    
    if output_symbols.find(SEP) == -1:
        output_symbols.add_symbol(SEP)
    
    # Build FST that collapses markers
    fst = pynini.Fst()
    start = fst.add_state()
    in_markers = fst.add_state()
    fst.set_start(start)
    fst.set_final(start)    
    marker_id = input_symbols.find("0")
    sep_end_id = output_symbols.find(SEP)
    eps_id = 0
    
    # From start: first marker -> output END_SEP, go to in_markers state
    arc_first_marker = pynini.Arc(marker_id, sep_end_id, pynini.Weight.one("tropical"), in_markers)
    fst.add_arc(start, arc_first_marker)
    # From in_markers: more markers -> consume without output (epsilon), stay in in_markers
    arc_more_markers = pynini.Arc(marker_id, eps_id, pynini.Weight.one("tropical"), in_markers)
    fst.add_arc(in_markers, arc_more_markers)
    
    # From in_markers: any non-marker -> output it and return to start
    for idx in range(input_symbols.num_symbols()):
        sym = input_symbols.find(idx)
        if sym is not None and idx != marker_id and idx != eps_id:
            arc = pynini.Arc(idx, idx, pynini.Weight.one("tropical"), start)
            fst.add_arc(in_markers, arc)
    # From start: any non-marker -> identity arc, stay in start (including spaces!)
    for idx in range(input_symbols.num_symbols()):
        sym = input_symbols.find(idx)
        if sym is not None and idx != marker_id and idx != eps_id:
            arc = pynini.Arc(idx, idx, pynini.Weight.one("tropical"), start)
            fst.add_arc(start, arc)
    fst.set_final(in_markers)
    fst.set_input_symbols(input_symbols)
    fst.set_output_symbols(output_symbols)
    return fst.optimize()


def build_ptb_fst_bytes() -> pynini.Fst:
    """
    Builds an FST that mimics the Penn Treebank tokenizer:
    https://www.nltk.org/_modules/nltk/tokenize/treebank.html#TreebankWordTokenizer

    Uses a two-stage approach:
    1. Build core PTB FST with byte symbols only
    2. Compose with a post-processor that inserts separator symbols on the output side only
    """

    # Build the core PTB FST using a shared symbol table
    # (separators will be added to output table in post-processing)
    symbols = pynini.SymbolTable()
    symbols.add_symbol(EPS)

    for bt in range(256):
        symbols.add_symbol(str(bt), bt + 1)

    # Use a marker byte that PTB will insert, this marker will later be converted to END_SEP
    MARKER_BYTE = "0"  # null byte as word boundary marker
    MARKER = pynini.accep(MARKER_BYTE, token_type=symbols)
    SPACE = char_to_byte(" ", symbols=symbols)

    sigma_chars = []
    for idx in range(1, symbols.num_symbols()):
        sym = symbols.find(idx)
        try:
            sigma_chars.append(pynini.accep(sym, token_type=symbols))
        except Exception as e:
            print(f"Error accepting symbol {sym}: {e}")
            if sym == " ":
                sigma_chars.append(SPACE)

    sigma = pynini.union(*sigma_chars)
    sigma_star = pynini.closure(sigma)

    # DEFINE SPECIAL CHARACTERS
    EOS = "[EOS]"#pynini.union(pynini.accep(EOS, token_type=symbols))
    BOS = "[BOS]"
    SEP_OR_BOS = pynini.union(MARKER, SPACE, BOS)
    SEP_OR_EOS = pynini.union(MARKER, SPACE, EOS)

    APOS = char_to_byte("'", symbols=symbols)
    DIGIT = pynini.union(*[char_to_byte(str(i), symbols=symbols) for i in range(10)])
    DOT = char_to_byte(".", symbols=symbols)

    NON_DOT = pynini.difference(sigma, DOT).optimize()
    NON_DIGIT = pynini.difference(sigma, DIGIT).optimize()
    NON_APOS = pynini.difference(sigma, APOS).optimize()
    NON_APOS_OR_SPACE_OR_MARKER = pynini.difference(
        sigma, pynini.union(APOS, SPACE, MARKER)
    ).optimize()

    # ^\" -> `` checked
    QUOTE = char_to_byte('"', symbols=symbols)
    BACKTICK = char_to_byte("`", symbols=symbols)
    DOUBLE_BACKTICK = BACKTICK + BACKTICK

    start_quotes_1 = cdrewrite(
        cross(QUOTE, DOUBLE_BACKTICK),
        BOS,
        "",
        sigma_star,
    )

    # No special space handling in core PTB - will add separators in post-processing
    # Start with identity FST
    identity_fst = pynini.Fst()
    s = identity_fst.add_state()
    identity_fst.set_start(s)
    identity_fst.set_final(s)
    for idx in range(symbols.num_symbols()):
        if idx != 0:  # skip epsilon
            arc = pynini.Arc(idx, idx, pynini.Weight.one("tropical"), s)
            identity_fst.add_arc(s, arc)
    identity_fst.set_input_symbols(symbols)
    identity_fst.set_output_symbols(symbols)
    identity_fst.closure()  # Allow any sequence

    # (``) -> MARKER `` MARKER -> Checked
    start_quotes_2 = cdrewrite(
        cross(DOUBLE_BACKTICK, MARKER + DOUBLE_BACKTICK + MARKER),
        "",
        "",
        sigma_star,
    )

    # ([ \(\[{<])(\"|\'{2}) -> \1 `` -> checked
    R_BRACKET_L = char_to_byte("(", symbols=symbols)
    BRACKET_L = char_to_byte("[", symbols=symbols)
    BRACE_L = char_to_byte("{", symbols=symbols)
    ANGLE_L = char_to_byte("<", symbols=symbols)

    start_quotes_3 = cdrewrite(
        cross(QUOTE, MARKER + DOUBLE_BACKTICK + MARKER),
        union(
            R_BRACKET_L, BRACKET_L, BRACE_L, ANGLE_L, MARKER, SPACE
        ).plus,
        "",
        sigma_star,
    )

    starting_quotes_fst = start_quotes_1 @ start_quotes_2 @ start_quotes_3

    # ([:,])([^\d]) -> MARKER \1 MARKER -> checked
    COMMA = char_to_byte(",", symbols=symbols)
    COLON = char_to_byte(":", symbols=symbols)

    punct_1 = cdrewrite(
        cross(COMMA, MARKER + COMMA + MARKER)
        | cross(COLON, MARKER + COLON + MARKER),
        "",
        NON_DIGIT,
        sigma_star,
    )

    # ([:,])$ -> r"MARKER \1 MARKER" checked
    punct_2 = cdrewrite(
        cross(COMMA, MARKER + COMMA + MARKER)
        | cross(COLON, MARKER + COLON + MARKER),
        "",
        EOS,
        sigma_star,
    )

    # \.\.\. -> MARKER ... MARKER -> checked
    DOT = char_to_byte(".", symbols=symbols)

    ellipsis_rule = cdrewrite(
        cross(DOT + DOT + DOT, MARKER + DOT + DOT + DOT + MARKER),
        "",
        "",
        sigma_star,
    )

    # [;@#$%&] -> MARKER \g<0> MARKER -> checked
    SEMICOLON = char_to_byte(";", symbols=symbols)
    AT = char_to_byte("@", symbols=symbols)
    PERCENT = char_to_byte("%", symbols=symbols)
    AMPERSAND = char_to_byte("&", symbols=symbols)
    DOLLAR = char_to_byte("$", symbols=symbols)
    special_punct = [
        SEMICOLON,
        AT,
        PERCENT,
        AMPERSAND,
        DOLLAR,
    ]
    spaced_punct = pynini.union(
        *(pynini.cross(sym, MARKER + sym + MARKER) for sym in special_punct)
    )
    punct_4 = cdrewrite(spaced_punct, "", "", sigma_star)

    # ([^\.])(\.)([\]\)}>"']*)\s*$ -> \1 MARKER \2\3 TODO check
    R_BRACKET_R = char_to_byte(")", symbols=symbols)
    punct_5 = cdrewrite(
        cross(DOT, MARKER + DOT),
        NON_DOT,
        union(
            EOS, APOS + APOS + EOS, QUOTE + EOS, APOS + EOS, R_BRACKET_R + EOS
        ),
        sigma_star,
    )

    # [?!] -> MARKER ? MARKER or MARKER ! MARKER
    QUESTION = char_to_byte("?", symbols=symbols)
    EXCLAMATION = char_to_byte("!", symbols=symbols)
    punct_6 = cdrewrite(
        cross(QUESTION, MARKER + QUESTION + MARKER)
        | cross(EXCLAMATION, MARKER + EXCLAMATION + MARKER),
        "",
        "",
        sigma_star,
    )

    # ([^'])'  -> \1 MARKER '
    punct_7 = cdrewrite(
        cross(APOS, MARKER + APOS),
        NON_APOS,
        pynini.union(MARKER, SPACE, EOS),
        sigma_star,
    )

    punct_fst = (
        punct_1 @ punct_2 @ ellipsis_rule @ punct_4 @ punct_5 @ punct_6 @ punct_7
    )

    # r"[\]\[\(\)\{\}\<\>]"), r"MARKER \g<0> MARKER" -> checked
    BRACKET_R = char_to_byte("]", symbols=symbols)
    BRACE_R = char_to_byte("}", symbols=symbols)
    ANGLE_R = char_to_byte(">", symbols=symbols)
    parens_chars = [
        BRACKET_R,
        BRACKET_L,
        R_BRACKET_R,
        R_BRACKET_L,
        BRACE_R,
        BRACE_L,
        ANGLE_R,
        ANGLE_L,
    ]

    spaced_parens = pynini.union(
        *(pynini.cross(sym, MARKER + sym + MARKER) for sym in parens_chars)
    )
    parens_brackets_fst = cdrewrite(
        spaced_parens,
        "",
        "",
        sigma_star,
    )

    # (r"--"), r"MARKER -- MARKER" -> checked
    DASH = char_to_byte("-", symbols=symbols)
    DASH_DASH = DASH + DASH
    double_dashes_fst = cdrewrite(
        cross(DASH_DASH, MARKER + DASH_DASH + MARKER),
        "",
        "",
        sigma_star,
    )

    # (r"''"), "MARKER '' MARKER" -> checked
    endq_1 = cdrewrite(
        cross(APOS + APOS, MARKER + APOS + APOS + MARKER), "", "", sigma_star
    )

    # (r'"'), "MARKER '' MARKER" -> checked
    endq_2 = cdrewrite(
        cross(QUOTE, MARKER + APOS + APOS + MARKER), "", "", sigma_star
    )

    # accept clitics

    SMALL_S = char_to_byte("s", symbols=symbols)
    SMALL_M = char_to_byte("m", symbols=symbols)
    SMALL_D = char_to_byte("d", symbols=symbols)
    CAPS_S = char_to_byte("S", symbols=symbols)
    CAPS_M = char_to_byte("M", symbols=symbols)
    CAPS_D = char_to_byte("D", symbols=symbols)

    clitics_1 = [
        APOS + SMALL_S,
        APOS + SMALL_M,
        APOS + SMALL_D,
        APOS + CAPS_S,
        APOS + CAPS_M,
        APOS + CAPS_D,
    ]

    # approx (r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 MARKER \2 " checked
    endq_3 = cdrewrite(
        pynini.union(*(pynini.cross(clit, MARKER + clit) for clit in clitics_1)),
        NON_APOS_OR_SPACE_OR_MARKER,
        SEP_OR_EOS,
        sigma_star,
    )
    apos = cdrewrite(
        pynini.cross(APOS, MARKER + APOS),
        NON_APOS_OR_SPACE_OR_MARKER,
        SEP_OR_EOS,
        sigma_star,
    )
    # Check capitalized
    # approx (r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 MARKER \2 " -> checked
    SMALL_L = char_to_byte("l", symbols=symbols)
    CAPS_L = char_to_byte("L", symbols=symbols)
    SMALL_R = char_to_byte("r", symbols=symbols)
    CAPS_R = char_to_byte("R", symbols=symbols)
    SMALL_E = char_to_byte("e", symbols=symbols)
    CAPS_E = char_to_byte("E", symbols=symbols)
    SMALL_V = char_to_byte("v", symbols=symbols)
    CAPS_V = char_to_byte("V", symbols=symbols)
    SMALL_N = char_to_byte("n", symbols=symbols)
    CAPS_N = char_to_byte("N", symbols=symbols)
    SMALL_T = char_to_byte("t", symbols=symbols)
    CAPS_T = char_to_byte("T", symbols=symbols)
    clitics_2 = [
        APOS + SMALL_L + SMALL_L,
        APOS + CAPS_L + CAPS_L,
        APOS + SMALL_R + SMALL_E,
        APOS + CAPS_R + CAPS_E,
        APOS + SMALL_V + SMALL_E,
        APOS + CAPS_V + CAPS_E,
        SMALL_N + APOS + SMALL_T,
        CAPS_N + APOS + CAPS_T,
    ]

    # clitics_2 = ["'ll", "'LL", "'re", "'RE", "'ve", "'VE", "n't", "N'T"]
    endq_4 = cdrewrite(
        pynini.union(*(pynini.cross(clit, MARKER + clit) for clit in clitics_2)),
        NON_APOS_OR_SPACE_OR_MARKER,
        SEP_OR_EOS,
        sigma_star,
    )

    ending_quotes_fst = endq_1 @ endq_2 @ endq_3 @ apos @ endq_4

    # Use MacIntyreContractions, CONTRACTIONS2 and CONTRACTIONS3
    contractions_raw_patterns = [
        ("cannot", ("can", "not")),
        ("d'ye", ("d", "'ye")),
        ("more'n", ("more", "'n")),
        ("'tis", ("'t", "is")),
        ("'twas", ("'t", "was")),
        ("gonna", ("gon", "na")),
        ("gotta", ("got", "ta")),
        ("lemme", ("lem", "me")),
        ("wanna", ("wan", "na")),
        ("gimme", ("gim", "me")),
        # Upper case
        ("Cannot", ("Can", "not")),
        ("D'ye", ("D", "'ye")),
        ("More'n", ("More", "'n")),
        ("Gonna", ("Gon", "na")),
        ("Gotta", ("Got", "ta")),
        ("Lemme", ("Lem", "me")),
        ("Wanna", ("Wan", "na")),
        ("Gimme", ("Gim", "me")),
    ]
    contractions_patterns = []
    for pattern in contractions_raw_patterns:
        input_str = pattern[0]
        pieces = pattern[1]
        accepted_input = chars_to_bytes(input_str, symbols=symbols)
        accepted_pieces0 = chars_to_bytes(pieces[0], symbols=symbols)
        accepted_pieces1 = chars_to_bytes(pieces[1], symbols=symbols)

        contractions_patterns.append(
            (accepted_input, (accepted_pieces0, accepted_pieces1))
        )

    contractions2_fsts = [
        cdrewrite(
            cross(
                orig, MARKER + pieces[0] + MARKER + MARKER + pieces[1] + MARKER
            ),
            SEP_OR_BOS,
            SEP_OR_EOS,
            sigma_star,
        )
        for orig, pieces in contractions_patterns
    ]
    if contractions2_fsts:
        contractions_fst = contractions2_fsts[0]
        for c2 in contractions2_fsts[1:]:
            contractions_fst @= c2
    else:
        contractions_fst = pynini.accep("", token_type=symbols)

    # Build core PTB FST (with spaces, no separators yet)
    core_ptb_fst = identity_fst
    core_ptb_fst @= starting_quotes_fst.optimize()
    core_ptb_fst @= punct_fst.optimize()
    core_ptb_fst @= parens_brackets_fst.optimize()
    core_ptb_fst @= double_dashes_fst.optimize()
    core_ptb_fst @= ending_quotes_fst.optimize()
    core_ptb_fst @= contractions_fst.optimize()

    core_ptb_fst.set_input_symbols(symbols)
    core_ptb_fst.set_output_symbols(symbols)
    core_ptb_fst.optimize()
    
    print(f"Core PTB FST: {core_ptb_fst.num_states()} states")

    # Build separator insertion post-processor
    # This FST wraps each space with SEP_BEGIN (before) and SEP_END (after) on OUTPUT only
    separator_fst = _build_separator_inserter(
        symbols
    )
    
    print(f"Separator inserter: {separator_fst.num_states()} states")

    # Compose: input -> core_ptb -> separator_inserter -> output
    final_fst = pynini.compose(core_ptb_fst, separator_fst).optimize()

    print(f"Final composed FST: {final_fst.num_states()} states")
    # Get number of arcs
    print(f"Final composed FST: {calculate_num_states_arcs(final_fst)} states and arcs")
    
    # Debugging - check for separator arcs in final FST
    sep_end_id_output = final_fst.output_symbols().find(SEP)

    num_sep_begin_out_arcs = 0
    num_sep_end_out_arcs = 0
    num_sep_begin_in_arcs = 0
    num_sep_end_in_arcs = 0

    num_out_eps_arcs = 0
    num_in_eps_arcs = 0

    for state in final_fst.states():
        for arc in final_fst.arcs(state):
            if arc.olabel == sep_end_id_output and sep_end_id_output != -1:
                num_sep_end_out_arcs += 1
            if arc.ilabel == sep_end_id_output and sep_end_id_output != -1:
                num_sep_end_in_arcs += 1
            if arc.olabel == 0:
                num_out_eps_arcs += 1
            if arc.ilabel == 0:
                num_in_eps_arcs += 1
    print(
        f"\nFinal PTB FST Arc Counts:"
        f"\n  SEP_BEGIN output arcs: {num_sep_begin_out_arcs}"
        f"\n  SEP_END output arcs: {num_sep_end_out_arcs}"
        f"\n  SEP_BEGIN input arcs: {num_sep_begin_in_arcs}"
        f"\n  SEP_END input arcs: {num_sep_end_in_arcs}"
        f"\n  OUT_EPS arcs: {num_out_eps_arcs}"
        f"\n  IN_EPS arcs: {num_in_eps_arcs}"
    )
    
    return final_fst