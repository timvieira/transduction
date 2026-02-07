from transduction.fst import FST
from transduction.fsa import EPSILON


# Special symbols - use strings to match pynini convention and FSA.language compatibility
MARKER = '0'  # Null byte used as internal word boundary marker (string)
SEP = '258'   # Output separator symbol (word boundary in final output)


def build_byte_alphabet():
    """Build the set of all byte values as strings (1-255, excluding 0 which is MARKER)."""
    return {str(b) for b in range(1, 256)}


def char_to_byte_str(ch):
    """Convert a single ASCII character to its byte value as string."""
    return str(ord(ch))


def string_to_byte_strs(s):
    """Convert a string to tuple of byte value strings."""
    return tuple(str(b) for b in s.encode('utf-8'))


def build_ptb_fst_simple():
    """
    Build a simplified PTB tokenizer FST.

    This version handles the most common cases:
    - Add word boundaries around punctuation
    - Normalize quotes
    - Handle ellipsis and double dashes

    Input: byte value strings (e.g., "72" for 'H')
    Output: byte value strings + SEP ("258") for word boundaries
    """
    fst = FST()

    # Special byte strings
    space = char_to_byte_str(' ')  # "32"
    quote = char_to_byte_str('"')  # "34"
    apostrophe = char_to_byte_str("'")  # "39"
    backtick = char_to_byte_str('`')  # "96"
    dot = char_to_byte_str('.')  # "46"
    dash = char_to_byte_str('-')  # "45"

    # Letters for contractions
    char_n = char_to_byte_str('n')
    char_N = char_to_byte_str('N')
    char_t = char_to_byte_str('t')
    char_T = char_to_byte_str('T')
    char_s = char_to_byte_str('s')
    char_S = char_to_byte_str('S')
    char_d = char_to_byte_str('d')
    char_D = char_to_byte_str('D')
    char_m = char_to_byte_str('m')
    char_M = char_to_byte_str('M')
    char_l = char_to_byte_str('l')
    char_L = char_to_byte_str('L')
    char_r = char_to_byte_str('r')
    char_R = char_to_byte_str('R')
    char_e = char_to_byte_str('e')
    char_E = char_to_byte_str('E')
    char_v = char_to_byte_str('v')
    char_V = char_to_byte_str('V')
    

    # Clitic endings: 's, 'd, 'm (single letter after apostrophe)
    clitic_single = {char_s, char_S, char_d, char_D, char_m, char_M}
    # First letters of double-letter clitics: 'l(l), 'r(e), 'v(e)

    # Punctuation that gets spaced
    simple_punct = {char_to_byte_str(c) for c in ';@%&$,:[]{}<>()!?'}

    # States
    START = 'start'  # At word boundary
    WORD = 'word'    # In a word
    WORD_n = 'word_n'  # Just saw 'n' in word (for n't detection)
    WORD_N = 'word_N'  # Just saw 'N' in word (uppercase)
    DOT1 = 'dot1'    # Saw one dot (after word)
    DOT2 = 'dot2'    # Saw two dots
    DASH1 = 'dash1'  # Saw one dash
    APOS = 'apos'    # Saw apostrophe in word (contraction)
    APOS_N = 'apos_n'  # Saw 'n after apostrophe (for n't)
    n_APOS = 'n_apos'  # Saw n then ' (for n't - need to remove n)
    N_APOS = 'N_apos'  # Saw N then ' (uppercase, for N'T)
    # Double-letter clitic states: 'll, 're, 've (lowercase)
    APOS_l = 'apos_l'  # Saw 'l (could be 'll)
    APOS_r = 'apos_r'  # Saw 'r (could be 're)
    APOS_v = 'apos_v'  # Saw 'v (could be 've)
    # Double-letter clitic states: 'LL, 'RE, 'VE (uppercase)
    APOS_L = 'apos_L'  # Saw 'L (could be 'LL)
    APOS_R = 'apos_R'  # Saw 'R (could be 'RE)
    APOS_V = 'apos_V'  # Saw 'V (could be 'VE)
    # Final states for emitting buffered content
    DOT1_END = 'dot1_end'  # Emit period at end of input
    DOT2_END = 'dot2_end'  # Emit two periods at end of input
    DASH1_END = 'dash1_end'  # Emit dash at end of input
    APOS_END = 'apos_end'  # Emit apostrophe at end of input
    APOS_N_END = 'apos_n_end'  # Emit 'n at end of input
    WORD_n_END = 'word_n_end'  # End after n
    WORD_N_END = 'word_N_end'  # End after N (uppercase)
    APOS_l_END = 'apos_l_end'  # End after 'l
    APOS_r_END = 'apos_r_end'  # End after 'r
    APOS_v_END = 'apos_v_end'  # End after 'v
    APOS_L_END = 'apos_L_end'  # End after 'L
    APOS_R_END = 'apos_R_end'  # End after 'R
    APOS_V_END = 'apos_V_end'  # End after 'V

    fst.add_I(START)
    fst.add_F(START)
    fst.add_F(WORD)
    fst.add_F(WORD_n_END)
    fst.add_F(WORD_N_END)
    fst.add_F(DOT1_END)
    fst.add_F(DOT2_END)
    fst.add_F(DASH1_END)
    fst.add_F(APOS_END)
    fst.add_F(APOS_N_END)
    fst.add_F(APOS_l_END)
    fst.add_F(APOS_r_END)
    fst.add_F(APOS_v_END)
    fst.add_F(APOS_L_END)
    fst.add_F(APOS_R_END)
    fst.add_F(APOS_V_END)

    # All byte value strings (1-255, as strings)
    alphabet = build_byte_alphabet()

    # Helper to create intermediate states for multi-output
    state_id = [0]
    def new_state():
        state_id[0] += 1
        return f's{state_id[0]}'

    def emit_sequence(from_state, input_sym, outputs, to_state):
        """Emit a sequence of outputs for a single input."""
        if len(outputs) == 0:
            fst.add_arc(from_state, input_sym, EPSILON, to_state)
        elif len(outputs) == 1:
            fst.add_arc(from_state, input_sym, outputs[0], to_state)
        else:
            # First output with input
            s = new_state()
            fst.add_arc(from_state, input_sym, outputs[0], s)
            # Don't mark intermediate states as final - they have incomplete output
            # Middle outputs with epsilon input
            for out in outputs[1:-1]:
                s_next = new_state()
                fst.add_arc(s, EPSILON, out, s_next)
                s = s_next
            # Last output
            fst.add_arc(s, EPSILON, outputs[-1], to_state)

    # Define special contractions: (word, split_position)
    contractions = [
        ("cannot", 3),      # can + not
        ("Cannot", 3),      # Can + not
        ("gonna", 3),       # gon + na
        ("Gonna", 3),       # Gon + na
        ("gotta", 3),       # got + ta
        ("Gotta", 3),       # Got + ta
        ("lemme", 3),       # lem + me
        ("Lemme", 3),       # Lem + me
        ("wanna", 3),       # wan + na
        ("Wanna", 3),       # Wan + na
        ("gimme", 3),       # gim + me
        ("Gimme", 3),       # Gim + me
    ]

    # Build a trie of contractions for deterministic matching
    # trie[prefix] = {char: next_prefix} or {None: (word, split_at)} for terminals
    class TrieNode:
        def __init__(self):
            self.children = {}  # char -> TrieNode
            self.terminal = None  # (word_bytes, split_output) if this is end of a word

    trie_root = TrieNode()

    for word, split_at in contractions:
        word_bytes = [char_to_byte_str(c) for c in word]
        first_part = word_bytes[:split_at]
        second_part = word_bytes[split_at:]
        split_output = first_part + [SEP] + second_part

        node = trie_root
        for char_byte in word_bytes:
            if char_byte not in node.children:
                node.children[char_byte] = TrieNode()
            node = node.children[char_byte]
        node.terminal = (word_bytes, split_output)

    # Create FST states for each trie node using BFS
    # state_name -> (TrieNode, prefix_bytes)
    trie_states = {}
    trie_state_id = [0]

    def get_trie_state(prefix_bytes):
        key = tuple(prefix_bytes)
        if key not in trie_states:
            trie_state_id[0] += 1
            state_name = f"trie_{trie_state_id[0]}"
            trie_states[key] = state_name
        return trie_states[key]

    # BFS to create all trie states and transitions
    from collections import deque
    queue = deque()

    # Initialize: for each first character, create transition from START
    contraction_first_chars = set()
    for char_byte, child_node in trie_root.children.items():
        contraction_first_chars.add(char_byte)
        first_state = get_trie_state([char_byte])
        queue.append((child_node, [char_byte], first_state))

    # Process trie nodes
    while queue:
        node, prefix_bytes, state_name = queue.popleft()

        # If this is a terminal node, add transitions for word completion
        if node.terminal:
            word_bytes, split_output = node.terminal

            # Handle what comes after the completed word
            handled_chars = set(node.children.keys())

            # Check if word ends with 'n' or 'N' for n't handling
            ends_with_n = word_bytes and word_bytes[-1] == char_n
            ends_with_N = word_bytes and word_bytes[-1] == char_N

            for b in alphabet:
                if b in handled_chars:
                    continue  # Will be handled by child transitions
                elif b == space:
                    emit_sequence(state_name, b, split_output + [SEP], START)
                elif b == apostrophe:
                    # Special handling for n't: if word ends with n, could be n't
                    if ends_with_n:
                        # Emit split output minus final 'n', go to n_APOS
                        emit_sequence(state_name, b, split_output[:-1], n_APOS)
                    elif ends_with_N:
                        emit_sequence(state_name, b, split_output[:-1], N_APOS)
                    else:
                        emit_sequence(state_name, b, split_output, APOS)
                elif b in simple_punct:
                    emit_sequence(state_name, b, split_output + [SEP, b, SEP], START)
                elif b == dot:
                    emit_sequence(state_name, b, split_output, DOT1)
                elif b == quote:
                    emit_sequence(state_name, b, split_output + [SEP, apostrophe, apostrophe, SEP], START)
                else:
                    # Word continues - don't split
                    emit_sequence(state_name, b, word_bytes + [b], WORD)

            # End of input - emit split output
            end_state = f"{state_name}_end"
            fst.add_F(end_state)
            emit_sequence(state_name, EPSILON, split_output, end_state)

        else:
            # Non-terminal: if input doesn't match any child, emit prefix and continue
            handled_chars = set(node.children.keys())

            # Check if prefix ends with 'n' or 'N' for n't handling
            ends_with_n = prefix_bytes and prefix_bytes[-1] == char_n
            ends_with_N = prefix_bytes and prefix_bytes[-1] == char_N

            for b in alphabet:
                if b in handled_chars:
                    continue  # Will be handled by child transitions
                elif b == space:
                    emit_sequence(state_name, b, prefix_bytes + [SEP], START)
                elif b == apostrophe:
                    # Special handling for n't: if prefix ends with n, go to n_APOS
                    if ends_with_n:
                        emit_sequence(state_name, b, prefix_bytes[:-1], n_APOS)
                    elif ends_with_N:
                        emit_sequence(state_name, b, prefix_bytes[:-1], N_APOS)
                    else:
                        emit_sequence(state_name, b, prefix_bytes, APOS)
                elif b in simple_punct:
                    emit_sequence(state_name, b, prefix_bytes + [SEP, b, SEP], START)
                elif b == dot:
                    emit_sequence(state_name, b, prefix_bytes, DOT1)
                elif b == quote:
                    emit_sequence(state_name, b, prefix_bytes + [SEP, apostrophe, apostrophe, SEP], START)
                else:
                    emit_sequence(state_name, b, prefix_bytes + [b], WORD)

            # End of input - emit prefix
            end_state = f"{state_name}_end"
            fst.add_F(end_state)
            emit_sequence(state_name, EPSILON, prefix_bytes, end_state)

        # Add transitions to children
        for char_byte, child_node in node.children.items():
            child_prefix = prefix_bytes + [char_byte]
            child_state = get_trie_state(child_prefix)
            fst.add_arc(state_name, char_byte, EPSILON, child_state)
            queue.append((child_node, child_prefix, child_state))

    # Collect first characters that start contractions
    contraction_first_states = {
        char_byte: get_trie_state([char_byte])
        for char_byte in contraction_first_chars
    }

    # From START (at word boundary):
    for b in alphabet:
        if b == space:
            # Skip whitespace at word boundary (already have SEP)
            fst.add_arc(START, b, EPSILON, START)
        elif b == quote:
            # Opening quote: " -> `` SEP (separate token)
            emit_sequence(START, b, [backtick, backtick, SEP], START)
        elif b in simple_punct:
            # Punctuation: output directly with SEP after
            emit_sequence(START, b, [b, SEP], START)
        elif b == dot:
            fst.add_arc(START, b, EPSILON, DOT1)
        elif b == dash:
            fst.add_arc(START, b, EPSILON, DASH1)
        elif b in contraction_first_states:
            # First char of a special contraction - go to trie state
            fst.add_arc(START, b, EPSILON, contraction_first_states[b])
        else:
            fst.add_arc(START, b, b, WORD)

    # From WORD (in a word):
    for b in alphabet:
        if b == space:
            # Whitespace ends word, emit SEP
            fst.add_arc(WORD, b, SEP, START)
        elif b == quote:
            # Closing quote: " -> '' with SEP before and after
            emit_sequence(WORD, b, [SEP, apostrophe, apostrophe, SEP], START)
        elif b == apostrophe:
            # Potential contraction - buffer the apostrophe
            fst.add_arc(WORD, b, EPSILON, APOS)
        elif b in simple_punct:
            # Punct after word: SEP punct SEP
            emit_sequence(WORD, b, [SEP, b, SEP], START)
        elif b == dot:
            fst.add_arc(WORD, b, EPSILON, DOT1)
        elif b == dash:
            fst.add_arc(WORD, b, EPSILON, DASH1)
        elif b == char_n:
            # Could be start of n't - buffer the n (lowercase)
            fst.add_arc(WORD, b, EPSILON, WORD_n)
        elif b == char_N:
            # Could be start of N'T - buffer the N (uppercase)
            fst.add_arc(WORD, b, EPSILON, WORD_N)
        else:
            fst.add_arc(WORD, b, b, WORD)

    # From WORD_n (saw 'n' lowercase in word - could be start of n't):
    for b in alphabet:
        if b == apostrophe:
            # n' - could be n't contraction
            fst.add_arc(WORD_n, b, EPSILON, n_APOS)
        elif b == char_n:
            # nn - output first n, buffer second
            fst.add_arc(WORD_n, b, char_n, WORD_n)
        elif b == char_N:
            # nN - output first n, buffer second (uppercase)
            fst.add_arc(WORD_n, b, char_n, WORD_N)
        elif b == space:
            emit_sequence(WORD_n, b, [char_n, SEP], START)
        elif b == quote:
            emit_sequence(WORD_n, b, [char_n, SEP, apostrophe, apostrophe, SEP], START)
        elif b in simple_punct:
            emit_sequence(WORD_n, b, [char_n, SEP, b, SEP], START)
        elif b == dot:
            fst.add_arc(WORD_n, b, char_n, DOT1)
        elif b == dash:
            fst.add_arc(WORD_n, b, char_n, DASH1)
        else:
            # Not n't pattern - output n and continue
            emit_sequence(WORD_n, b, [char_n, b], WORD)

    # From WORD_N (saw 'N' uppercase in word - could be start of N'T):
    for b in alphabet:
        if b == apostrophe:
            # N' - could be N'T contraction
            fst.add_arc(WORD_N, b, EPSILON, N_APOS)
        elif b == char_n:
            # Nn - output first N, buffer second (lowercase)
            fst.add_arc(WORD_N, b, char_N, WORD_n)
        elif b == char_N:
            # NN - output first N, buffer second
            fst.add_arc(WORD_N, b, char_N, WORD_N)
        elif b == space:
            emit_sequence(WORD_N, b, [char_N, SEP], START)
        elif b == quote:
            emit_sequence(WORD_N, b, [char_N, SEP, apostrophe, apostrophe, SEP], START)
        elif b in simple_punct:
            emit_sequence(WORD_N, b, [char_N, SEP, b, SEP], START)
        elif b == dot:
            fst.add_arc(WORD_N, b, char_N, DOT1)
        elif b == dash:
            fst.add_arc(WORD_N, b, char_N, DASH1)
        else:
            # Not N'T pattern - output N and continue
            emit_sequence(WORD_N, b, [char_N, b], WORD)

    # From DOT1 (saw one dot - could be abbreviation or sentence end):
    for b in alphabet:
        if b == dot:
            fst.add_arc(DOT1, b, EPSILON, DOT2)
        elif b == space:
            # Period at end of sentence: SEP . SEP
            emit_sequence(DOT1, b, [SEP, dot, SEP], START)
        else:
            # Abbreviation or mid-sentence: keep dot attached
            emit_sequence(DOT1, b, [dot, b], WORD)

    # From DOT2 (saw two dots):
    for b in alphabet:
        if b == dot:
            # Ellipsis: ... -> SEP ... SEP
            emit_sequence(DOT2, b, [SEP, dot, dot, dot, SEP], START)
        elif b == space:
            # Two dots at end: SEP .. SEP
            emit_sequence(DOT2, b, [SEP, dot, dot, SEP], START)
        else:
            emit_sequence(DOT2, b, [dot, dot, b], WORD)

    # From DASH1 (saw one dash):
    for b in alphabet:
        if b == dash:
            # Double dash: -- -> SEP -- SEP
            emit_sequence(DASH1, b, [SEP, dash, dash, SEP], START)
        elif b == space:
            # Single dash at end: keep attached to word, then SEP
            emit_sequence(DASH1, b, [dash, SEP], START)
        else:
            emit_sequence(DASH1, b, [dash, b], WORD)

    # From APOS (saw apostrophe in word - potential contraction):
    for b in alphabet:
        if b == char_n or b == char_N:
            # Could be 'n (like 'n' in "rock 'n' roll") - buffer the n
            fst.add_arc(APOS, b, EPSILON, APOS_N)
        elif b == char_l:
            # Could be 'll - buffer the l (lowercase)
            fst.add_arc(APOS, b, EPSILON, APOS_l)
        elif b == char_L:
            # Could be 'LL - buffer the L (uppercase)
            fst.add_arc(APOS, b, EPSILON, APOS_L)
        elif b == char_r:
            # Could be 're - buffer the r (lowercase)
            fst.add_arc(APOS, b, EPSILON, APOS_r)
        elif b == char_R:
            # Could be 'RE - buffer the R (uppercase)
            fst.add_arc(APOS, b, EPSILON, APOS_R)
        elif b == char_v:
            # Could be 've - buffer the v (lowercase)
            fst.add_arc(APOS, b, EPSILON, APOS_v)
        elif b == char_V:
            # Could be 'VE - buffer the V (uppercase)
            fst.add_arc(APOS, b, EPSILON, APOS_V)
        elif b in clitic_single:
            # 's, 'd, 'm - separate as clitic: SEP 'x
            emit_sequence(APOS, b, [SEP, apostrophe, b], WORD)
        elif b == space:
            # Apostrophe at end of word (e.g., possessive)
            emit_sequence(APOS, b, [apostrophe, SEP], START)
        elif b == quote:
            # Apostrophe then quote - emit apostrophe, then handle quote
            emit_sequence(APOS, b, [apostrophe, SEP, apostrophe, apostrophe, SEP], START)
        elif b in simple_punct:
            emit_sequence(APOS, b, [apostrophe, SEP, b, SEP], START)
        elif b == dot:
            emit_sequence(APOS, b, [apostrophe], DOT1)
        else:
            # Not a contraction pattern - just a regular apostrophe in word
            emit_sequence(APOS, b, [apostrophe, b], WORD)

    # From n_APOS (saw lowercase n then ' - detecting n't contraction):
    for b in alphabet:
        if b == char_t or b == char_T:
            # n't contraction! Emit SEP n't (splitting off the n from previous word)
            emit_sequence(n_APOS, b, [SEP, char_n, apostrophe, b], WORD)
        elif b in clitic_single:
            # n's, n'd, etc. - the n is part of word, clitic is separate
            emit_sequence(n_APOS, b, [char_n, SEP, apostrophe, b], WORD)
        elif b == space:
            # n' at end
            emit_sequence(n_APOS, b, [char_n, apostrophe, SEP], START)
        elif b == quote:
            emit_sequence(n_APOS, b, [char_n, apostrophe, SEP, apostrophe, apostrophe, SEP], START)
        elif b in simple_punct:
            emit_sequence(n_APOS, b, [char_n, apostrophe, SEP, b, SEP], START)
        elif b == dot:
            emit_sequence(n_APOS, b, [char_n, apostrophe], DOT1)
        else:
            # Not n't - just n' followed by something else
            emit_sequence(n_APOS, b, [char_n, apostrophe, b], WORD)

    # From N_APOS (saw uppercase N then ' - detecting N'T contraction):
    for b in alphabet:
        if b == char_t or b == char_T:
            # N'T contraction! Emit SEP N't (splitting off the N from previous word)
            emit_sequence(N_APOS, b, [SEP, char_N, apostrophe, b], WORD)
        elif b in clitic_single:
            # N's, N'd, etc. - the N is part of word, clitic is separate
            emit_sequence(N_APOS, b, [char_N, SEP, apostrophe, b], WORD)
        elif b == space:
            # N' at end
            emit_sequence(N_APOS, b, [char_N, apostrophe, SEP], START)
        elif b == quote:
            emit_sequence(N_APOS, b, [char_N, apostrophe, SEP, apostrophe, apostrophe, SEP], START)
        elif b in simple_punct:
            emit_sequence(N_APOS, b, [char_N, apostrophe, SEP, b, SEP], START)
        elif b == dot:
            emit_sequence(N_APOS, b, [char_N, apostrophe], DOT1)
        else:
            # Not N'T - just N' followed by something else
            emit_sequence(N_APOS, b, [char_N, apostrophe, b], WORD)

    # From APOS_l (saw 'l lowercase - could be 'll):
    for b in alphabet:
        if b == char_l or b == char_L:
            # 'll clitic: SEP 'll
            emit_sequence(APOS_l, b, [SEP, apostrophe, char_l, b], WORD)
        elif b == space:
            emit_sequence(APOS_l, b, [apostrophe, char_l, SEP], START)
        elif b in simple_punct:
            emit_sequence(APOS_l, b, [apostrophe, char_l, SEP, b, SEP], START)
        elif b == dot:
            emit_sequence(APOS_l, b, [apostrophe, char_l], DOT1)
        else:
            # Not 'll - just 'l followed by something else
            emit_sequence(APOS_l, b, [apostrophe, char_l, b], WORD)

    # From APOS_L (saw 'L uppercase - could be 'LL):
    for b in alphabet:
        if b == char_l or b == char_L:
            # 'LL clitic: SEP 'LL
            emit_sequence(APOS_L, b, [SEP, apostrophe, char_L, b], WORD)
        elif b == space:
            emit_sequence(APOS_L, b, [apostrophe, char_L, SEP], START)
        elif b in simple_punct:
            emit_sequence(APOS_L, b, [apostrophe, char_L, SEP, b, SEP], START)
        elif b == dot:
            emit_sequence(APOS_L, b, [apostrophe, char_L], DOT1)
        else:
            # Not 'LL - just 'L followed by something else
            emit_sequence(APOS_L, b, [apostrophe, char_L, b], WORD)

    # From APOS_r (saw 'r lowercase - could be 're):
    for b in alphabet:
        if b == char_e or b == char_E:
            # 're clitic: SEP 're
            emit_sequence(APOS_r, b, [SEP, apostrophe, char_r, b], WORD)
        elif b == space:
            emit_sequence(APOS_r, b, [apostrophe, char_r, SEP], START)
        elif b in simple_punct:
            emit_sequence(APOS_r, b, [apostrophe, char_r, SEP, b, SEP], START)
        elif b == dot:
            emit_sequence(APOS_r, b, [apostrophe, char_r], DOT1)
        else:
            # Not 're - just 'r followed by something else
            emit_sequence(APOS_r, b, [apostrophe, char_r, b], WORD)

    # From APOS_R (saw 'R uppercase - could be 'RE):
    for b in alphabet:
        if b == char_e or b == char_E:
            # 'RE clitic: SEP 'RE
            emit_sequence(APOS_R, b, [SEP, apostrophe, char_R, b], WORD)
        elif b == space:
            emit_sequence(APOS_R, b, [apostrophe, char_R, SEP], START)
        elif b in simple_punct:
            emit_sequence(APOS_R, b, [apostrophe, char_R, SEP, b, SEP], START)
        elif b == dot:
            emit_sequence(APOS_R, b, [apostrophe, char_R], DOT1)
        else:
            # Not 'RE - just 'R followed by something else
            emit_sequence(APOS_R, b, [apostrophe, char_R, b], WORD)

    # From APOS_v (saw 'v lowercase - could be 've):
    for b in alphabet:
        if b == char_e or b == char_E:
            # 've clitic: SEP 've
            emit_sequence(APOS_v, b, [SEP, apostrophe, char_v, b], WORD)
        elif b == space:
            emit_sequence(APOS_v, b, [apostrophe, char_v, SEP], START)
        elif b in simple_punct:
            emit_sequence(APOS_v, b, [apostrophe, char_v, SEP, b, SEP], START)
        elif b == dot:
            emit_sequence(APOS_v, b, [apostrophe, char_v], DOT1)
        else:
            # Not 've - just 'v followed by something else
            emit_sequence(APOS_v, b, [apostrophe, char_v, b], WORD)

    # From APOS_V (saw 'V uppercase - could be 'VE):
    for b in alphabet:
        if b == char_e or b == char_E:
            # 'VE clitic: SEP 'VE
            emit_sequence(APOS_V, b, [SEP, apostrophe, char_V, b], WORD)
        elif b == space:
            emit_sequence(APOS_V, b, [apostrophe, char_V, SEP], START)
        elif b in simple_punct:
            emit_sequence(APOS_V, b, [apostrophe, char_V, SEP, b, SEP], START)
        elif b == dot:
            emit_sequence(APOS_V, b, [apostrophe, char_V], DOT1)
        else:
            # Not 'VE - just 'V followed by something else
            emit_sequence(APOS_V, b, [apostrophe, char_V, b], WORD)

    # From APOS_N (saw 'n - could be n't):
    for b in alphabet:
        if b == char_t:
            # n't contraction: SEP n't
            emit_sequence(APOS_N, b, [SEP, apostrophe, char_n, char_t], WORD)
        elif b == space:
            # 'n at end of word
            emit_sequence(APOS_N, b, [apostrophe, char_n, SEP], START)
        elif b in simple_punct:
            emit_sequence(APOS_N, b, [apostrophe, char_n, SEP, b, SEP], START)
        elif b == dot:
            emit_sequence(APOS_N, b, [apostrophe, char_n], DOT1)
        else:
            # Not n't - just 'n followed by something else
            emit_sequence(APOS_N, b, [apostrophe, char_n, b], WORD)

    # End-of-input handling: emit buffered content via epsilon transitions
    # DOT1 at end of input: emit SEP . SEP (period ends sentence)
    emit_sequence(DOT1, EPSILON, [SEP, dot, SEP], DOT1_END)
    # DOT2 at end of input: emit SEP .. SEP
    emit_sequence(DOT2, EPSILON, [SEP, dot, dot, SEP], DOT2_END)
    # DASH1 at end of input: emit dash (attached to word)
    emit_sequence(DASH1, EPSILON, [dash], DASH1_END)
    # APOS at end of input: emit apostrophe (e.g., possessive without following char)
    emit_sequence(APOS, EPSILON, [apostrophe], APOS_END)
    # APOS_N at end of input: emit 'n
    emit_sequence(APOS_N, EPSILON, [apostrophe, char_n], APOS_N_END)
    # WORD_n at end of input: emit n (lowercase)
    emit_sequence(WORD_n, EPSILON, [char_n], WORD_n_END)
    # WORD_N at end of input: emit N (uppercase)
    emit_sequence(WORD_N, EPSILON, [char_N], WORD_N_END)
    # n_APOS at end of input: emit n' (lowercase)
    emit_sequence(n_APOS, EPSILON, [char_n, apostrophe], APOS_END)
    # N_APOS at end of input: emit N' (uppercase)
    emit_sequence(N_APOS, EPSILON, [char_N, apostrophe], APOS_END)
    # APOS_l at end of input: emit 'l (lowercase)
    emit_sequence(APOS_l, EPSILON, [apostrophe, char_l], APOS_l_END)
    # APOS_L at end of input: emit 'L (uppercase)
    emit_sequence(APOS_L, EPSILON, [apostrophe, char_L], APOS_L_END)
    # APOS_r at end of input: emit 'r (lowercase)
    emit_sequence(APOS_r, EPSILON, [apostrophe, char_r], APOS_r_END)
    # APOS_R at end of input: emit 'R (uppercase)
    emit_sequence(APOS_R, EPSILON, [apostrophe, char_R], APOS_R_END)
    # APOS_v at end of input: emit 'v (lowercase)
    emit_sequence(APOS_v, EPSILON, [apostrophe, char_v], APOS_v_END)
    # APOS_V at end of input: emit 'V (uppercase)
    emit_sequence(APOS_V, EPSILON, [apostrophe, char_V], APOS_V_END)

    return fst


def decode_ptb_output(output_tuple):
    """
    Decode PTB output tuple to human-readable string.

    Shows word boundaries as '|' and converts byte strings back to characters.
    """
    parts = []
    current_word = []

    for sym in output_tuple:
        if sym == SEP:
            if current_word:
                try:
                    parts.append(bytes(int(b) for b in current_word).decode('utf-8', errors='replace'))
                except:
                    parts.append(f"<{current_word}>")
                current_word = []
            parts.append('|')
        elif sym != EPSILON:
            try:
                byte_val = int(sym)
                if 0 <= byte_val <= 255:
                    current_word.append(sym)
            except (ValueError, TypeError):
                pass

    if current_word:
        try:
            parts.append(bytes(int(b) for b in current_word).decode('utf-8', errors='replace'))
        except:
            parts.append(f"<{current_word}>")

    return ''.join(parts)


if __name__ == '__main__':
    # Test the FST
    print("Building PTB FST (simple version)...")
    fst = build_ptb_fst_simple()
    print(f"States: {len(fst.states)}")
    print(f"Input alphabet: {len(fst.A)}")
    print(f"Output alphabet: {len(fst.B)}")

    # Test with a simple string
    test_str = 'Hello, world!'
    test_bytes = string_to_byte_strs(test_str)

    print(f"\nInput: {test_str}")
    print(f"Input bytes: {test_bytes}")

    # Run through FST
    from transduction.benchmarking.fst_utils import fst_output_language
    try:
        output = next(fst_output_language(fst, test_bytes))
        print(f"Output: {output}")
        print(f"Decoded: {decode_ptb_output(output)}")
    except StopIteration:
        print("No output (FST rejected input)")
