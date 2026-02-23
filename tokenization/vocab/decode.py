# output of `_bytes_to_unicode`
_encode_bytes_str = [
    'Ā', 'ā', 'Ă', 'ă', 'Ą', 'ą', 'Ć', 'ć', 'Ĉ', 'ĉ', 'Ċ', 'ċ', 'Č', 'č', 'Ď', 'ď',
    'Đ', 'đ', 'Ē', 'ē', 'Ĕ', 'ĕ', 'Ė', 'ė', 'Ę', 'ę', 'Ě', 'ě', 'Ĝ', 'ĝ', 'Ğ', 'ğ',
    'Ġ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
    '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'ġ',
    'Ģ', 'ģ', 'Ĥ', 'ĥ', 'Ħ', 'ħ', 'Ĩ', 'ĩ', 'Ī', 'ī', 'Ĭ', 'ĭ', 'Į', 'į', 'İ', 'ı',
    'Ĳ', 'ĳ', 'Ĵ', 'ĵ', 'Ķ', 'ķ', 'ĸ', 'Ĺ', 'ĺ', 'Ļ', 'ļ', 'Ľ', 'ľ', 'Ŀ', 'ŀ', 'Ł',
    'ł', '¡', '¢', '£', '¤', '¥', '¦', '§', '¨', '©', 'ª', '«', '¬', 'Ń', '®', '¯',
    '°', '±', '²', '³', '´', 'µ', '¶', '·', '¸', '¹', 'º', '»', '¼', '½', '¾', '¿',
    'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï',
    'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß',
    'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï',
    'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ',
]

# this is the inverse mapping of `_bytes_to_unicode`
_decode_str_bytes = {s: i for i, s in enumerate(_encode_bytes_str)}
_default_byte_decoder = _decode_str_bytes


def decode_hf_tokenizer(tokenizer):
    "Extract what we need from a 🤗 tokenizer."
    _merges = []
    V = tokenizer.get_vocab()
    if hasattr(tokenizer, 'bpe_ranks'):
        for (u,v) in tokenizer.bpe_ranks:
            _merges.append((V[u], V[v], V[u + v]))
    else:
        import json
        subtokenizer_dict = json.loads(tokenizer._tokenizer.to_str())
        for (u,v) in subtokenizer_dict["model"]["merges"]:
            _merges.append((V[u], V[v], V[u + v]))

    if hasattr(tokenizer, 'byte_decoder'):
        byte_decoder = tokenizer.byte_decoder
    else:
        byte_decoder = _default_byte_decoder

    _encode = {}
    _decode = [None]*len(V)
    for bs, token_id in V.items():
        b = bytes([byte_decoder[b] for b in bs])
        _encode[b] = token_id
        _decode[token_id] = b

    # map each byte (0-255) to token id (they are annoyingly not the same)
    _encode_byte = [None]*256
    for i in range(256):
        _encode_byte[i] = _encode[bytes([i])]

    return (_merges, _encode, _decode, _encode_byte)


#def _bytes_to_unicode():
#    """Create a mapping from bytes to Unicode characters.
#
#    Returns:
#        dict: Mapping from byte values to Unicode characters
#    """
#
#    # https://chatgpt.com/share/6793c6f3-76cc-800c-97d0-0ae80e9df19f
#    #
#    # In language models like GPT, this mapping is used to tokenize raw input
#    # text into Unicode-compatible strings while preserving all possible byte
#    # sequences. Starting with printable characters ensures that frequent text
#    # remains human-readable, while less frequent or non-printable bytes are
#    # handled gracefully with unique Unicode characters.
#    #
#    # These characters overlap with standard encodings like ASCII and ISO-8859-1
#    # (Latin-1). Ensuring these byte values map directly to readable characters
#    # helps maintain compatibility with text encoded in those formats.
#    #
#    # The initial `bs` includes the following ranges:
#    #
#    # * ord("!") to ord("~"):
#    #
#    #   These correspond to the printable ASCII characters (e.g., !, A-Z, a-z,
#    #   0-9, and symbols like #, $, %).
#    #
#    #   These are among the most commonly used characters in plain text, so
#    #   mapping these bytes directly to themselves ensures intuitive and
#    #   readable output.
#    #
#    # * ord("¡") to ord("¬") and ord("®") to ord("ÿ"):
#    #
#    #   These are printable Latin-1 Supplement characters, including accented
#    #   letters (e.g., é, ü), punctuation marks (e.g., ¡, ¿), and symbols.
#    #
#    #   These characters are important for supporting European languages and are
#    #   frequently used in multilingual text.
#    #
#    # The function avoids including non-printable control characters (e.g.,
#    # 0x00--0x1F in ASCII) in the initial bs list. These characters are not
#    # typically useful for readable text and could cause display or processing
#    # issues if included.
#
#    bs = (
#        list(range(ord("!"), ord("~") + 1))
#        + list(range(ord("¡"), ord("¬") + 1))
#        + list(range(ord("®"), ord("ÿ") + 1))
#    )
#    cs = bs[:]
#    n = 0
#    for b in range(256):
#        if b not in bs:
#            bs.append(b)
#            cs.append(256 + n)
#            n += 1
#    cs = [chr(n) for n in cs]
#    return dict(zip(bs, cs))
