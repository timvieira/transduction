import re

from datasets import load_dataset


def wikitext_detokenize(string: str) -> str:
    """
    Wikitext is whitespace tokenized and we remove these whitespaces.

    Taken from https://github.com/NVIDIA/Megatron-LM/blob/main/tasks/zeroshot_gpt2/detokenizer.py
    """
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace("  . ", ". ")

    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def load_fasta(T, file_path):
    sequences = []
    current_seq = []
    total_len = 0
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    transformed = [T.aa_map[c] for c in list("".join(current_seq))]
                    sequences.append(transformed)
                    total_len += len(transformed)
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            transformed = [T.aa_map[c] for c in list("".join(current_seq))]
            # byte_seq = [str(c) for c in list("".join(current_seq).encode('utf-8'))]
            sequences.append(transformed)
            total_len += len(transformed)
    return sequences, total_len


def load_wikitext(split):
    return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)


def load_wikitext_paragraphs(T, split, n=4):
    """
    Load the first n paragraphs from the wikitext dataset.
    """
    dataset = load_wikitext(split)
    paragraphs = []
    for item in dataset:
        text = item["text"].strip()
        if text and not text.startswith("="):  # Skip headings
            detokenized = wikitext_detokenize(text)
            transduced = T.apply(detokenized)
            paragraphs.append(transduced)
            if len(paragraphs) == n:
                break
    total_len = 0
    for i, para in enumerate(paragraphs):
        total_len += len(para)
        print(
            f"Paragraph {i+1} len {len(para)} cumulative length {total_len}:\n{para}\n"
        )
    return paragraphs, total_len


def load_wikitext_paragraphs_bytes(
    T,
    split,
    tokenizer,
    n=4,
    join_paragraphs=False,
    text_length=None
):
    """
    Load the first n paragraphs from the wikitext dataset.
    """
    dataset = load_wikitext(split)
    paragraphs = []
    original = []
    lens = 0
    for item in dataset:
        text = item["text"].strip()
        if text and not text.startswith("="):  # Skip headings
            detokenized = wikitext_detokenize(text)
            if text_length:
                detokenized = detokenized[:text_length]
            detokenized = tuple(tokenizer.encode(detokenized))
            from transduction.benchmarking.fst_utils import fst_output_language
            transduced = next(fst_output_language(T, detokenized))

            if join_paragraphs:
                paragraphs.extend(transduced)
            else:
                paragraphs.append(transduced)
                original.append(detokenized)
            lens += 1
            if lens == n:
                break
    total_len = 0
    for i, para in enumerate(paragraphs):
        total_len += len(para)
        print(
            f"Paragraph {i+1} len {len(para)} cumulative length {total_len}:\n{para}\n"
        )
    return paragraphs, original, total_len


def load_hf_data_paragraphs_bytes(
    T,
    split,
    n=100,
    verbose=True,
    join_paragraphs=False,
    transducer_name="hf_realpha",
    dataset_name="JulesBelveze/tldr_news",
    text_column="content",
    max_length=None,
):
    """
    Load the first n paragraphs from the wikitext dataset.
    """
    dataset_config = None
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    dataset = dataset.select(range(n))
    paragraphs = []
    original = []
    lens = 0

    for item in dataset:
        text = item[text_column].strip()
        if max_length is not None and len(text) > max_length:
            text = text[:max_length]
        if text:
            fsa = T(text, None)
            transduced = next(fsa.language())

            if join_paragraphs:
                paragraphs.extend(transduced)
            else:
                paragraphs.append(transduced)
                original.append(text)
            lens += 1
            if lens == n:
                break
    total_len = 0
    for i, para in enumerate(paragraphs):
        total_len += len(para)
        print(
            f"Paragraph {i+1} len {len(para)} cumulative length {total_len}:\n{para}\n"
        )
    return paragraphs, original, total_len