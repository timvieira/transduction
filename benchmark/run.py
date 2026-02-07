"""
Simple benchmarking script using realpha FST.
"""
from benchmark.fsts.realpha import build_realpha
from benchmark.data import load_wikitext_paragraphs_bytes
from transduction.enumeration import prioritized_enumeration
from transduction.lm import StateLM

import tqdm

import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


def main():
    model_name = "gpt2"
    fst = build_realpha(model_name)
    lm = StateLM.initial(model_name)
    tokenizer = lm.lm.tokenizer

    n_pgs = 1
    pgs, _, total_len = load_wikitext_paragraphs_bytes(
        fst, "test", tokenizer, n=n_pgs
    )
    print(f"Loaded {total_len}")

    result = {}

    for idx, pg in enumerate(pgs):
        result[idx] = []
        for i in tqdm.tqdm(range(1, len(pg), 1)):
            precover = prioritized_enumeration(lm, fst, pg[:i], max_steps=4)
            print(precover)

    # todo eval


if __name__ == "__main__":
    main()
