from transduction.benchmarking.fsts.realpha import build_realpha
from transduction.benchmarking.data import load_wikitext_paragraphs_bytes
from transduction.benchmarking.legacy.hf_model import HFModel
from transduction.benchmarking.legacy.algorithm import prioritized_enumeration

import tqdm

import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


def main():
    model_name = "gpt2"
    fst = build_realpha(model_name)
    lm = HFModel(model_name)
    tokenizer = lm.tokenizer

    n_pgs = 1
    pgs, _, total_len = load_wikitext_paragraphs_bytes(
        fst, "test", tokenizer, n=n_pgs 
    )
    print(f"Loaded {total_len}")
    # todo pruning

    result = {}

    for idx, pg in enumerate(pgs):
        result[idx] = []
        for i in tqdm.tqdm(range(1, len(pg), 1)):
            
            precover = prioritized_enumeration(lm, fst, pg[:i], 4, tokenizer.eos_token)

            print(precover)

    # todo eval
            


if __name__ == "__main__":
    # TODO: argparse everything

    main()