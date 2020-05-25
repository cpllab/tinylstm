#!/usr/bin/env python
"""
Helper script to save a `Corpus` object for easy future loading.
"""

from argparse import ArgumentParser
from pathlib import Path

import torch

from data import Corpus



p = ArgumentParser()
p.add_argument("corpus_path")
p.add_argument("out_file")


def main(args):
    if not Path(args.out_file).parent.exists():
        raise ValueError("Invalid out_file %s. Does the directory exist?" % (args.out_file,))

    corpus = Corpus(args.corpus_path)

    # Remove actual dataset -- just keep the vocabulary
    corpus.train = None
    corpus.valid = None
    corpus.test = None

    torch.save(corpus, args.out_file)


if __name__ == "__main__":
    main(p.parse_args())
