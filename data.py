import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def get_mask(self, token_seq, whitelist):
        """
        Get an integer mask sequence for the given 1D token sequence.
        The mask will be of the same length as `token_seq`, and contain a `1` wherever
        the corresponding token in `token_seq` is in `whitelist`.
        """
        whitelist = set(list(whitelist) + ["<eos>"])
        try:
            whitelist_idxs = [self.dictionary.word2idx[word] for word in whitelist]
        except KeyError:
            print("provided whitelist has values not present in model vocab")
            raise

        whitelist_vec = torch.LongTensor(whitelist_idxs)
        return in1d(token_seq, whitelist_vec)


def in1d(source, check):
    """
    Return a boolean mask over `source` where each value is `1` if the corresponding value
    of `source` is in the 1D array `check`.
    """
    return (source[..., None] == check).any(-1)
