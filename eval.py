###############################################################################
# Language Modeling on Penn Tree Bank (preprocessed for RNNG)
#
# This file evaluates new sentences and print surprisal (base e) of each word.
#
###############################################################################

import argparse
import numpy as np
import torch
from torch.autograd import Variable
import sys
from scipy.stats import gmean
import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/ptb',
                    help='location of the data corpus')
parser.add_argument('--corpus', type=str, default=None,
                    help='location of cached corpus')
parser.add_argument('--checkpoint', type=str, default='./model_small.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--ppl', default=False, action='store_true',
                    help='toggle whether or not to report sentence-level perplexity')
parser.add_argument('--eval_data', type=str, default='stimuli_items/input_test.raw')
parser.add_argument('--outf', type=argparse.FileType("w", encoding="utf-8"), default=sys.stdout,
                    help='output file for generated text')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    if args.cuda:
        model = torch.load(f).to(device)
    else:
        model = torch.load(f, map_location=lambda storage, loc: storage)
        model.cpu()
model.eval()

if args.corpus:
    corpus = torch.load(args.corpus)
else:
    corpus = data.Corpus(args.data)
    torch.save(corpus, args.data + '/cache.pt')

ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)


# read eval data
with open(args.eval_data, 'r') as f:
    lines = f.readlines()
sents = [line.strip().split() for line in lines]


with args.outf as f:
    # write header.
    f.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")
    with torch.no_grad():  # no tracking history
        # all_ppls = []
        for sent_id, sent in enumerate(sents):
            hidden = model.init_hidden(1)
            input = torch.tensor([[corpus.dictionary.word2idx[sent[0]]]],dtype=torch.long).to(device)

            f.write("%i\t%i\t%s\t%f\n" % (sent_id + 1, 1, sent[0], 0.0))

            probs = []
            for i, w in enumerate(sent[1:]):

                output, hidden = model(input, hidden)
                word_weights = torch.Tensor.numpy(output.squeeze().div(args.temperature).exp().cpu())
                total_weight = np.sum(word_weights)
                word_idx = corpus.dictionary.word2idx[w]
                word_surprisal = -np.log(word_weights[word_idx]/total_weight)
                word_prob = word_weights[word_idx]/total_weight
                probs.append(word_prob)

                f.write("%i\t%i\t%s\t%f\n" % (sent_id + 1, i + 1, w, word_surprisal))
       
                input.fill_(word_idx)
        #     ppl = np.power(np.prod(probs), -1.0/len(sent))
        #     if args.ppl:
        #         print('sentence {} ppl: {}'.format(sent_id, ppl))
        #     all_ppls.append(ppl)
        # if args.ppl:
        #     print('MEAN PPL ACROSS SENTENCES: {}'.format(np.mean(all_ppls)))