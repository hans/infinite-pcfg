"""
Learn a nontrivial PCFG parser by expectation-maximization.
"""

import itertools

import numpy as np
from tqdm import tqdm, trange

from infinite_parser import pcfg as P


sentences = [
    "the dog bit the man",
    "the man bit the dog",
    # "the dog bit",
    # "the man bit",
]
sentences = [sent.split() for sent in sentences]
vocabulary = set(itertools.chain.from_iterable(sentences))

# Build PCFG with random weights.
pcfg = P.FixedPCFG("S",
                   terminals=list(vocabulary),
                   nonterminals=["S", "NP", "VP", "N", "V", "D"],
                   productions=[("NP", "VP"), ("V", "NP"), ("D", "N")])

for e in trange(40, desc="Epoch"):
  for sentence in tqdm(sentences):
    pcfg = P.inside_outside_update(pcfg, sentence)

  # Calculate total probability of corpus.
  ll = 0
  for sentence in tqdm(sentences):
    alphas, betas, _ = P.inside_outside(pcfg, sentence)
    total_prob = alphas[pcfg.nonterm2idx[pcfg.start], 0, len(sentence) - 1]
    ll += np.log(total_prob)

  tqdm.write("%i ll: %f" % (e, ll))

for sentence in sentences:
  print(sentence)
  alphas, betas, backtrace = P.inside_outside(pcfg, sentence)
  tree = P.tree_from_backtrace(pcfg, sentence, backtrace)
  tree.pretty_print()
