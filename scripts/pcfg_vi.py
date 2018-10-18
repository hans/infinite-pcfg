"""
Learn a nontrivial PCFG parser by expectation-maximization.
"""

import itertools

import numpy as np
from scipy.special import digamma
from tqdm import tqdm, trange

from infinite_parser.inference import inside_outside as I
from infinite_parser import pcfg as P


sentences = [
    "the dog bit the man",
    "the man bit the dog",
    "the man gave the dog the treat",
    "the dog gave the man the newspaper",
    "the dog bit",
    "the man bit",
    "the man sent the dog the newspaper",
    "the man sent the dog",
]
sentences = [sent.split() for sent in sentences]
vocabulary = set(itertools.chain.from_iterable(sentences))

# Build PCFG with random weights.
pcfg = P.FixedPCFG("S",
                   terminals=list(vocabulary),
                   nonterminals=["S", "NP", "VP", "VP$"],
                   preterminals=["N", "V", "D"],
                   productions=[("NP", "VP"), ("V", "NP"), ("V", "VP$"), ("NP", "NP"), ("D", "N")])

unary_prior = np.ones_like(pcfg.unary_weights)
binary_prior = np.ones_like(pcfg.binary_weights)

prev_ll = -np.inf
for e in trange(40, desc="Epoch"):
  for sentence in tqdm(sentences):
    unary_prior, binary_prior = I.update_mean_field(
        pcfg, sentence, unary_prior, binary_prior)

  # Calculate total probability of corpus.
  ll = 0
  pcfg.unary_weights = np.exp(digamma(unary_prior))
  pcfg.binary_weights = np.exp(digamma(binary_prior))
  pcfg.unary_weights /= np.exp(digamma(unary_prior.sum(axis=1, keepdims=True)))
  pcfg.binary_weights /= np.exp(digamma(binary_prior.sum(axis=1, keepdims=True)))

  for sentence in tqdm(sentences):
    alphas, betas, _ = I.parse(pcfg, sentence)
    total_prob = alphas[pcfg.nonterm2idx[pcfg.start], 0, len(sentence) - 1]
    ll += np.log(total_prob)

  tqdm.write("%i ll: %f" % (e, ll))

  if ll - prev_ll > 0 and ll - prev_ll <= 1e-3:
    break
  prev_ll = ll

for sentence in sentences:
  print(" ".join(sentence))
  alphas, betas, backtrace = I.parse(pcfg, sentence)
  tree = P.tree_from_backtrace(pcfg, sentence, backtrace)
  tree.pretty_print()
