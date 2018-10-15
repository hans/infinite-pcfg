"""
Currently:
  Inference and learning via inside-outside + EM for PCFGs.

Soon:

Implementation of the infinite PCFG (HDP-PCFG) model from Liang et al. (2007).

PCFG parameters are approximated with mean-field variational inference.
"""

from copy import deepcopy

import numpy as np

from nltk import Tree


def _index(xs):
  return {x: idx for idx, x in enumerate(xs)}


class FixedPCFG(object):
  def __init__(self, start, terminals, nonterminals, preterminals,
               productions, binary_weights=None, unary_weights=None):
    self.start = start
    self.terminals = terminals
    self.nonterminals = nonterminals
    self.preterminals = preterminals
    self.productions = productions

    assert not set(self.preterminals) & set(self.nonterminals)

    self.term2idx = _index(self.terminals)
    self.nonterm2idx = _index(self.nonterminals)
    self.preterm2idx = _index(self.preterminals)
    self.production2idx = _index(self.productions)

    if binary_weights is None:
      binary_weights = np.random.uniform(size=(len(self.nonterminals), len(self.productions)))
    if unary_weights is None:
      unary_weights = np.random.uniform(size=(len(self.preterminals), len(self.terminals)))

    # Normalize weights.
    binary_weights /= binary_weights.sum(axis=1, keepdims=True)
    unary_weights /= unary_weights.sum(axis=1, keepdims=True)

    # Binary rule weights, of dimension `len(nonterminals) * len(productions)`
    self.binary_weights = binary_weights
    # Unary rule weights, of dimension `len(preterminals) * len(terminals)`
    self.unary_weights = unary_weights

    assert self.start in self.nonterminals
    assert self.binary_weights.shape == (len(self.nonterminals), len(self.productions))
    assert self.unary_weights.shape == (len(self.preterminals), len(self.terminals))

  def score_tree(self, tree):
    log_score = 0
    for symbol, left, right in tree.iter_productions():
      if right is None:
        prod_score = self.unary_weights[self.preterm2idx[symbol], self.term2idx[left]]
      else:
        prod_score = self.binary_weights[self.nonterm2idx[symbol], self.production2idx[(left, right)]]
      log_score += np.log(prod_score)

    return log_score


def tree_from_backtrace(pcfg, sentence, backtrace):
  def inner(i, j, k):
    if j == k:
      if i < len(pcfg.nonterminals):
        raise ValueError("Nonterminal %s used as preterminal in parse", pcfg.nonterminals[i])

      i -= len(pcfg.nonterminals)
      return Tree(pcfg.preterminals[i], [sentence[j]])

    left, right, split = backtrace[i, j, k]
    left = inner(left, j, j + split - 1)
    right = inner(right, j + split, k)
    return Tree(pcfg.nonterminals[i], [left, right])

  return inner(pcfg.nonterm2idx[pcfg.start], 0, backtrace.shape[1] - 1)
