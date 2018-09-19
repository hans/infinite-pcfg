"""
Implementation of the infinite PCFG (HDP-PCFG) model from Liang et al. (2007).

PCFG parameters are approximated with mean-field variational inference.
"""

import numpy as np


# class PCFG(object):
#   """
#   Represents a particular parameter setting of a PCFG.
#   """

#   def __init__(self, phi_Z, phi_E, phi_B):
#     """
#     Args:
#       phi_Z: Rule type parameters
#       phi_E: Emission parameters
#       phi_B: Branching parameters
#     """
#     self.phi_Z = phi_Z
#     self.phi_E = phi_E
#     self.phi_B = phi_B


class Tree(object):
  def __init__(self, value, left, right):
    self.value = value
    self.left = left
    self.right = right

  def iter_leaves(self):
    if isinstance(self.left, Tree):
      for leaf in self.left.iter_leaves():
        yield leaf
    else:
      yield self.left

    if isinstance(self.right, Tree):
      for leaf in self.right.iter_leaves():
        yield leaf
    else:
      yield self.right

  def iter_productions(self):
    if isinstance(self.left, Tree):
      for production in self.left.iter_productions():
        yield production
      left_val = self.left.value
    else:
      left_val = self.left

    right_val = self.right.value if isinstance(self.right, Tree) else self.right

    yield (self.value, left_val, right_val)

    if isinstance(self.right, Tree):
      for production in self.right.iter_productions():
        yield production


class FixedPCFG(object):
  def __init__(self, start, terminals, nonterminals, productions, binary_weights, unary_weights):
    self.start = start
    self.terminals = terminals
    self.nonterminals = nonterminals
    self.productions = productions

    self.term2idx = {symbol: idx for idx, symbol in enumerate(self.terminals)}
    self.nonterm2idx = {symbol: idx for idx, symbol in enumerate(self.nonterminals)}
    self.production2idx = {production: idx for idx, production in enumerate(self.productions)}

    self.binary_weights = binary_weights
    self.unary_weights = unary_weights

    assert self.start in self.nonterminals
    assert self.binary_weights.shape == (len(self.nonterminals), len(self.productions))
    assert self.unary_weights.shape == (len(self.nonterminals), len(self.terminals))

  def score_tree(self, tree):
    log_score = 0
    for symbol, left, right in tree.iter_productions():
      print(symbol, left, right)
      if right is None:
        prod_score = self.unary_weights[self.nonterm2idx[symbol], self.term2idx[left]]
      else:
        prod_score = self.binary_weights[self.nonterm2idx[symbol], self.production2idx[(left, right)]]
      print("\t", prod_score)
      log_score += np.log(prod_score)

    return log_score


def inside_outside(pcfg, sentence):
  # INSIDE
  # alpha[i, j, k] = inside score for nonterminal i with span [j, k]
  alpha = np.zeros((len(pcfg.nonterminals), len(sentence), len(sentence)))
  # base case: unary rewrites
  for i, nonterm in enumerate(pcfg.nonterminals):
    for j, word in enumerate(sentence):
      alpha[i, j, j] = pcfg.unary_weights[i, pcfg.term2idx[word]]

  # recursive case
  for span in range(2, len(sentence) + 1): # range(2, 3)
    for j in range(0, len(sentence) - span + 1): # range(0, 3 - span)
      # End of nonterminal span (up to and including the word at this index)
      k = j + span - 1
      # just one case: start = 0, span = 2, end = start + span - 1
      # where `end` denotes an end up to and including the word at index `end`
      for i, nonterm in enumerate(pcfg.nonterminals):
        score = 0
        for split in range(1, span): # range(1, 2)
          # ==> split = 1
          for prod_idx, (left, right) in enumerate(pcfg.productions):
            local_score = (
                # Production score
                np.log(pcfg.binary_weights[pcfg.nonterm2idx[nonterm], prod_idx]) +
                # Left child score
                np.log(alpha[pcfg.nonterm2idx[left], j, j + split - 1]) +
                # Right child score
                np.log(alpha[pcfg.nonterm2idx[right], j + split, k]))

            score += np.exp(local_score)

        alpha[i, j, k] = score

  # OUTSIDE
  # beta[i, j, k] = outside score for nonterminal i with span [j, k]
  beta = np.zeros((len(pcfg.nonterminals), len(sentence), len(sentence)))
  # base case
  beta[pcfg.nonterm2idx[pcfg.start], 0, len(sentence) - 1] = 1.0
  for i, nonterm in enumerate(pcfg.nonterminals):
    for j in range(0, len(sentence)):
      for k in range(j, len(sentence)):
        if i == 0 and k == len(sentence) - 1:
          # Do not recompute base case.
          continue

        left_score, right_score = 0, 0

        # First option: nonterm `i` appears with a sibling to the left
        for left_start in range(0, j):
          for par_idx, left_parent in enumerate(pcfg.nonterminals):
            for prod_idx, (left, right) in enumerate(pcfg.productions):
              if right != nonterm:
                continue

              local_score = (
                  # Production score
                  np.log(pcfg.binary_weights[par_idx, prod_idx]) +
                  # Left inner score
                  np.log(alpha[pcfg.nonterm2idx[left], left_start, j - 1]) +
                  # Outer score
                  np.log(beta[par_idx, left_start, k]))

              left_score += np.exp(local_score)

        # Second option: nonterm `i` appears with a sibling to the right
        for right_end in range(k + 1, len(sentence)):
          for par_idx, right_parent in enumerate(pcfg.nonterminals):
            for prod_idx, (left, right) in enumerate(pcfg.productions):
              if left != nonterm:
                continue
              elif left == right:
                # Don't double-count case where siblings are identical.
                continue

            local_score = (
                # Production score
                np.log(pcfg.binary_weights[par_idx, prod_idx]) +
                # Outer score
                np.log(beta[par_idx, j, right_end]) +
                # Right inner score
                np.log(alpha[pcfg.nonterm2idx[right], k + 1, right_end]))

            right_score += np.exp(local_score)

        beta[i, j, k] = left_score + right_score

  return alpha, beta


if __name__ == '__main__':
  t = Tree("x", Tree("a", Tree("b", "c", None), Tree("b", "d", None)), Tree("e", "f", None))
  for x in t.iter_productions():
    print(x)

  pcfg = FixedPCFG("x",
                   ["c", "d", "f"],
                   ["x", "a", "b", "e"],
                   [("a", "e"), ("b", "b")],
                   np.array([[1, 0],
                             [0, 1],
                             [0, 0],
                             [0, 0]]),
                   np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0.5, 0.5, 0],
                             [0, 0, 1]]))
  print(pcfg.score_tree(t))



from nose.tools import assert_equal

def test_inside_outside():
  pcfg = FixedPCFG("x",
                   ["c", "d"],
                   ["x", "b"],
                   [("b", "b")],
                   np.array([[0.5],
                             [0]]),
                   np.array([[0, 0],
                             [0.5, 0.5]]))

  alphas, betas = inside_outside(pcfg, "c d".split())

  from pprint import pprint
  pprint(list(zip(pcfg.nonterminals, alphas)))
  pprint(list(zip(pcfg.nonterminals, betas)))

  # check alpha[x]
  np.testing.assert_allclose(alphas[0], [[0, 0.125],
                                         [0, 0]])
  # check alpha[b] (preterminals)
  np.testing.assert_allclose(alphas[1], [[0.5, 0],
                                         [0, 0.5]])
