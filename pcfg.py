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


class FixedPCFG(object):
  def __init__(self, start, terminals, nonterminals, productions,
               binary_weights=None, unary_weights=None):
    self.start = start
    self.terminals = terminals
    self.nonterminals = nonterminals
    self.productions = productions

    self.term2idx = {symbol: idx for idx, symbol in enumerate(self.terminals)}
    self.nonterm2idx = {symbol: idx for idx, symbol in enumerate(self.nonterminals)}
    self.production2idx = {production: idx for idx, production in enumerate(self.productions)}

    if binary_weights is None:
      binary_weights = np.random.uniform(size=(len(self.nonterminals), len(self.productions)))
    if unary_weights is None:
      unary_weights = np.random.uniform(size=(len(self.nonterminals), len(self.terminals)))

    # Normalize weights.
    total_mass = binary_weights.sum() + unary_weights.sum()
    binary_weights /= total_mass
    unary_weights /= total_mass

    # Binary rule weights, of dimension `len(nonterminals) * len(productions)`
    self.binary_weights = binary_weights
    # Unary rule weights, of dimension `len(nonterminals) * len(terminals)`
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
  """
  Infer PCFG parses for `sentence` by the inside-outside algorithm.

  Returns:
    alpha: inside probabilities
    beta: outside probabilities:
    backtrace:
  """
  # INSIDE
  # alpha[i, j, k] = inside score for nonterminal i with span [j, k]
  alpha = np.zeros((len(pcfg.nonterminals), len(sentence), len(sentence)))
  backtrace = np.zeros((len(pcfg.nonterminals), len(sentence), len(sentence), 3), dtype=int)
  # base case: unary rewrites
  for i, nonterm in enumerate(pcfg.nonterminals):
    for j, word in enumerate(sentence):
      alpha[i, j, j] = pcfg.unary_weights[i, pcfg.term2idx[word]]

  # recursive case
  for span in range(2, len(sentence) + 1):
    for j in range(0, len(sentence) - span + 1):
      # End of nonterminal span (up to and including the word at this index)
      k = j + span - 1
      # where `end` denotes an end up to and including the word at index `end`
      for i, nonterm in enumerate(pcfg.nonterminals):
        score = 0

        # Keep backtrace for maximally scoring element
        best_backtrace, best_backtrace_score = None, 0

        for split in range(1, span):
          # ==> split = 1
          for prod_idx, (left, right) in enumerate(pcfg.productions):
            local_score = np.exp(
                # Production score
                np.log(pcfg.binary_weights[pcfg.nonterm2idx[nonterm], prod_idx]) +
                # Left child score
                np.log(alpha[pcfg.nonterm2idx[left], j, j + split - 1]) +
                # Right child score
                np.log(alpha[pcfg.nonterm2idx[right], j + split, k]))

            score += local_score

            if local_score > best_backtrace_score:
              best_backtrace = (pcfg.nonterm2idx[left], pcfg.nonterm2idx[right],
                                split)
              best_backtrace_score = local_score

        alpha[i, j, k] = score
        if best_backtrace is not None:
          backtrace[i, j, k] = best_backtrace

  # OUTSIDE
  # beta[i, j, k] = outside score for nonterminal i with span [j, k]
  beta = np.zeros((len(pcfg.nonterminals), len(sentence), len(sentence)))
  # base case
  beta[pcfg.nonterm2idx[pcfg.start], 0, len(sentence) - 1] = 1.0
  for i, nonterm in enumerate(pcfg.nonterminals):
    for j in range(0, len(sentence)):
      for k in range(j, len(sentence)):
        if j == 0 and k == len(sentence) - 1:
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

  return alpha, beta, backtrace


def inside_outside_update(pcfg, sentence):
  """
  Perform an inside-outside EM parameter update with the given sentence input.
  """

  # Run inside-outside inference.
  alpha, beta, _ = inside_outside(pcfg, sentence)

  # Calculate expected counts.
  binary_counts = np.zeros((len(pcfg.nonterminals), len(pcfg.productions)))
  unary_counts = np.zeros((len(pcfg.nonterminals), len(pcfg.terminals)))

  # Calculate binary counts
  for span in range(2, len(sentence) + 1):
    for j in range(0, len(sentence) - span + 1):
      # End of nonterminal span (up to and including the word at this index)
      k = j + span - 1

      for i, nonterm in enumerate(pcfg.nonterminals):
        for split in range(1, span):
          for prod_idx, production in enumerate(pcfg.productions):
            left_idx = pcfg.nonterm2idx[production[0]]
            right_idx = pcfg.nonterm2idx[production[1]]

            # mu(i -> l r, j, k): marginal probability of observing node i -> l
            # r at span [j, k]
            mu = np.exp(
                # outside probability of parent
                np.log(beta[i, j, k]) +
                # binary production weight
                np.log(pcfg.binary_weights[i, prod_idx]) +
                # inside probability of left child
                np.log(alpha[left_idx, j, j + split - 1]) +
                # inside probability of right child
                np.log(alpha[right_idx, j + split, k]))
            binary_counts[i, prod_idx] += mu

  # Calculate unary counts
  for j, word in enumerate(sentence):
    for i, nonterm in enumerate(pcfg.nonterminals):
      term_idx = pcfg.term2idx[word]
      unary_counts[i, term_idx] += np.exp(
          # outside probability of parent
          np.log(beta[i, j, j]) +
          # unary production weight
          np.log(pcfg.unary_weights[i, term_idx]))

  # Normalize counts by total probability mass assigned to tree.
  total_potential = alpha[pcfg.nonterm2idx[pcfg.start], 0, len(sentence) - 1]
  binary_counts /= total_potential
  unary_counts /= total_potential

  # Perform weight update.
  new_binary_weights = pcfg.binary_weights + binary_counts
  new_unary_weights = pcfg.unary_weights + unary_counts
  # Normalize per parent nonterminal.
  Z = new_binary_weights.sum(axis=1, keepdims=True) + \
      new_unary_weights.sum(axis=1, keepdims=True)
  new_binary_weights /= Z
  new_unary_weights /= Z

  ret = deepcopy(pcfg)
  ret.binary_weights = new_binary_weights
  ret.unary_weights = new_unary_weights
  return ret


def tree_from_backtrace(pcfg, sentence, backtrace):
  def inner(i, j, k):
    if j == k:
      return Tree(pcfg.nonterminals[i], [sentence[j]])

    left, right, split = backtrace[i, j, k]
    left = inner(left, j, j + split - 1)
    right = inner(right, j + split, k)
    return Tree(pcfg.nonterminals[i], [left, right])

  return inner(pcfg.nonterm2idx[pcfg.start], 0, backtrace.shape[1] - 1)


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

  sentence = "c d".split()
  alphas, betas, backtrace = inside_outside(pcfg, sentence)

  from pprint import pprint
  pprint(list(zip(pcfg.nonterminals, alphas)))
  pprint(list(zip(pcfg.nonterminals, betas)))

  tree_from_backtrace(pcfg, sentence, backtrace).pretty_print()
  assert_equal(tree_from_backtrace(pcfg, sentence, backtrace),
               Tree.fromstring("(x (b c) (b d))"))

  # check alpha[x]
  np.testing.assert_allclose(alphas[0], [[0, 0.03703],
                                         [0, 0]], atol=1e-5)
  # check alpha[b] (preterminals)
  np.testing.assert_allclose(alphas[1], [[1/3., 0],
                                         [0, 1/3.]])


def test_inside_outside2():
  pcfg = FixedPCFG("x",
                   ["c", "d"],
                   ["x", "b"],
                   [("b", "b"), ("b", "x")],
                   np.array([[0.25, 0.75],
                             [0, 0]]),
                   np.array([[0, 0],
                             [0.5, 0.5]]))

  sentence = "c d d".split()
  alphas, betas, backtrace = inside_outside(pcfg, sentence)

  from pprint import pprint
  pprint(list(zip(pcfg.nonterminals, alphas)))
  pprint(list(zip(pcfg.nonterminals, betas)))

  tree_from_backtrace(pcfg, sentence, backtrace).pretty_print()
  assert_equal(tree_from_backtrace(pcfg, sentence, backtrace),
               Tree.fromstring("(x (b c) (x (b d) (b d)))"))

  # check alpha[x]
  np.testing.assert_allclose(alphas[0], [[0, 0.0078125, 0.00073242],
                                         [0, 0, 0.0078125],
                                         [0, 0, 0]], atol=1e-5)
  # check alpha[b] (preterminals)
  np.testing.assert_allclose(alphas[1], [[1/4., 0, 0],
                                         [0, 1/4., 0],
                                         [0, 0, 1/4.]])


def test_inside_outside_update():
  pcfg = FixedPCFG("x",
                   ["c", "d"],
                   ["x", "b"],
                   [("b", "b"), ("b", "x")])

  sentence = "c d d".split()

  prev_total_prob = 0

  for i in range(20):
    alphas, betas, backtrace = inside_outside(pcfg, sentence)
    total_prob = alphas[pcfg.nonterm2idx[pcfg.start], 0, len(sentence) - 1]
    print("%d\t%f" % (i, total_prob))

    # NB include small tolerance due to float imprecision
    assert total_prob - prev_total_prob >= -1e4, \
        "Total prob should never decrease: %f -> %f (iter %d)" % \
        (prev_total_prob, total_prob, i)
    prev_total_prob = total_prob

    pcfg = inside_outside_update(pcfg, sentence)
