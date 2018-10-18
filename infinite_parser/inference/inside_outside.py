"""
Implements algorithms for inference and learning of PCFG by the inside-outside
algorithm.

Inference algorithms:

  - exact inside-outside

Update algorithms:

  - EM inside-outside
  - mean-field inside-outside (coming soon)
"""

from copy import deepcopy

import numpy as np
from scipy.special import digamma


def parse(pcfg, sentence):
  """
  Infer PCFG parses for `sentence` by the inside-outside algorithm.

  Returns:
    alpha: 3-dimensional array of inside probabilities. `alpha[i, j, k]`
      denotes the inside probability of nonterminal `i` (or preterminal `i -
      len(nonterminals)`) yielding the span `[j, k]` (inclusive on both ends).
      Thus shape is `(len(nonterminals) + len(preterminals), len(sentence),
      len(sentence))`.
    beta: 3-dimensional array of outside probabilities; same design as `alpha`.
    backtrace: Chart backtrace; can be used to reconstruct a maximal-scoring
      tree. `backtrace[i, j, k]` is a 3-tuple `(left, right, split)` describing
      the optimal left-nonterminal, right-nonterminal, and split point for a
      nonterminal `i` spanning `[j, k]` (inclusive both ends). Here the left
      child spans `[j, j + split - 1]` and the right child spans `[j + split,
      k]`.
  """
  # INSIDE
  # alpha[i, j, k] = inside score for nonterminal i or preterminal
  # `i - len(nonterminals)` with span [j, k]
  alpha = np.zeros((len(pcfg.nonterminals) + len(pcfg.preterminals),
                    len(sentence), len(sentence)))
  backtrace = np.zeros((len(pcfg.nonterminals) + len(pcfg.preterminals),
                        len(sentence), len(sentence), 3), dtype=int)
  # base case: unary rewrites
  for i, preterm in enumerate(pcfg.preterminals):
    for j, word in enumerate(sentence):
      # preterminals are indexed after nonterminals in alpha array.
      idx = len(pcfg.nonterminals) + i
      alpha[idx, j, j] = pcfg.unary_weights[i, pcfg.term2idx[word]]

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
          for prod_idx, (left, right) in enumerate(pcfg.productions):
            # Prepare index lookups for left/right children.
            left_idx = len(pcfg.nonterminals) + pcfg.preterm2idx[left] \
                if left in pcfg.preterm2idx else pcfg.nonterm2idx[left]
            right_idx = len(pcfg.nonterminals) + pcfg.preterm2idx[right] \
                if right in pcfg.preterm2idx else pcfg.nonterm2idx[right]

            # Calculate inside probabilities of left and right children.
            left_score = alpha[left_idx, j, j + split - 1]
            right_score = alpha[right_idx, j + split, k]

            local_score = np.exp(
                # Production score
                np.log(pcfg.binary_weights[pcfg.nonterm2idx[nonterm], prod_idx]) +
                # Left child score
                np.log(left_score) +
                # Right child score
                np.log(right_score))

            score += local_score

            if local_score > best_backtrace_score:
              best_backtrace = (left_idx, right_idx, split)
              best_backtrace_score = local_score

        alpha[i, j, k] = score
        if best_backtrace is not None:
          backtrace[i, j, k] = best_backtrace

  # OUTSIDE
  # beta[i, j, k] = outside score for nonterminal i with span [j, k]
  beta = np.zeros((len(pcfg.nonterminals) + len(pcfg.preterminals),
                   len(sentence), len(sentence)))
  # base case
  beta[pcfg.nonterm2idx[pcfg.start], 0, len(sentence) - 1] = 1.0
  # recursive case
  for i, node in enumerate(pcfg.nonterminals + pcfg.preterminals):
    for j in range(0, len(sentence)):
      for k in range(j, len(sentence)):
        if j == 0 and k == len(sentence) - 1:
          # Do not recompute base case.
          continue
        elif i > len(pcfg.nonterminals) and j != k:
          # Preterminals can only apply when j == k. Skip.
          continue

        left_score, right_score = 0, 0

        # First option: node `i` appears with a sibling to the left
        for left_start in range(0, j):
          for par_idx, left_parent in enumerate(pcfg.nonterminals):
            for prod_idx, (left, right) in enumerate(pcfg.productions):
              if right != node:
                continue

              left_idx = len(pcfg.nonterminals) + pcfg.preterm2idx[left] \
                  if left in pcfg.preterm2idx else pcfg.nonterm2idx[left]

              local_score = (
                  # Production score
                  np.log(pcfg.binary_weights[par_idx, prod_idx]) +
                  # Left inner score
                  np.log(alpha[left_idx, left_start, j - 1]) +
                  # Outer score
                  np.log(beta[par_idx, left_start, k]))

              left_score += np.exp(local_score)

        # Second option: node `i` appears with a sibling to the right
        for right_end in range(k + 1, len(sentence)):
          for par_idx, right_parent in enumerate(pcfg.nonterminals):
            for prod_idx, (left, right) in enumerate(pcfg.productions):
              if left != node:
                continue
              elif left == right:
                # Don't double-count case where siblings are identical.
                continue

              right_idx = len(pcfg.nonterminals) + pcfg.preterm2idx[right] \
                if right in pcfg.preterm2idx else pcfg.nonterm2idx[right]

              local_score = (
                  # Production score
                  np.log(pcfg.binary_weights[par_idx, prod_idx]) +
                  # Outer score
                  np.log(beta[par_idx, j, right_end]) +
                  # Right inner score
                  np.log(alpha[right_idx, k + 1, right_end]))

              right_score += np.exp(local_score)

        beta[i, j, k] = left_score + right_score

  return alpha, beta, backtrace


def expected_counts(pcfg, sentence):
  """
  Estimate marginal distributions over grammar weights, marginalizing out all
  possible parses of `sentence` under `pcfg` via inside-outside.
  """
  # Run inside-outside inference.
  alpha, beta, _ = parse(pcfg, sentence)

  # Calculate expected counts.
  binary_counts = np.zeros((len(pcfg.nonterminals), len(pcfg.productions)))
  unary_counts = np.zeros((len(pcfg.preterminals), len(pcfg.terminals)))

  # Calculate binary counts
  for span in range(2, len(sentence) + 1):
    for j in range(0, len(sentence) - span + 1):
      # End of nonterminal span (up to and including the word at this index)
      k = j + span - 1

      for i, nonterm in enumerate(pcfg.nonterminals):
        for split in range(1, span):
          for prod_idx, (left, right) in enumerate(pcfg.productions):
            left_idx = pcfg.nonterm2idx[left] if left in pcfg.nonterm2idx \
                else len(pcfg.nonterminals) + pcfg.preterm2idx[left]
            right_idx = pcfg.nonterm2idx[right] if right in pcfg.nonterm2idx \
                else len(pcfg.nonterminals) + pcfg.preterm2idx[right]

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
    for i, preterm in enumerate(pcfg.preterminals):
      term_idx = pcfg.term2idx[word]
      unary_counts[i, term_idx] += np.exp(
          # outside probability of parent
          np.log(beta[len(pcfg.nonterminals) + i, j, j]) +
          # unary production weight
          np.log(pcfg.unary_weights[i, term_idx]))

  # Weight counts by total probability mass assigned to tree, marginalizing
  # over parses.
  Z = alpha[pcfg.nonterm2idx[pcfg.start], 0, len(sentence) - 1]
  binary_counts /= Z
  unary_counts /= Z

  return unary_counts, binary_counts, Z


def update_em(pcfg, sentence):
  """
  Perform an inside-outside EM parameter update with the given sentence input.

  Returns:
    new_pcfg: Copy of `pcfg` with updated weights.
  """
  unary_counts, binary_counts, _ = expected_counts(pcfg, sentence)

  # Perform weight update.
  new_binary_weights = pcfg.binary_weights + binary_counts
  new_unary_weights = pcfg.unary_weights + unary_counts
  # Normalize per parent nonterminal. NB that nonterminals and preterminals are
  # exclusive.
  new_binary_weights /= new_binary_weights.sum(axis=1, keepdims=True)
  new_unary_weights /= new_unary_weights.sum(axis=1, keepdims=True)

  ret = deepcopy(pcfg)
  ret.binary_weights = new_binary_weights
  ret.unary_weights = new_unary_weights
  return ret


def update_mean_field(pcfg, sentence, unary_prior=None, binary_prior=None):
  """
  Infer parser parameters $\\theta$ and a tree $z$ for the sentence which
  maximizes the mean-field variational approximation

  $$q(\\theta) q(z)$$

  to the exact posterior $p(\\theta, z \\mid x)$, where $q(\\theta) =
  q(\\phi^E) q(\\phi^B)$ are each distributed Dirichlet with prior parameters
  $\\alpha^E, \\alpha^B$. This inference is performed
  approximately, via a single step of coordinate ascent on the objective above.

  Args:
    unary_prior: Dirichlet prior parameters over unary rewrites.
    binary_prior: Dirichlet prior parameters over binary rewrites.
  """
  # TODO resolve awkward non-use of pcfg here, except for structural params
  # (e.g. shape of parameters)

  # Prior over rewrite parameters phi^E, phi^B.
  unary_shape = pcfg.unary_weights.shape
  binary_shape = pcfg.binary_weights.shape
  if unary_prior is None:
    unary_prior = np.ones(unary_shape)
  if binary_prior is None:
    binary_prior = np.ones(binary_shape)
  assert unary_prior.shape == unary_shape
  assert binary_prior.shape == binary_shape

  pcfg_ = deepcopy(pcfg)

  # Prepare mean-field estimates of rewrite weights (eqs. 6--8).
  unary_weights = np.exp(digamma(unary_prior))
  binary_weights = np.exp(digamma(binary_prior))
  unary_weights /= np.exp(digamma(unary_prior.sum(axis=1, keepdims=True)))
  binary_weights /= np.exp(digamma(binary_prior.sum(axis=1, keepdims=True)))

  pcfg_.unary_weights = unary_weights
  pcfg_.binary_weights = binary_weights

  # Mean-field coordinate update for q(z), computed using mean-field estimates
  # over rewrite weights.
  # (In other words: compute a posterior over parse trees q(z), via
  # inside-outside.)
  alphas, betas, backtrace = parse(pcfg_, sentence)

  # Mean-field coordinate update for q(phi), computed using mean-field estimate
  # over parse trees (eqns. 9--11).
  unary_counts, binary_counts, Z = expected_counts(pcfg_, sentence)
  print(unary_counts)
  print(Z)
  # TODO unary_counts, binary_counts are zero matrices!

  # Compute conjugate posterior over rewrite weights.
  unary_prior += unary_counts
  binary_prior += binary_counts

  return unary_prior, binary_prior
