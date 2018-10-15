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
          # ==> split = 1
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

        # Second option: nonterm `i` appears with a sibling to the right
        for right_end in range(k + 1, len(sentence)):
          for par_idx, right_parent in enumerate(pcfg.nonterminals):
            for prod_idx, (left, right) in enumerate(pcfg.productions):
              if left != nonterm:
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


def update_em(pcfg, sentence):
  """
  Perform an inside-outside EM parameter update with the given sentence input.

  Returns:
    new_pcfg: Copy of `pcfg` with updated weights.
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
                else pcfg.preterm2idx[left]
            right_idx = pcfg.nonterm2idx[right] if right in pcfg.nonterm2idx \
                else pcfg.preterm2idx[right]

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

  # Weight counts by total probability mass assigned to tree.
  total_potential = alpha[pcfg.nonterm2idx[pcfg.start], 0, len(sentence) - 1]
  binary_counts /= total_potential
  unary_counts /= total_potential

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

