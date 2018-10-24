"""
Pitman-Yor adaptor grammar inference by MCMC.
"""

from collections import defaultdict, Counter
from copy import copy
import itertools

import numpy as np

from infinite_parser.inference.inside_outside import InsideOutsideParser
from infinite_parser.pcfg import FixedPCFG


def _index(xs):
  return {x: idx for idx, x in enumerate(xs)}


ADAPTED_PRETERMINAL = "#AD"


class InsideOutsideProjectedParser(InsideOutsideParser):
  """
  Inside-outside parser modified to work with PCFGs that are projections of an
  adaptor grammar.

  Projected PCFGs have a special logic for assigning probabilities to adapted
  rewrites.
  """

  def __init__(self, pcfg):
    self.pcfg = pcfg

  def _inside(self, sentence):
    pcfg = self.pcfg

    alpha = np.zeros((len(pcfg.nonterminals) + len(pcfg.preterminals),
                      len(sentence), len(sentence)))
    backtrace = np.zeros((len(pcfg.nonterminals) + len(pcfg.preterminals),
                          len(sentence), len(sentence), 3), dtype=int)
    # base case: unary rewrites
    for i, preterm in enumerate(pcfg.preterminals):
      for j, word in enumerate(sentence):
        weight = pcfg.unary_weights[i, pcfg.term2idx[word]]

        # preterminals are indexed after nonterminals in alpha array.
        idx = len(pcfg.nonterminals) + i
        alpha[idx, j, j] = weight

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

    return alpha, backtrace


class AdaptorGrammar(object):

  def __init__(self, start, terminals, nonterminals, preterminals, productions,
               adapted_nonterminals=None, adaptor_parameters=None,
               binary_weights=None, unary_weights=None):
    adapted_nonterminals = adapted_nonterminals or []
    assert set(adapted_nonterminals).issubset(set(nonterminals))

    self.start = start
    self.terminals = terminals
    self.nonterminals = nonterminals
    self.preterminals = preterminals
    self.adapted_nonterminals = adapted_nonterminals
    self.productions = productions

    self.term2idx = _index(self.terminals)
    self.nonterm2idx = _index(self.nonterminals)
    self.preterm2idx = _index(self.preterminals)
    self.production2idx = _index(self.productions)

    if adaptor_parameters is None:
      adaptor_parameters = [(0, 1) for _ in self.adapted_nonterminals]
    assert len(adaptor_parameters) == len(self.adapted_nonterminals)

    self.adaptor_parameters = adaptor_parameters
    self.adapted_productions = {nonterm: Counter() for nonterm in self.adapted_nonterminals}

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

  def project_pcfg(self):
    """
    Project this grammar into an approximating PCFG. The PCFG can then be used
    to e.g. calculate top-k parses and make update proposals for this grammar.
    """

    # The projected PCFG will merge the rewrite rules of the base grammar with
    # direct rewrites of adapted nonterminals to their yields.
    #
    # Because parsing assumse CNF, we first have to prepare to produc
    # CNF-friendly rewrites of adapted nonterminals to their yields.

    # Create a special preterminal for adapted subtrees.
    preterminals = copy(self.preterminals)
    assert ADAPTED_PRETERMINAL not in preterminals
    preterminals.append(ADAPTED_PRETERMINAL)

    # An adapted nonterminal rewritten to `k` children will require `k - 2`
    # subsymbols in the CNF. We'll use this character to denote those
    # subsymbols.
    cnf_rewrite_char = "#"

    words_in_adapted_yields = defaultdict(set)
    nonterminals = copy(self.nonterminals)
    productions = copy(self.productions)
    nonterminal_adapted_weights = {}
    for adapted_nonterm, adapted_productions in self.adapted_productions.items():
      max_production_length = 0
      adapted_production_weights = {}

      for adapted_production in adapted_productions:
        max_production_length = max(max_production_length, len(adapted_production))

        # TODO compute grammar weight for this adapted rewrite.
        weight = 0.0
        adapted_production_weights[adapted_production] = weight

        words_in_adapted_yields[adapted_nonterm] |= set(adapted_production)

      nonterminal_adapted_weights[adapted_nonterm] = adapted_production_weights

      # Add the relevant number of CNF-friendly nonterminals.
      new_nonterms = ["%s%s" % (adapted_nonterm, cnf_rewrite_char * i)
                      for i in range(0, max_production_length - 1)]
      nonterminals.extend(new_nonterms)
      # Also add the relevant production rules.
      productions.extend([(ADAPTED_PRETERMINAL, new_nonterm)
                          for new_nonterm in new_nonterms])

    # TODO prep weights

    return ProjectedPCFG(
        start=self.start,
        terminals=self.terminals,
        nonterminals=nonterminals,
        preterminals=preterminals,
        productions=productions,
        words_in_adapted_yields=words_in_adapted_yields)

    # # Production list: all grammar production rules + yields of adapted
    # # nonterminals
    # productions = copy(self.productions)
    # binary_weights = self.binary_weights.copy()
    # adapted_yields = {}
    # adapted_yield_weights = {}
    # for i, nonterminal in enumerate(self.adapted_nonterminals):
    #   adapted_prods = self.adapted_productions[nonterminal]
    #   # Collect counts for all rewrite rules observed in adapted_prods
    #   rewrite_counts = {tuple(c.label() for c in adapted_prod.children): count
    #                     for adapted_prod, count in adapted_prods.items()}

    #   for j, production in enumerate(productions):


    # for i, nonterminal in enumerate(self.adapted_nonterminals):
    #   for n_prods in self.adapted_productions[nonterminal]:
    #     total_count = sum(n_prods.values()) # n_A
    #     total_types = len(n_prods) # m_A

    #     # Normalization constant for likelihood
    #     Z = 0
    #     for n_prod, count in n_prods.items():
    #       prod_yield = n_prod.leaves()

    #       # Get an index for this yield.
    #       prod_idx = adapted_yields.get(prod_yield, len(adapted_yields))
    #       adapted_yields[prod_yield] = prod_idx

    #       # Retrieve hyperparameters for this nonterminal.
    #       a, b = self.adaptor_parameters[i]

    #       # Prior probability of the relevant adaptor generating a new yield.
    #       p_new_yield = (total_types * a + b) / (total_count + b)
    #       # Normalized count of the yield in the corpus.
    #       # TODO where does alpha come from?
    #       # TODO Z needs to be calculated later
    #       unnorm_p_likelihood = count + alpha
    #       Z += unnorm_p_likelihood

    #       # Probability of choosing this yield
    #       # TODO understand


class ProjectedPCFG(FixedPCFG):

  def __init__(self, words_in_adapted_yields=None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.words_in_adapted_yields = words_in_adapted_yields
