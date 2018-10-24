from nose.tools import assert_equal

from nltk import Tree
import numpy as np
from scipy.special import digamma

from infinite_parser.inference.adaptor import *
from infinite_parser.pcfg import FixedPCFG, tree_from_backtrace



def _dummy_ag():
  ag = AdaptorGrammar(
      start='Morph',
      terminals=['a', 'b', 'c'],
      nonterminals=['Morph'],
      preterminals=['Phoneme'],
      adapted_nonterminals=['Morph'],
      productions=[('Phoneme', 'Morph'), ('Morph', 'Phoneme'), ('Phoneme', 'Phoneme')])

  # Simulate an adapted production "aba"
  # ag.adapted_productions['Morph'][Tree.fromstring("(Morph (Phoneme a) (Morph$ (Phoneme b) (Phoneme a)))")] = 1.0
  ag.adapted_productions['Morph'][("a", "b", "a")] = 1.0

  return ag


def _get_tree_logprob(pcfg, alpha, tree):
  """
  Get the log-probability of a full tree derivation for a sentence under the given
  inside probability chart.
  """
  lprob = 0
  # stack elements: node, left bound of span, right bound of span (both inclusive)
  stack = [(tree, 0, len(tree.leaves()) - 1)]
  while stack:
    node, i, j = stack.pop()
    idx = pcfg.nonterm2idx[node.label()] if node.label() in pcfg.nonterm2idx \
        else len(pcfg.nonterminals) + pcfg.preterm2idx[node.label()]

    lprob += np.log(alpha[idx, i, j])

    # NB assumes binary tree
    if len(node) == 2:
      left, right = list(node)
      split = len(left.leaves())
      stack.append((left, i, i + split - 1))
      stack.append((right, i + split, j))
    else:
      assert i == j

  return lprob


def test_project_pcfg():
  ag = _dummy_ag()
  pcfg = ag.project_pcfg()

  adapted_preterm = "#AD"
  assert adapted_preterm in pcfg.preterminals, \
      "Adapted preterminal %s should exist" % adapted_preterm

  assert "Morph#" in pcfg.nonterminals, \
      "CNF-friendly binarized nonterminals of adapted nonterminal 'Morph' should exist"

  # TODO test weights


def test_parse_projected_pcfg():
  ag = _dummy_ag()
  pcfg = ag.project_pcfg()
  parser = InsideOutsideProjectedParser(pcfg)

  cases = [
    ("a b a", ("(Morph (#AD a) (Morph# (#AD b) (#AD a)))",
               "(Morph (Phoneme a) (Morph (Phoneme b) (Phoneme a)))")),
    ("a b a b", ("(Morph (Morph (#AD a) (Morph# (#AD b) (#AD a))) (Phoneme b))",
                 "(Morph (Phoneme a) (Morph (Phoneme b) (Morph (Phoneme a) (Phoneme b))))")),
  ]

  def test_case(sentence, parse_cases):
    sentence = sentence.split()
    alphas, betas, backtrace = parser.parse(sentence)

    t = tree_from_backtrace(pcfg, sentence, backtrace)
    assert t is not None, \
        "Sentence '%s' should parse in projected PCFG" % " ".join(sentence)
    t.pretty_print()

    adapted_parse, non_adapted_parse = parse_cases
    adapted_parse = Tree.fromstring(adapted_parse)
    non_adapted_parse = Tree.fromstring(non_adapted_parse)

    np.testing.assert_array_less(-np.inf, _get_tree_logprob(pcfg, alphas, adapted_parse),
                                "Adapted nonterminal parse should have nonzero probability")
    np.testing.assert_array_less(-np.inf, _get_tree_logprob(pcfg, alphas, non_adapted_parse),
                                "Non-adapted parse should have nonzero probability")


  for sentence, parse_cases in cases:
    yield test_case, sentence, parse_cases

  # Test that we can parse other sentences, too.
  sentence = "a b".split()
  alphas, betas, backtrace = parser.parse(sentence)
  t = tree_from_backtrace(pcfg, sentence, backtrace)
  # TODO

  # Test that we can parse a sentence containing the adapted rewrite
  sentence = "a b a b".split()
  alphas, betas, backtrace = parser.parse(sentence)
  t = tree_from_backtrace(pcfg, sentence, backtrace)
  # TODO check that adapted nonterminal is an option
  # TODO check that non-adapted parse is also an option

