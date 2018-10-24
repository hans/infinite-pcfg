from nose.tools import assert_equal

from nltk import Tree
import numpy as np
from scipy.special import digamma

from infinite_parser.inference.adaptor import *
from infinite_parser.pcfg import FixedPCFG, tree_from_backtrace



def _dummy_ag():
  ag = AdaptorGrammar(
      start='Word',
      terminals=['a', 'b', 'c'],
      nonterminals=['Word', 'Morph', 'Morph$'],
      preterminals=['Phoneme'],
      adapted_nonterminals=['Morph'],
      productions=[('Morph', 'Morph'), ('Phoneme', 'Phoneme')])

  # Simulate an adapted production "aba"
  # ag.adapted_productions['Morph'][Tree.fromstring("(Morph (Phoneme a) (Morph$ (Phoneme b) (Phoneme a)))")] = 1.0
  ag.adapted_productions['Morph'][("a", "b", "a", "b")] = 1.0

  return ag


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

  sentence = "a b a b".split()
  alphas, betas, backtrace = parser.parse(sentence)

  t = tree_from_backtrace(pcfg, sentence, backtrace)
  t.pretty_print()
  assert t is not None
  # TODO check that adapted nonterminal is an option
  non_adapted_parse = Tree.fromstring("(Word (Morph (Phoneme a) (Phoneme b)) (Morph (Phoneme a) (Phoneme b)))")
  # TODO check that non-adapted parse is also an option

  return

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

