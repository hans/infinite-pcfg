from nose.tools import assert_equal

from nltk import Tree
import numpy as np
from scipy.special import digamma

from infinite_parser.inference.adaptor import *
from infinite_parser.inference.inside_outside import parse
from infinite_parser.pcfg import FixedPCFG, tree_from_backtrace



def _dummy_ag():
  return AdaptorGrammar(
      start='Word',
      terminals=['a', 'b', 'c'],
      nonterminals=['Word', 'Morph', 'Morph$'],
      preterminals=['Phoneme'],
      adapted_nonterminals=['Morph'],
      productions=[('Morph', 'Morph$')])


def test_project_pcfg():
  ag = _dummy_ag()

  # Simulate an adapted production "aba"
  # ag.adapted_productions['Morph'][Tree.fromstring("(Morph (Phoneme a) (Morph$ (Phoneme b) (Phoneme a)))")] = 1.0
  ag.adapted_productions['Morph'][("a", "b", "a")] = 1.0
  pcfg = ag.project_pcfg()

  adapted_preterm = "#AD"
  assert adapted_preterm in pcfg.preterminals, \
      "Adapted preterminal %s should exist" % adapted_preterm

  assert "Morph#" in pcfg.nonterminals, \
      "CNF-friendly binarized nonterminals of adapted nonterminal 'Morph' should exist"

  # TODO test weights
