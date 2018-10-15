from nose.tools import assert_equal

from nltk import Tree
import numpy as np

from infinite_parser.inference.inside_outside import parse, update_em
from infinite_parser.pcfg import FixedPCFG, tree_from_backtrace


def test_inside_outside():
  pcfg = FixedPCFG("x",
                   terminals=["c", "d"],
                   nonterminals=["x"],
                   preterminals=["b"],
                   productions=[("b", "b")],
                   binary_weights=np.array([[0.5]]),
                   unary_weights=np.array([[0.5, 0.5]]))

  sentence = "c d".split()
  alphas, betas, backtrace = parse(pcfg, sentence)

  from pprint import pprint
  pprint(list(zip(pcfg.nonterminals, alphas)))
  pprint(list(zip(pcfg.nonterminals, betas)))

  tree_from_backtrace(pcfg, sentence, backtrace).pretty_print()
  assert_equal(tree_from_backtrace(pcfg, sentence, backtrace),
               Tree.fromstring("(x (b c) (b d))"))

  # check alpha[x]
  np.testing.assert_allclose(alphas[0], [[0, 0.25],
                                         [0, 0]], atol=1e-5)
  # check alpha[b] (preterminals)
  np.testing.assert_allclose(alphas[1], [[0.5, 0],
                                         [0, 0.5]])


def test_inside_outside2():
  pcfg = FixedPCFG("x",
                   terminals=["c", "d"],
                   nonterminals=["x"],
                   preterminals=["b"],
                   productions=[("b", "b"), ("b", "x")],
                   binary_weights=np.array([[0.25, 0.75]]),
                   unary_weights=np.array([[0.5, 0.5]]))

  sentence = "c d d".split()
  alphas, betas, backtrace = parse(pcfg, sentence)

  from pprint import pprint
  pprint(list(zip(pcfg.nonterminals, alphas)))
  pprint(list(zip(pcfg.nonterminals, betas)))

  tree_from_backtrace(pcfg, sentence, backtrace).pretty_print()
  assert_equal(tree_from_backtrace(pcfg, sentence, backtrace),
               Tree.fromstring("(x (b c) (x (b d) (b d)))"))

  # check alpha[x]
  np.testing.assert_allclose(alphas[0], [[0, 0.0625, 0.023438],
                                         [0, 0, 0.0625],
                                         [0, 0, 0]], atol=1e-5)
  # check alpha[b] (preterminals)
  np.testing.assert_allclose(alphas[1], [[0.5, 0, 0],
                                         [0, 0.5, 0],
                                         [0, 0, 0.5]])


def test_inside_outside_update():
  pcfg = FixedPCFG("x",
                   terminals=["c", "d"],
                   nonterminals=["x"],
                   preterminals=["b"],
                   productions=[("b", "b"), ("b", "x")])

  sentence = "c d d".split()

  prev_total_prob = 0

  for i in range(20):
    alphas, betas, backtrace = parse(pcfg, sentence)
    total_prob = alphas[pcfg.nonterm2idx[pcfg.start], 0, len(sentence) - 1]
    print("%d\t%f" % (i, total_prob))

    # NB include small tolerance due to float imprecision
    assert total_prob - prev_total_prob >= -1e4, \
        "Total prob should never decrease: %f -> %f (iter %d)" % \
        (prev_total_prob, total_prob, i)
    prev_total_prob = total_prob

    pcfg = update_em(pcfg, sentence)
