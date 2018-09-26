from nose.tools import assert_equal

from nltk import Tree
import numpy as np

from infinite_parser.pcfg import FixedPCFG, inside_outside, inside_outside_update, tree_from_backtrace


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
