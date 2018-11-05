from functools import partial

from nltk import Tree
import torch as T
import pyro as P
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta, AutoContinuous
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist


def model(observed, start, nonterminals, productions, preterminals, terminals, batch_size=1):
  assert start in nonterminals

  # Sample model parameters.
  binary_params = {nonterm: P.sample("binary_%s" % nonterm,
                                     dist.Dirichlet(T.ones(len(productions))))
                   for nonterm in nonterminals}
  unary_params = {preterm: P.sample("unary_%s" % preterm,
                                    dist.Dirichlet(T.ones(len(terminals))))
                  for preterm in preterminals}

  term2idx = {term: i for i, term in enumerate(terminals)}

  def inner(observed, node, node_count, term_count, prefix=""):
    if node in nonterminals:
      node_params = binary_params[node].unsqueeze(0)
      prod_idx = P.sample("%snode_%i_prod" % (prefix, node_count), dist.Categorical(node_params))
      left, right = productions[prod_idx]
      left, node_count, term_count = inner(observed, left, node_count + 1, term_count, prefix=prefix)
      right, node_count, term_count = inner(observed, right, node_count + 1, term_count, prefix=prefix)
      return Tree(node, [left, right]), node_count, term_count
    elif node in preterminals:
      if term_count >= len(observed):
        # Not possible!
        print("too long")
        P.sample("%i_too_long" % node_count, dist.Bernoulli(1), obs=0)
        return None, node_count, term_count

      obs_term_idx = term2idx[observed[term_count]]
      node_params = unary_params[node].unsqueeze(0)
      term_idx = P.sample("%snode_%i_term" % (prefix, node_count), dist.Categorical(node_params),
                          obs=T.tensor([obs_term_idx]))
      term = terminals[term_idx]
      return Tree(node, [term]), node_count + 1, term_count + 1

  # with P.iarange("trees", len(observed)) as batch:
  #   print(batch)
  #   ret = []
  #   for x in batch:
  #     ret.append(inner(observed[x], start, 0, 0, prefix=str(x))[0])

  #   return ret
  return inner(observed, start, 0, 0)




if __name__ == '__main__':
  observed = ["the dog bit the man".split()]

  nonterminals = ["S", "NP", "VP"]
  preterminals = ["DT", "NN", "VB"]

  model = partial(model, start="S",
                  nonterminals=nonterminals,
                  productions=[("NP", "VP"), ("DT", "NN"), ("VB", "NN")],
                  preterminals=preterminals,
                  terminals=["the", "dog", "man", "bit"])

  expose_params = ["binary_%s" % nonterm for nonterm in nonterminals] \
      + ["unary_%s" % preterm for preterm in preterminals]
  guide = AutoContinuous(poutine.block(model, expose=expose_params))
  elbo = Trace_ELBO()
  optim = Adam({"lr": 0.001})
  svi = SVI(model, guide, optim, elbo)

  for _ in range(30):
    loss = svi.step(observed[0])
    print(loss)
