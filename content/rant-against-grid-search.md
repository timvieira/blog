Title: Rant against grid search
date: 2014-07-22
comments: true
tags: math

Grid search is a simple and intuitive algorithm for optimizing and/or exploring
the effects of parameters to a function. However, given its rigid definition
grid search is susceptible to degenerate behavior. One type of unfortunate
behavior occurs in the presence of unimportant parameters, which results in many
(potentially expensive) function evaluations being wasted.

This is a very simple point, but nonetheless I'll illustrate with a simple
example.

Consider the following simple example, let's find the argmax of $f(x,y) = -x^2$.

Suppose we search over a $10$-by-$10$ grid, resulting in a total of $100$
function evaluations. For this function, we expect precision which proportional
of the number of samples in the $x$-dimension, which is only $10$ samples! On
the other hand, randomly sampling points over the same space results in $100$
samples in every dimension.

In other words, randomly sample instead of using a rigid grid. If you have
points, which are not uniformly spaced, I'm willing to bet that an appropriate
probability distribution exists.

This type of problem is common on hyperparameter optimizations. For futher
reading see
[Bergstra & Bengio (2012)](http://jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).

**Other thoughts**:

1. Local search is often much more effective. For example, gradient-based
   optimization, Nelder-Mead, stochastic local search, coordinate ascent.

2. Grid search tends to produce nicer-looking plots.

3. What about variance in the results? Two things: (a) This is a concern for
   replicability, but is easily remedied by making sampled parameters
   available. (b) There is always some probability that the sampling gives you a
   terrible set of points. This shouldn't be a problem if you use enough
   samples.
