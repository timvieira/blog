Title: Gumbel max trick
date: 2014-07-31
comments: true
tags: math


**Goal**: Sampling from a discrete distribution parametrized by unnormalized
log-probabilities:

$$
\pi\_k = \frac{1}{z} \exp(x\_k)   \ \ \ \text{where } z = \sum\_{j=1}^K \exp(x_j)
$$

**The usual way**: Exponentiate and normalize (using the
[exp-normalize trick](/blog/post/2014/02/11/exp-normalize-trick/)), then use the
an algorithm for sampling from a discrete distribution (aka categorical):

    def usual(x):
        cdf = exp(x - x.max()).cumsum()     # the exp-normalize trick
        z = cdf[-1]
        u = uniform(0,1)
        return cdf.searchsorted(u * z)

**The Gumbel max trick**:

$$
y = \text{argmax}\_{i \in \\{1,\cdots,K\\}} x\_i + z\_i
$$

where $z\_1 \cdots z\_K$ are i.i.d. $\text{Gumbel}(0,1)$ random variates. It
turns out that $y$ is distributed according to $\pi$. (See the short derivations
in this
[blog post](https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/).)

Implementing the Gumbel max trick is remarkable easy:

    def gumbel_max_sample(x):
        z = gumbel(loc=0, scale=1, size=x.shape)
        return (x + g).argmax(axis=1)

If you don't have access to a Gumbel random variate generator, you can use
$-\log(-\log(\text{Uniform}(0,1))$

**Comparison**:

  1. Gumbel max requires $K$ samples from a uniform. Usual requires only $1$.

  2. Gumbel is one-pass because it does not require normalization (or a pass to
     compute the max for use in the exp-normalize trick). More on this in a
     later post!

  3. The Gumbel max trick requires $2K$ calls to $\log$, whereas ordinary
     requires $K$ calls to $\exp$. Since $\exp$ and $\log$ are expensive
     function, we'd like to avoid calling them. What gives? Well, Gumbel's calls
     to $\log$ do not depend on the data so they can be precomputed; this is
     handy for implementations which rely on vectorization for efficiency,
     e.g. python+numpy. By the way, an interesting alternative approach to
     avoiding $\exp$ is described on Justin Domke's blog post on
     [speeding up sampling from a sigmoid](http://justindomke.wordpress.com/2014/01/08/reducing-sigmoid-computations-by-at-least-88-0797077977882/),
     which discusses a trick for $K=2$. (Make sure you read the comments!)
