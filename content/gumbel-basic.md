Title: Gumbel max trick
date: 2014-07-31
comments: true
tags: math, sampling, Gumbel


**Goal**: Sampling from a discrete distribution parametrized by unnormalized
log-probabilities:

$$
\pi_k = \frac{1}{z} \exp(x_k)   \ \ \ \text{where } z = \sum_{j=1}^K \exp(x_j)
$$

**The usual way**: Exponentiate and normalize (using the
[exp-normalize trick](/blog/post/2014/02/11/exp-normalize-trick/)), then use the
an algorithm for sampling from a discrete distribution (aka categorical):

```python
def usual(x):
    cdf = exp(x - x.max()).cumsum()     # the exp-normalize trick
    z = cdf[-1]
    u = uniform(0,1)
    return cdf.searchsorted(u * z)
```

**The Gumbel max trick**:

$$
y = \underset{ i \in \{1,\cdots,K\} }{\operatorname{argmax}} x_i + z_i
$$

where $z_1 \cdots z_K$ are i.i.d. $\text{Gumbel}(0,1)$ random variates. It
turns out that $y$ is distributed according to $\pi$. (See the short derivations
in this
[blog post](https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/).)

Implementing the Gumbel max trick is remarkable easy:

```python
def gumbel_max_sample(x):
    z = gumbel(loc=0, scale=1, size=x.shape)
    return (x + g).argmax(axis=1)
```

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
     e.g. python+numpy. See my later [post](/blog/fast-sampling-sigmoid.md) on
     fast sigmoid sampling (i.e., when $K=2$).
