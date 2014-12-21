title: Importance Sampling
date: 2014-12-21
comments: true
tags: math, statistics, randomized

Importance sampling is a powerful and pervasive technique in statistics, machine
learning and randomized algorithms.

Basics
------

Importance sampling is a technique for estimating the expectation $\mu$ of a
random variable $f(x)$ under distribution $p$ from samples of a different
distribution $q$.

The key observation is that $\mu$ is can expressed as the expectation of a
different random variable $f^*(x)=\frac{p(x)}{q(x)}\! \cdot\! f(x)$ under $q$.

$$
\mathbb{E}_{q}\! \left[ f^*(x) \right] = \mathbb{E}_{q}\! \left[ \frac{p(x)}{q(x)} f(x) \right] = \sum_{x} q(x) \frac{p(x)}{q(x)} f(x) = \sum_{x} p(x) f(x) = \mathbb{E}_{p}\! \left[ f(x) \right] = \mu
$$

Technical condition: $q$ must have support everywhere $p$ does, $p(x) > 0
\Rightarrow q(x) > 0$. Without this condition, the equation is biased! Note: $q$
can support things that $p$ doesn't.

Terminology: The quantity $w(x) = \frac{p(x)}{q(x)}$ is often referred to as the
"importance weight" or "importance correction". We often refer to $p$ as the
target density and $q$ the proposal density.

Now, given samples $\{ x^{(i)} \}_{i=1}^{n}$ from $q$, we can use the Monte
Carlo estimate, $\hat{\mu} \approx \frac{1}{n} \sum_{i=1}^n f^{*}(x^{(i)})$, as
an unbiased estimator of $\mu$.

Remarks
-------

There are a few reasons we might want use importance sampling:

  1. **Convenience**: It might be trickier to sample directly from $p$.

  2. **Bias-correction**: Suppose, we're developing an algorithm which requires
     samples to satisfy some "safety" condition (e.g., a minimum support
     threshold) and be unbiased. Importance sampling can be used to remove bias,
     while satisfying the condition.

  3. **Variance reduction**: It might be the case that sampling directly from
     $p$ would require more samples to estimate $\mu$. Check out these
     [great notes](http://www.columbia.edu/~mh2078/MCS04/MCS_var_red2.pdf) for
     more.

  4. **Off-policy evaluation and learning**: We might want to collect some
     "exploratory data" from $q$ and evaluate different "policies", $p$ (e.g.,
     to pick the best one). Some cool papers:
     [counterfactual reasoning](http://arxiv.org/abs/1209.2355),
     [reinforcement learning](http://arxiv.org/abs/cs/0204043),
     [contextual bandits](http://arxiv.org/abs/1103.4601),
     [domain adaptation](http://papers.nips.cc/paper/4156-learning-bounds-for-importance-weighting.pdf).

There are a few common cases for $q$ worth separate consideration:

  1. **Control over $q$**: This is the case in experimental design, variance
     reduction, active learning and reinforcement learning. It's often difficult
     to design $q$, which results in an estimator with "reasonable" variance. A
     very difficult case is in off-policy evaluation because it (essentially)
     requires a good exploratory distribution for every possible policy. (I have
     much more to say on this topic.)

  2. **Little to no control over $q$**: For example, you're given some dataset
     (e.g., new articles) and you want to estimate performance on a different
     dataset (e.g., Twitter).

  3. **Unknown $q$**: In this case, we want to estimate $q$ (typically referred
     to as the propensity score) and use it in the importance sampling
     estimator. This technique, as far as I can tell, is widely used to remove
     selection bias when estimating effects of different treatments.

**Drawbacks**: The main drawback of importance sampling is variance. A few bad
samples with large weights can drastically throw off the estimator. Thus, it's
often the case that a biased estimator is preferred, e.g.,
[estimating the partition function](https://hips.seas.harvard.edu/blog/2013/01/14/unbiased-estimators-of-partition-functions-are-basically-lower-bounds/),
[clipping weights](http://arxiv.org/abs/1209.2355),
[indirect importance sampling](http://arxiv.org/abs/cs/0204043). A secondary
drawback is that both densities must be normalized, which is often intractable.

**What's next?** I plan to cover "variance reduction" and "off-policy
evaluation" in more detail in future posts.
