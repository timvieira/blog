Title: Exp-normalize trick
date: 2014-02-11
comments: true
tags: math

This trick is the very close cousin of the infamous log-sum-exp trick
([scipy.misc.logsumexp](http://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.logsumexp.html)),

Supposed you'd like to evaluate a probability distribution $\boldsymbol{\pi}$
parametrized by a vector $\boldsymbol{x} \in \mathbb{R}^n$ as follows:

$$
\pi_i = \frac{ \exp(x_i) }{ \sum_{j=1}^n \exp(x_j) }
$$

The exp-normalize trick leverages the following identity to avoid numerical
overflow. For any $b \in \mathbb{R}$,

$$
\pi_i
= \frac{ \exp(x_i - b) \exp(b) }{ \sum_{j=1}^n \exp(x_j - b) \exp(b) }
= \frac{ \exp(x_i - b) }{ \sum_{j=1}^n \exp(x_j - b) }
$$

In other words, the $\boldsymbol{\pi}$ is shift-invariant. A reasonable choice
is $b = \max_{i=1}^n x_i$. With this choice, overflow due to $\exp$ is
impossible$-$the largest number exponentiated after shifting is $0$.

**Exp-normalize v. log-sum-exp**

If what you want to remain in log-space, that is, compute
$\log(\boldsymbol{\pi})$, you should use logsumexp. However, if
$\boldsymbol{\pi}$ is your goal, then exp-normalize trick is for you! Since it
avoids additional calls to $\exp$, which would be required if using log-sum-exp.


**Log-sum-exp for computing the log-distibution**

$$
\log \pi_i = x_i - \mathrm{logsumexp}(\boldsymbol{x})
$$

where
$$
\mathrm{logsumexp}(\boldsymbol{x}) = b + \log \sum_{j=1}^n \exp(x_j - b)
$$

Typically with the same choice for $b$ as above.

**Numerically-stable sigmoid function**

The sigmoid function can be computed with the exp-normalize trick in order to
avoid numerical overflow. In the case of $\text{sigmoid}(x)$, we have a
distribution with unnormalized log probabilities $[x,0]$, where we are only
interested in the probability of the first event. From the exp-normalize
identity, we know that the distributions $[x,0]$ and $[0,-x]$ are equivalent (to
see why, plug in $b=\max(0,x)$). This is why sigmoid is often expressed in one
of two equivalent ways:

$$
\text{sigmoid}(x) = 1/(1+\exp(-x)) = \exp(x) / (\exp(x) + 1)
$$

Interestingly, each version covers an extreme case: $x=\infty$ and $x=-\infty$,
respectively. Below is some python code which implements the trick:

    :::python
    def sigmoid(x):
        "Numerically-stable sigmoid function."
        if x >= 0:
            z = exp(-x)
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = exp(x)
            return z / (1 + z)
