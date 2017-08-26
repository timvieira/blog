title: A tempting algorithm for inverting max
date: 2017-08-15
comments: true
status: draft
tags: algorithm, failed-ideas

In my recent TACL paper, we came up with a clever algorithm for turning an n^5
algorithm into an n^3 algorithm, but leveraging an annealing approximation. The
algorithm we came up with was correct, but only under exact arithmetic.


Given two n-dimenstional vectors $\boldsymbol{x}$ and $\boldsymbol{f}$.

We want to compute, $f^* = f_{\text{argmax}_i x_i}$, i.e., compute the f-value that
corresponds to the highest x index. I'm going to assume that there are no ideas
in x's dimensions.

One way to approximate this which is correct in the limit $\gamma \rightarrow \inf$

$$
r_\gamma(\boldsymbol{x}) = \frac{ \bar{r}_\gamma(\boldsymbol{x}) }{ Z_\gamma(\boldsymbol{x}) }
$$

where
$$
\bar{r}_\gamma(\boldsymbol{x}) = \sum_i x_i^\gamma f_i
$$
and
$$
Z_\gamma(\boldsymbol{x}) = \sum_i x_i^\gamma
$$

$$
g_\gamma(x, y) = \sum_i x_i^\gamma y_i
$$


Cool. So that works for approximating $f^*$, but what about re-computing $f^*$
after deleting an element from $\boldsymbol{x}$?

For $\gamma = 1$ this is actually really easy because the numerator and
denominator are each linear functions in $\boldsymbol{x}$.

So we can use seperate linear extrapolations

$$
r_1(\boldsymbol{x} + \boldsymbol{\delta}) = \frac{ \bar{r}_1(\boldsymbol{x}) + \boldsymbol{\delta}^\top\! \nabla \bar{r}_1(\boldsymbol{x}) }{ Z_1(\boldsymbol{x}) + \boldsymbol{\delta}^\top\! \nabla Z_1(\boldsymbol{x}) }
$$

It's actually not hard to extend to the simple polynomial that we get for any
$\gamma$

$$
r_\gamma(\boldsymbol{x} + \boldsymbol{\delta}) = \frac{ \bar{r}_\gamma(\boldsymbol{x}) + ((x + \boldsymbol{\delta})^\gamma - x^\gamma)^\top\! \nabla \bar{r}_\gamma(\boldsymbol{x}) }{ Z_\gamma(\boldsymbol{x}) + ((x + \boldsymbol{\delta})^\gamma - x^\gamma)^\top\! \nabla Z_\gamma(\boldsymbol{x}) }
$$

------------

max is a binary associative operator that is not invertible in the sense that

We do however have a sense of *weak* inversion
(x max y) imax y = x
(x max y) imax x = y

since it does make sense to remove one of the elements we added in.

This is essentially the same computation as the "gradient of a product"
operation, which is a leave-one-out computation, that I've written about before.

We know from standard annealing arguments that we can get use summation (more
precisely, expectation) to approximate a maximization.

Here we use the softmax (i.e., log-sum-exp augmented with an inverse temperature
parameter) to approximate $\max_x f(x)$,

$$
s_\gamma(\boldsymbol{x}) = \frac{1}{\gamma} \log \sum_j \exp(x_j \cdot \gamma)
$$

$$
\lim_{\gamma \rightarrow \infty} s_\gamma(\boldsymbol{x}) = \max(\boldsymbol{x}).
$$

The gradient of $s_\gamma(\boldsymbol{x})$ is an annealed probability
distribution, $p(x_i) \propto \exp(x_i \gamma)$,

$$
\frac{\partial s_\gamma(\boldsymbol{x})}{\partial x_i} = \exp(x_i \cdot \gamma) / \left( \sum_{j} \exp(x_j \cdot \gamma) \right)
$$
