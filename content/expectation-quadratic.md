Title: Expected value of a quadratic function and the Delta method
date: 2014-07-21
comments: true
tags: math

**Expected value of a quadratic**: Suppose we'd like to compute the expectation
of a quadratic function, i.e.,
$\mathbb{E}\left[ x^{\top}\negthinspace\negthinspace A x \right]$ , where $x$ is
a random vector and $A$ is deterministic _symmetric_ matrix. Let $\mu$ and
$\Sigma$ be the mean and variance of $x$. It turns out the expected value of a
quadratic has the following simple form:

$$
\mathbb{E}\left[ x^{\top}\negthinspace\negthinspace A x \right]
=
\text{trace}\left( A \Sigma \right) + \mu^{\top}\negthinspace A \mu
$$

**Delta Method**: Suppose we'd like to compute expected value of a nonlinear
function $f$ applied our random variable $x$,
$\mathbb{E}\left[ f(x) \right]$. The Delta method approximates this expection by
replacing $f$ by it's second-order Talylor approximation $\hat{f\_{a}}$ taken at
some point $a$

$$
\hat{f_{a}}(x) = f(a) + \nabla\negthinspace f(a)^\{\top} (x - a) + \frac{1}{2} (x - a)^\top H(a)\, (x - a)
$$

The expectation of this Talyor approximation is a quadratic function! Let's try
to apply our new equation for the expected value of quadratic. We can use the
trick from above with $A=H(a)$ and $x = (x-a)$. Note, covariance matrix is shift
invariant and the Hessian is a symmetric matrix!

$$
\begin{aligned}
\mathbb{E}\left[ \hat{f_{a}}(x) \right]
 & = \mathbb{E} \left[ f(a) + \nabla\negthinspace f(a)^{\top} (x - a) + \frac{1}{2} (x - a)^{\top} H(a)\, (x - a) \right] \\\
 & = f(a) + \nabla\negthinspace f(a)^{\top} ( \mu - a ) + \frac{1}{2} \mathbb{E} \left[ (x - a)^{\top} H(a)\, (x - a) \right] \\\
 & = f(a) + \nabla\negthinspace f(a)^{\top} ( \mu - a ) +
   \frac{1}{2}\left( \text{trace}\left( H(a) \, \Sigma \right) + (\mu - a)^{\top} H(a)\, (\mu - a) \right)
\end{aligned}
$$

It's common to take the Taylor expansion around $\mu$. This simplifies the equation

\begin{aligned}
\mathbb{E}\left[ \hat{f_{\mu}} (x) \right]
&= \mathbb{E}\left[ f(\mu) + \nabla\negthinspace f(\mu) (x - \mu) + \frac{1}{2} (x - \mu)^{\top} H(\mu)\, (x - \mu) \right] \\\
&= f(\mu) + \frac{1}{2} \, \text{trace}\Big( H(\mu) \, \Sigma \Big)
\end{aligned}

That looks much more tractable! Error bounds are possible to derive, but outside
to scope of this post. For a nice use of the delta method in machine learning
see [(Wager+,'13)](http://arxiv.org/pdf/1307.1493v2.pdf) and
[(Smith & Eisner,'06)](http://cs.jhu.edu/~jason/papers/smith+eisner.acl06-risk.pdf)
