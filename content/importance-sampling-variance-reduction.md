title: Importance sampling for variance reduction
date: 2017-08-18
comments: true
status: draft
tags: calculus, automatic-differentiation

I've written before CITE about how the "optimial proposal distribution" is not
$p$. Here, I'll spell out how to derive proposal distributions that improve on
$p$.

We $I_n$ is lower variance reduction the $M_n$ *if and only if* the following
holds:

$$
\underset{x \sim p}{\mathbb{E}}\left[ r(x) \cdot \left( \frac{p(x)}{q(x)} - 1 \right) \right] < 0
$$

XXX: is this correct? not $r^2$?

This is because the variance of the importance sampled estimator $I_n$ after $n$
samples is related to the Monte Carlo estimator $M_n$ via the following
equation:

$$
\begin{eqnarray*}
V( I_n )
&=& \mathbb{E}_{q}\!\left[ \left( \frac{1}{n} \sum_{i=1}^n \frac{p_i}{q_i} r_i - \mu\right)^2 \right] \\
&=& \frac{1}{n} \mathbb{E}_{q}\!\left[ \left( \frac{p}{q} r \right)^2 \right] - \mu^2/n \\
&=& \frac{1}{n} \mathbb{E}_{p}\!\left[ \frac{q}{p} \frac{p^2}{q^2} r^2 \right] - \mu^2/n \\
&=& \frac{1}{n} \mathbb{E}_{p}\!\left[ \frac{p}{q} r^2 \right] - \mu^2/n \\
\\
V( M_n )
&=& \frac{1}{n} \mathbb{E}_{p}\!\left[ r^2 \right] - \mu^2/n \\
\\
V( M_n ) - V( I_n )
&=&
  \left(\frac{1}{n} \mathbb{E}_{p}\!\left[ r^2 \right] - \mu^2/n\right)
  - \left( \frac{1}{n} \mathbb{E}_{p}\!\left[ \frac{p}{q} r^2 \right] - \mu^2/n \right) \\
&=&
  \frac{1}{n} \left( \mathbb{E}_{p}\!\left[ r^2 \right]
  - \mathbb{E}_{p}\!\left[ \frac{p}{q} r^2 \right] \right) \\
&=&
  \frac{1}{n} \mathbb{E}_{p}\!\left[ r^2 - \frac{p}{q} r^2\right] \\
&=&
  \frac{1}{n} \mathbb{E}_{p}\!\left[ \left( 1 - \frac{p}{q} \right) r^2\right] \\
\end{eqnarray*}
$$

Note that all proposal distributions $q$ are *also* valid control variates since
we know their expected value is one. Control variates basically always reduce
variance. The only caveat is that we need to learn a combination coefficient -
bad or expensive estimation can lead to a bad control variate estimator... this
seems rare.


# Adaptive importance sampling proposal distributions with the cross-entropy method

The [Cross Entropy Method](http://en.wikipedia.org/wiki/Cross-entropy_method)
(see also: [tutorial](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf)).

Minimizes KL divergence between the sampling density $q$ and unwieldy "optimal"
proposal density (much in which direction!?), $q^{*}(x) \propto r(x)\! \cdot\!
p(x)$. CE leverages the fact that minimizing KL diverenge doesn't requires
normalizing $q^*$.

There are two ways that you can do this minimization: in the "wrong" direction
(cite KL post), or the "right" direction (with something RAML/SMM, which is
another importance sampling procedure).

It would be bad to minimize in the wrong direction because it could easily
violate the importance support conditions $p(x)>0 \Rightarrow q(x)>0$ for all
$x$ in the domain of $p$.

The optimization procedures looks exactly like policy gradient or black-box
stochastic variational inference.

This seems to imply that we can get a better estimator by running two rounds of
estimation. One with samples from $q(\theta_0)$, then with updated samples
$q(\theta_1)$, the difference is that the policy we're trying to estimate is
constant (normally we'd estimate $V(\pi_{\theta_t})$, in this case $\pi$ is not
linked to the same $\theta$ as the proposal distribution).
