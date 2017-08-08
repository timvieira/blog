title: KL-divergence as an objective function
date: 2014-10-06
tags: statistics, machine-learning
comments: true

It's well-known that
[KL-divergence](http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
is not symmetric, but which direction is right for fitting your model?

First, which one is which?

**Cheat sheet**: If we're fitting $q$ to $p$ using

$\textbf{KL}(p || q)$

  - mean-seeking, inclusive (more principled because approximates the *full* distribution)

  - requires normalization wrt $p$ (i.e., often *not* computationally convenient)

$\textbf{KL}(q || p)$

  - mode-seeking, exclusive

  - no normalization wrt $p$ (i.e., computationally convenient)


**How I remember which is which**:
[Philip Resnik](https://www.umiacs.umd.edu/~resnik/) has an excellent
mantra/mnemonic: "the truth comes first." This&mdash;combined with the fact that
inclusive KL is the more principled choice for fitting a
distribution&mdash;makes the order really easy to remember!

As far as remembering the equation, I pretend that "$||$" is a division symbol,
which happens to correspond nicely to a division symbol in the equation (I'm not
sure it's intentional).



## Computational perspecive

Let's look at what's involved in fitting a model $q_\theta$ in each
direction. In this section, I'll describe the gradient and pay special attention
to the issue of normalization.

**Notation**: $p,q_\theta$ are probabilty distributions. $p = \bar{p} / Z_p$,
where $Z_p$ is the normalization constant. Similarly for $q$.

### The easy direction $\textbf{KL}(q_\theta || p)$

\begin{align*}
\textbf{KL}(q_\theta || p)
&= \sum_d q(d) \log \left( \frac{q(d)}{p(d)} \right) \\
&= \sum_d q(d) \left( \log q(d) - \log p(d) \right) \\
&= \underbrace{\sum_d q(d) \log q(d)}_{-\text{entropy}} - \underbrace{\sum_d q(d) \log p(d)}_{\text{cross-entropy}} \\
\end{align*}

Let's look at normalization of $p$, the entropy term is easy because there is no $p$ in it.
\begin{align*}
\sum_d q(d) \log p(d)
&= \sum_d q(d) \log (\bar{p}(d) / Z_p) \\
&= \sum_d q(d) \left( \log \bar{p}(d) - \log Z_p) \right) \\
&= \sum_d q(d) \log \bar{p}(d) - \sum_d q(d) \log Z_p \\
&= \sum_d q(d) \log \bar{p}(d) - \log Z_p
\end{align*}

In this case, $-\log Z_p$ is an additive constant, which can be dropped because
we're optimizing.

This leaves us with the following optimization problem:
\begin{align*}
& \text{argmin}_\theta \textbf{KL}(q_\theta || p) \\
&\qquad = \text{argmin}_\theta \sum_d q_\theta(d) \log q_\theta(d) - \sum_d q_\theta(d) \log \bar{p}(d)
\end{align*}

Let's work out the gradient
\begin{align*}
& \nabla\left[ \sum_d q_\theta(d) \log q_\theta(d) - \sum_d q_\theta(d) \log \bar{p}(d) \right] \\
&\qquad = \sum_d \nabla \left[ q_\theta(d) \log q_\theta(d) \right] - \sum_d \nabla\left[ q_\theta(d) \right] \log \bar{p}(d) \\
&\qquad = \sum_d \nabla \left[ q_\theta(d) \right] \left( 1 + \log q_\theta(d) \right) - \sum_d \nabla\left[ q_\theta(d) \right] \log \bar{p}(d) \\
&\qquad = \sum_d \nabla \left[ q_\theta(d) \right] \left( 1 + \log q_\theta(d) - \log \bar{p}(d) \right) \\
&\qquad = \sum_d \nabla \left[ q_\theta(d) \right] \left( \log q_\theta(d) - \log \bar{p}(d) \right) \\
\end{align*}

We killed the one in the last equality because $\sum_d \nabla
\left[ q(d) \right] = \nabla \left[ \sum_d q(d) \right] = \nabla
\left[ 1 \right] = 0$, for any $q$ which is a probability distribution.

This direction is convenient because we don't need to normalize
$p$. Unfortunately, the "easy" direction is nonconvex in general --- unlike the
"hard" direction, which is convex.

### Harder direction $\textbf{KL}(p || q_\theta)$

\begin{align*}
\textbf{KL}(p || q_\theta)
&= \sum_d p(d) \log \left( \frac{p(d)}{q(d)} \right) \\
&= \sum_d p(d) \left( \log p(d) - \log q(d) \right) \\
&= \sum_d p(d) \log p(d) - \sum_d p(d) \log q(d) \\
\end{align*}

Clearly the first term (entropy) won't matter if we're just trying optimize wrt
$\theta$. So, let's focus on the second term (cross-entropy).
\begin{align*}
\sum_d p(d) \log q(d)
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \log \left( \bar{q}(d)/Z_q \right) \\
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \log \left( \bar{q}(d)/Z_q \right) \\
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \left( \log \bar{q}(d) - \log Z_q \right) \\
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \log \bar{q}(d) - \frac{1}{Z_p} \sum_d \bar{p}(d) \log Z_q \\
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \log \bar{q}(d) - \log Z_q \frac{1}{Z_p} \sum_d \bar{p}(d) \\
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \log \bar{q}(d) - \log Z_q
\end{align*}

Unfortunately the $Z_p$ factor is unavoidable. The usual "approximate inference
story" is that $Z_p$ is hard to compute, while the approximating distributions
normalization constant $Z_q$ is easy.

Nonetheless, optimizing KL in this direction is still useful. Examples include,
expectation propagation, variational decoding and maximum likelihood
estimation. In the case of maximum likelihood estimation, $p$ is the empirical
distribution, so technically you don't have to compute its normalizing constant,
but you do need to sample from it (which can be just as hard).

The gradient, when $q$ is in the exponential family, is intuitive:

\begin{align*}
\nabla \left[ \frac{1}{Z_p} \sum_d \bar{p}(d) \log \bar{q}(d) - \log Z_q \right]
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \nabla \left[ \log \bar{q}(d) \right] - \nabla \log Z_q \\
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \phi_q(d) - \mathbb{E}_q \left[ \phi_q \right] \\
&= \mathbb{E}_p \left[ \phi_q \right] - \mathbb{E}_q \left[ \phi_q \right]
\end{align*}

Optimization problem is convex when $q_\theta$ is an exponential families ---
i.e., $p$ can be arbitrary. You can think of maximum likelihood estimation as a
method which minimizes KL divergence from samples of $p$. In this case, $p$ is
the true data distribution! The first term in the gradient is based on a sample
instead of an exact estimate (often called "observed feature counts").

Downside: computing $\mathbb{E}_p \left[ \phi_q \right]$ might not be tractable.


## Remarks

- Both directions of KL are special cases of $\alpha$-divergence. For a unified
  account of both directions consider looking into $\alpha$-divergence.

- Inclusive divergences require $q > 0$ whenever $p > 0$. No "false negatives".

- Exclusive divergences will often favor a single mode.

- Computing the value of KL (not just the gradient) in either direction requires
  normalization. However, in the "easy" direction, using unnormalized $p$
  results in only an additive constant difference. So, it's still just as
  useful, if all you care about is optimization (fitting the model).
