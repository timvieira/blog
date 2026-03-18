title: A Tempting Algorithm for Inverting Max
date: 2017-08-15
comments: true
status: draft
tags: algorithms, failed-ideas

**The problem:** Given a vector $\boldsymbol{x} \in \mathbb{R}^n$ with no ties,
we want to efficiently compute "leave-one-out" max operations.  That is, for
each $i$, compute
$$
m_{-i} = \max_{j \ne i}\, x_j.
$$

For **sum**, this is trivial: $\text{sum}_{-i} = S - x_i$ where $S = \sum_j
x_j$.  We say sum is *invertible* in the sense that we can undo the
contribution of any element in constant time.

**Max is not invertible** in this sense.  If you remove the *non*-maximum
element, easy: $m_{-i} = \max(\boldsymbol{x})$.  But if you remove the maximum
element, you have to rescan. And you don't know which case you're in without
checking.

(Yes, you can compute all $n$ leave-one-out max values in $O(n)$ time with
prefix and suffix scans&mdash;but those aren't *incremental*: a single change to
$\boldsymbol{x}$ means redoing the whole scan.)


## The tempting idea

Sum is invertible.  We know how to approximate max with a sum.  So: approximate
max with a sum-based surrogate, do the leave-one-out on the surrogate (cheap!),
and crank up a temperature parameter to make the approximation tight.


## Approach 1: Softmax (log-sum-exp)

The standard way to smooth max into a sum is the log-sum-exp with inverse
temperature $\gamma$:

$$
s_\gamma(\boldsymbol{x}) = \frac{1}{\gamma} \log \sum_j \exp(x_j \cdot \gamma)
$$

As $\gamma \to \infty$, this converges to $\max(\boldsymbol{x})$.

The gradient is a Boltzmann distribution over the indices,

$$
\frac{\partial s_\gamma(\boldsymbol{x})}{\partial x_i}
= \frac{\exp(x_i \cdot \gamma)}{\sum_{j} \exp(x_j \cdot \gamma)}
= p_\gamma(i \mid \boldsymbol{x}),
$$

which concentrates all mass on the argmax as $\gamma \to \infty$.

Because $s_\gamma$ is a log of a sum of exponentials, its leave-one-out version
is just as easy to write down: drop the $i$-th term from the sum.

$$
s_{\gamma,-i}(\boldsymbol{x})
= \frac{1}{\gamma} \log \!\left( \sum_j \exp(x_j \cdot \gamma) - \exp(x_i \cdot \gamma) \right)
$$

In other words, $\exp(\gamma \cdot s_\gamma)$ is a plain sum, so we can "invert"
it by subtraction and take the log to get back.


## Approach 2: Power-mean weighting

Here's a related trick for a slightly different problem.  Suppose we have two
vectors $\boldsymbol{x}$ and $\boldsymbol{f}$, and we want to compute the
$f$-value corresponding to the max $x$-index:

$$
f^* = f_{\text{argmax}_i\, x_i}.
$$

We can approximate this with a weighted average where the weights are powers of
$x_i$:

$$
r_\gamma(\boldsymbol{x})
= \frac{\bar{r}_\gamma(\boldsymbol{x})}{Z_\gamma(\boldsymbol{x})}
\quad \text{where} \quad
\bar{r}_\gamma(\boldsymbol{x}) = \sum_i x_i^\gamma\, f_i
\quad \text{and} \quad
Z_\gamma(\boldsymbol{x}) = \sum_i x_i^\gamma.
$$

As $\gamma \to \infty$, the weight concentrates entirely on the largest $x_i$
and we recover $f^*$.

For **$\gamma = 1$**, both numerator and denominator are linear in
$\boldsymbol{x}$, so updating after a perturbation $\boldsymbol{\delta}$ is
trivial:

$$
r_1(\boldsymbol{x} + \boldsymbol{\delta})
= \frac{
  \bar{r}_1(\boldsymbol{x}) + \boldsymbol{\delta}^\top\! \boldsymbol{f}
}{
  Z_1(\boldsymbol{x}) + \boldsymbol{\delta}^\top\! \boldsymbol{1}
}.
$$

For general $\gamma$, the numerator and denominator are still sums (just of
$x_i^\gamma$ terms), so we can apply the same subtract-and-recompute idea:

$$
r_{\gamma,-i}(\boldsymbol{x})
= \frac{
  \bar{r}_\gamma(\boldsymbol{x}) - x_i^\gamma\, f_i
}{
  Z_\gamma(\boldsymbol{x}) - x_i^\gamma
}.
$$


## Why it doesn't work

Both approaches share the same fatal flaw: **the limit that makes the
approximation accurate is the same limit that destroys numerical stability.**

As $\gamma$ grows:

- In the softmax version, $\exp(x_i \cdot \gamma)$ overflows for even
  moderate $x_i$.

- In the power-mean version, $x_i^\gamma$ overflows similarly (assuming
  $x_i > 1$; for $x_i < 1$ it underflows to zero).

- The subtraction step $(\text{big number} - \text{big number})$ suffers from
  catastrophic cancellation.  The very case we care about&mdash;removing the
  maximum element&mdash;is exactly the case where the two terms are closest in
  magnitude, so the relative error is worst precisely when it matters most.

For small $\gamma$, the arithmetic is fine but the approximation is poor.  For
large $\gamma$, the approximation is tight but the arithmetic falls apart.
There's no sweet spot.

I ran into this in
[Learning to Prune (Vieira & Eisner, TACL 2017)](https://aclanthology.org/Q17-1019.pdf).
We needed to efficiently compute leave-one-out rollouts for a CKY parser during
training with LOLS.  The naive approach ran the parser $T$ times (once per
rollout), giving $O(n^5)$ per sentence.  In theory, we could anneal from expected
recall to 1-best recall, turning the leave-one-out into an invertible sum and
bringing the cost down to $O(n^3)$.  The algorithm was correct under exact
arithmetic, but we found it to be numerically unstable even with high-precision
arithmetic libraries.  The fix was a completely different algorithmic
approach&mdash;change propagation and backpropagation&mdash;that avoided the need
to "invert" a max at all.


## Takeaway

"Smooth it and invert the smooth version" is a natural idea, and it works
beautifully for operations like sum and product.  For max, it's a mirage: the
smoothing parameter that makes the approximation faithful is the same one that
makes the numerics blow up.  Sometimes the right move is to find an algorithm
that sidesteps the inversion entirely.
