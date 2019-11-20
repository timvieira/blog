title: Hot take: Expectation maximization
date: 2019-11-20
comments: true
tags: better-explained, machine-learning, statistics,

The expectation maximization algorithm (EM) is a poorly understood and poorly
explained.

 * Explanations are littered with distracting model-specific details and messy
   variational inference notation that make it hard to see the signal in the
   noise.  Why does this happen? Well, the theoretical foundations for EM are
   based on an iterative lower-bound maximization algorithm.  This view is
   useful because they tie EM to a theoretical optimization foundation that is
   used to prove convergence properties, but at the end of the day they are just
   some dude's hot take.

 * Today, I will give my hot take.  I will create a bridge between EM and MLE,
   which I believe makes it easier to see the signal in the noise.  There are no
   KL divergences, lower bounds, or convergence proofs to distract.

EM is an algorithm for maximum-likelihood estimation when we have "incomplete
observations."

Let $\{ X_i \}_{i=1}^n$ be a set of i.i.d. random variables.

An **incomplete observation** is a *constraint* on what $X_i$ might have been if
only we had actually observed it.  In other words, we do not observe $X_i$, we
observe that $g(X_i)$ is true.

Examples,

1. Complete observation: $g(X_i) = 1(X_i = x_i)$ where $1(\cdot)$ is the
   indicator function.

2. Incomplete, interval observation: be $g(X_i) = 1(a_i \le X_i \le b_i)$ for
   some constant $a_i$ and $b_i$.  We recover the complete case when $a_i = b_i$
   for all $i$.  Interval observations are used in estimation for censored
   observations.

3. Incomplete, subset observations: we could observe subsets of X's domain (this
   works for but continuous and discrete domains)

There are tons of weird incomplete observations types (i.e., families of $g$
functions).  Every choice will result is some set of details that you will have
to work out.

The obvious thing to optimize would be the **incomplete log-likelihood**
$$
\newcommand{\defeq}[0]{\overset{\scriptsize\text{def}}{=}}
\mathcal{L}(\theta) \defeq \sum_{i=1}^n \log p_\theta( g(X_i) )
= \sum_{i=1}^n \log \sum_{x \in \mathcal{X} } p_\theta(x) g(x)
$$

We can optimize $\mathcal{L}$ directly assuming $p_\theta$ is continuously
differentiable with respect to $\theta$.  However, $\mathcal{L}$ is generally
nonconvex so we will only get locally optimal approximation.

EM takes a different approach based on a chicken and egg type of story:

 1. if we had the complete data observations, we'd get $\theta$ via MLE.

 2. if we had $\theta$, we could get complete the data by sampling from the
    model (or, more generally, using the model's distribution over it).

That's basically what EM is going to do:

- initialization: guess a model (alternatively, guess data completions and
  reorder the two steps below ME rather than EM)
- E step: generate fake complete data: use the model to complete that data in a
  manner which is consistent with the observations.
- M step: re-fit the model to that fake data,
- repeat until some convergence criterion is met

Consider a slightly dumbed down version of EM, called Monte Carlo EM (MCEM).

MCEM just (iteratively) fills in the missing information about the random
variables by sampling what the complete value were based on the current estimate
of the model $\widehat{\theta}$ (sometimes called bootstrapping).

- E-step: gives us "complete" dataset that satisfies the observed constraints,
  for each $i$ sample $\widehat{x}_i \sim  p_{\widehat{\theta}}(\cdot \mid g(X))$.

- M step: compute fully observed maximum-likelihood estimate of $\theta$ on the
  approximate data set $\{ \widehat{x}_i \}_{i=1}^n$.

From MCEM to EM: We could take more than one sample per $i$ in the E step as
long as we take an equal number per example (in expectation).  If we took
infinitely many samples, we match what traditional EM does.  In particular,
traditional EM uses the complete distribution, $q_i$, when it is tractible to do
so.  In other words, rather than sampling from $p_{\widehat{\theta}}(\cdot \mid
g(X))$, we use the distribution in the M step to reduce the approximation error.

Using the fully distribution introduces some ugly notion that hides the signal.
It is also more implementation: you have to extend the M-step to compute the
expectation over complete data sets.  The sampling version conveniently uses the
exact same code as fully observed MLE (i.e., a dataset of $\{ x_i \}_{i=1}^n$).

Leveraging the distribution is generally an efficiency win so it is worth doing
if you need the extra accuracy (no sampling error that results from the
approximate E step) and efficiency (no need to take lots of samples).  That
said, the MCEM algorithm is a pretty good algorithm since it is very fast to
implement.
