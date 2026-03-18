title: The Simple Idea Behind EM
date: 2019-11-20
comments: true
status: draft
tags: machine-learning, statistics, rant

$$
\newcommand{\ind}[1]{\boldsymbol{1}\left[ #1 \right]}
\newcommand{\defeq}[0]{\overset{\scriptsize\text{def}}{=}}
$$

The expectation maximization algorithm (EM) is simple, but poorly
explained in my opinion:

 * Explanations are littered with distracting model-specific details and messy
   variational inference notation that make it hard to see the signal in the
   noise.[ref]Many textbook presentations frame EM as iterative lower-bound
     maximization, e.g., following the variational view of
     [Neal & Hinton (1998)](https://link.springer.com/chapter/10.1007/978-94-011-5014-9_12).
     That perspective is valuable&mdash;it's the basis for most convergence
     proofs&mdash;but it front-loads a lot of variational machinery before the
     reader has any intuition for what the algorithm is actually doing.[/ref]

 * In this post, I'll build a bridge between EM and MLE that I think makes
   the algorithm much easier to understand.  No KL divergences, no lower
   bounds, no convergence proofs.

EM is an algorithm for maximum-likelihood estimation when we have "incomplete
observations."  There is an underlying i.i.d. process $\{ X_i \}_{i=1}^n$, which
we do not get to fully observe.  Instead, for each $i$ we observe only that some
constraint holds: $g(X_i) = 1$, where $g\colon \mathrm{domain}(X) \to \{0,1\}$
is a known function.  The constraint tells us something about $X_i$ without
pinning it down exactly.


Consider the following examples:

1. Complete observation: $g(X_i) = \ind{X_i = x_i}$ where $\ind{\cdot}$ is the
   indicator function.

2. Incomplete, interval observation: $g(X_i) = \ind{a_i \le X_i < b_i}$ for some
   constants $a_i < b_i$.  We recover the complete case as $a_i$ approaches $b_i$
   for all $i$.[ref]Interval observations are used in estimation for censored
     observations.[/ref]

3. Incomplete, subset observations: we could observe non-empty subsets,
   $\mathcal{X}_i$, of $X$'s domain, $\ind{X_i \in \mathcal{X}_i}$.

4. Incomplete, function observation: $g(X_i) = \ind{y_i = f(X_i)}$ for a function $f$ that is not necessarily invertible.


There are many possible forms of incomplete observation (i.e., families of $g$
functions), and each brings its own implementation details.  For example, to
handle intervals we
can use the cumulative distribution function, $F(\cdot; \theta)$, $p(a_i \le X <
b_i) = F(b_i; \theta) \cdot (1 - F(a_i; \theta))$.[ref]To support $a_i = b_i$,
you swap in the pdf via a piecewise function when $a_i = b_i$ happens to be observed.[/ref]


The obvious thing to optimize is the **incomplete log-likelihood**
$$
\mathcal{L}(\theta) \defeq \sum_{i=1}^n \log p_\theta( g(X_i) ) = \sum_{i=1}^n
\log \sum_{x \in \mathcal{X} } p_\theta(x) g(x)
$$

We can often optimize $\mathcal{L}$ directly with gradient-based methods.
However, $\mathcal{L}$ is generally nonconvex, so gradient methods only find
local optima.  EM is an alternative optimization strategy with its own
convergence guarantees.

EM is based on a chicken-and-egg type of story:

 1. If we had the complete data, we could estimate $\theta$ via MLE.

 2. If we had $\theta$, we could complete the data using the model's
    distribution over the missing values.

That's basically what EM does.  Start with an initial guess of
$\theta$,[ref]Alternatively, one can initialize with a guess of the data
completions and reorder the two steps below "M-E" rather than
"E-M."[/ref] then alternate:

 * **E step:** Use the current model $p_\theta$ to "complete" the data in a manner
   consistent with the observations.  That is, form a distribution over what
   the complete data could have been, given what we observed.

 * **M step:** Re-fit the model to that completed data.

Repeat until convergence.


## Monte Carlo EM

Consider a slightly dumbed-down version of EM, called Monte Carlo EM (MCEM).

MCEM iteratively fills in the missing data by sampling completions from the
current model $\theta$ (sometimes called bootstrapping).

- **E step:** Sample a completion for each $i$:
  $\widehat{x}_i \sim p_{\theta}(\cdot \mid g(X_i) = 1)$.
  This gives us a "complete" dataset that satisfies the observed constraints.

- **M step:** Compute the MLE of $\theta$ on the completed dataset
  $\{ \widehat{x}_i \}_{i=1}^n$.

Note that sampling from $p_{\theta}(\cdot \mid g(X_i) = 1)$ is not
necessarily easy&mdash;it requires sampling from the posterior under the current
model, which is often the hard part of EM regardless of the variant.  The point
of MCEM is not that it is easier to *implement*, but that it is easier to *think
about*: once you have the samples, the M step is just ordinary MLE on a complete
dataset&mdash;exactly the same code you'd write if you had observed the data
directly.  This cleanly separates the conceptual story (complete the data, then
fit) from the mathematical machinery.

## From MCEM to EM

We could take more than one sample per $i$ in the E step.  If we took
infinitely many samples, we would recover what traditional EM does.  In particular,
traditional EM uses the complete distribution, $q_i$, when it is tractable to do
so.  In other words, rather than sampling from $p_{\theta}(\cdot \mid
g(X_i) = 1)$, we use the distribution directly in the M step to reduce the approximation error.

Using the full distribution introduces some ugly notation that hides the signal.
It also requires extending the M step to compute expectations over complete
datasets, rather than just fitting a single dataset.  Leveraging the distribution
is generally an efficiency win (no sampling error, no need for many samples), so
it is worth doing when tractable.  But the conceptual content is the same: EM is
just MCEM with sampling replaced by exact expectations.

## Takeaway

EM is often presented as a variational lower-bound algorithm, which is useful for
proving convergence but obscures the core idea.  The core idea is simple: if you
knew the complete data you'd just do MLE; if you knew the parameters you could
fill in the missing data.  EM alternates between these two steps.  MCEM makes
this especially transparent&mdash;not because sampling is easy, but because
thinking in terms of "sample then fit" strips away the notational overhead and
reveals the simple structure underneath.