title: Hot Take: Expectation Maximization
date: 2019-11-20
comments: true
status: draft
tags: machine-learning, statistics, hot-take

$$
\newcommand{\ind}[1]{\boldsymbol{1}\left[ #1 \right]}
\newcommand{\defeq}[0]{\overset{\scriptsize\text{def}}{=}}
$$

The expectation maximization algorithm (EM) is a simple algorithm that is poorly
explained in my opinion:

 * Explanations are littered with distracting model-specific details and messy
   variational inference notation that make it hard to see the signal in the
   noise.[ref]Why such a messy presentation?  My guess is that it is
     because the theoretical foundations for EM are based on an iterative
     lower-bound maximization algorithm.  The theory is bleeding into the
     presentation.  This view is useful because they tie EM to a theoretical
     optimization foundation that is used to prove convergence properties, but
     at the end of the day they are just some dude's hot take.[/ref]

 * Today, I will give my hot take.  I will create a bridge between EM and MLE,
   which I believe makes it easier to see the signal in the noise.

   <!--
   There are no
   KL divergences, lower bounds, or convergence proofs to distract from what's
   going on.
   -->

EM is an algorithm for maximum-likelihood estimation when we have "incomplete
observations."  What that means is that there is an underlying i.i.d. process
$\{ X_i \}_{i=1}^n$, which we do not get to fully observe.  Instead, we see an
**incomplete observation**, which are (generally speaking) *constraints* on what
$X_i$ might have been if only we had actually observed it.  In other words, we
do not observe $X_i$, we observe the event $\{g(X_i) = 1\}$.  We assume knowledge of
the function $g$, and that $g(x) \in \{0, 1\}$ for all $x \in
\mathrm{domain}(X)$.


Consider the following examples:

1. Complete observation: $g(X_i) = \ind{X_i = x_i}$ where $\ind{\cdot}$ is the
   indicator function.

2. Incomplete, interval observation: $g(X_i) = \ind{a_i \le X_i < b_i}$ for some
   constants $a_i < b_i$.  We recover the complete case as $a_i$ approaches $b_i$
   for all $i$.[ref]Interval observations are used in estimation for censored
     observations.[/ref]

3. Incomplete, subset observations: we could observe non-empty subsets,
   $\mathcal{X}_i$, of $X$'s domain, $\ind{X_i \in \mathcal{X}_i}$.

4. $\ind{y_i = f(X_i)}$ for a function $f$ that is not necessarily invertible.


There are tons of weird incomplete observation types (i.e., families of $g$
functions).  Every choice will result in some set of details that you will have
to work out in order to accommodate.  For example, to accommodate intervals we
can use the cumulative distribution function, $F(\cdot; \theta)$, $p(a_i \le X <
b_i) = F(b_i; \theta) \cdot (1 - F(a_i; \theta))$.[ref]To support $a_i = b_i$,
you swap in the pdf via a piecewise function when $a_i = b_i$ happens to be observed.[/ref]


The obvious thing to optimize is the **incomplete log-likelihood**
$$
\mathcal{L}(\theta) \defeq \sum_{i=1}^n \log p_\theta( g(X_i) ) = \sum_{i=1}^n
\log \sum_{x \in \mathcal{X} } p_\theta(x) g(x)
$$

We can often optimize $\mathcal{L}$ directly with gradient-based optimization
methods.  However, $\mathcal{L}$ is generally nonconvex so we will only get
a locally optimal approximation.  The EM algorithm is a different type of search
algorithm which has different convergence guarantees.

EM is based on a chicken-and-egg type of story:

 1. If we had the complete data observations, we'd get $\theta$ via MLE.

 2. If we had $\theta$, we could complete the data by sampling from the
    model (or, more generally, using the model's distribution over it).

That's basically what EM is going to do:

 * **Initialize:** guess model parameters $\theta$.  [ref]Alternatively, one can
   initialize with a guess of the data completions and reorder the two steps
   below "M-E" rather than "E-M."[/ref]

 * **Repeat** until some convergence criterion is met:

     - **E step:** Use the current model to "complete" the data in a manner
       consistent with the observations.  That is, form a distribution over what
       the complete data could have been, given what we observed.

     - **M step:** Re-fit the model to that completed data.


## Monte Carlo EM

Consider a slightly dumbed-down version of EM, called Monte Carlo EM (MCEM).

MCEM just (iteratively) fills in the missing information about the random
variables by sampling what the complete values were based on the current estimate
of the model $\widehat{\theta}$ (sometimes called bootstrapping).

- **E step:** For each $i$, sample a completion consistent with the
  observation: $\widehat{x}_i \sim p_{\widehat{\theta}}(\cdot \mid g(X_i) = 1)$.
  This gives us a "complete" dataset that satisfies the observed constraints.

- **M step:** Compute the fully observed maximum-likelihood estimate of $\theta$ on the
  approximate dataset $\{ \widehat{x}_i \}_{i=1}^n$.

The beauty of MCEM is that the M step is just ordinary MLE on a complete
dataset&mdash;exactly the same code you'd write if you had observed the data
directly.

## From MCEM to EM

We could take more than one sample per $i$ in the E step as
long as we take an equal number per example (in expectation).  If we took
infinitely many samples, we match what traditional EM does.  In particular,
traditional EM uses the complete distribution, $q_i$, when it is tractable to do
so.  In other words, rather than sampling from $p_{\widehat{\theta}}(\cdot \mid
g(X_i) = 1)$, we use the distribution directly in the M step to reduce the approximation error.

Using the full distribution introduces some ugly notation that hides the signal.
It is also harder to implement: you have to extend the M step to compute the
expectation over complete datasets.  The sampling version conveniently uses the
exact same code as fully observed MLE (i.e., a dataset of $\{ x_i \}_{i=1}^n$).

Leveraging the distribution is generally an efficiency win, so it is worth doing
if you need the extra accuracy (no sampling error that results from the
approximate E step) and efficiency (no need to take lots of samples).  That
said, the MCEM algorithm is a pretty good algorithm since it is very fast to
implement.


## Takeaway

EM is often presented as a variational lower-bound algorithm, which is useful for
proving convergence but obscures the core idea.  The core idea is simple: if you
knew the complete data you'd just do MLE; if you knew the parameters you could
fill in the missing data.  EM alternates between these two steps.  MCEM makes
this especially transparent&mdash;the E step is sampling, the M step is ordinary
MLE&mdash;and traditional EM is just MCEM with the sampling replaced by exact
expectations.
