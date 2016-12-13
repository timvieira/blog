title: Counterfactual reasoning and learning from logged incomplete data
date: 2016-12-13
comments: true
tags: math, optimization, rl, counterfactual-reasoning, importance-sampling


<style> .toggle-button { background-color: #555555; border: none; color: white;
padding: 10px 15px; border-radius: 6px; text-align: center; text-decoration:
none; display: inline-block; font-size: 16px; cursor: pointer; } .derivation {
background-color: #f2f2f2; border: thin solid #ddd; padding: 10px;
margin-bottom: 10px; } </style>
<script>
/* workaround for when markdown/mathjax gets confused by the javascript dollar function. */
function toggle(x) { $(x).toggle(); }
</script>


Counterfactual reasoning in a nutshell is about reasoning about data that you
did *not* observe.

Rather than getting into 'philosophy' and discussion of all the inherent
difficulties. I will discuss some basic techniques for machine learning from
logged data. That is, learning from incomplete feedback without additional
interaction with the environment.

For the large part, this post is based on
[Bottou et al. (2013)](https://arxiv.org/abs/1209.2355), which is one of the
best papers I've ever read. It's not an easy read, IMO, which is why I'm going
to try to synthesize some of the concepts in this post.

Here we go!

**Setup**: We're trying to optimize a function of the form,

$$
J(\theta) = \underset{p_\theta}{\mathbb{E}} \left[ r(x) \right] = \sum_{x \in \mathcal{X}} p_\theta(x) r(x).
$$

<br/> But! we only have $m$ samples from a fixed distribution $q$, $\{ (r^{(j)},
x^{(j)} ) \}_{j=1}^m \overset{\text{i.i.d.}} \sim q$. This setup is sometimes
called *off-line off-policy optimization*.

Technicalities: To avoid difficulties later, we'll assume that $q$ assigns
positive probability everwhere, $q(x) > 0$ for all $x \in \mathcal{X}$ and that
we know $q(x^{(j)})$ for all $j$. (I've used the notation $r^{(j)}$ instead of
$r(x^{(j)})$ to emphasize that we can not evaluate $r$ at $x$ values other than
those in the sample.)

**Example**: A good motivating example for this setup is the contextual bandit
problem, where $x$ is a context-action pair, $x = (s,a)$, and the distribution
$q$ is factored as, $q(x) = q(s,a) = q(a|s) p(s)$ and $p$ is factored
$p_{\theta}(x) = p_{\theta}(s,a) = p_{\theta}(a|s) p(s)$. Note that $p$ and
$q$ share the distribution over contexts and only differ in the policy
component. (We can even extend it to a full-blown MDP or POMDP by taking $x$ to
be a sequence of state-action pairs, often called "trajectories".)

**Estimating $J(\theta)$**: We obtain an unbiased estimator of $J(\theta)$ with
[importance sampling](http://timvieira.github.io/blog/post/2014/12/21/importance-sampling/),

$$
J(\theta)
\approx \hat{J}_{\!\text{IS}}(\theta)
= \frac{1}{m} \sum_{j=1}^m r^{(j)} \!\cdot\! w^{(j)}_{\theta}
\quad \text{ where } w^{(j)}_{\theta} = \frac{p_{\theta}(x^{(j)}) }{ q(x^{(j)}) }.
$$

<br/> This estimator is remarkable: it uses importance sampling as a function
approximator! We have an *unbiased* estimate of $J(\theta)$ for any value of
$\theta$ that we like. *The catch* is that we have to pick $\theta$ a priori,
i.e., with no knowledge of the sample. We also require that the usual support
conditions importance sampling conditions ($p_{\theta}(x)>0 \Rightarrow q(x)>0$
for all $x \in \mathcal{X}$).

**Optimizing $J(\theta)$**: If we evaluate the importance sampling estimator
mutliple times, we suffer a **multiple testing problem** $\approx$
**overfitting** to the sample. We will want to do this in order to optimize
$\theta$.


After you've collected a large sample (big $m$) you *could* optimze
$\hat{J}_{\!\text{IS}}$ using your favorite deterministic optimization algorithm
(e.g., L-BFGS). Unless this sample is truley massive, you'll need to be careful
of a few things, which have to do with the fact that $\hat{J}_{\!\text{IS}}$ can
be a terrible approximation to the objective. The problem is that the objective
tends to favor regions of $\theta$, which are not well represented, i.e., places
which do not have high probability under $q$. This is because the importance
sampling estimator has high variance in this regions, thus we want some type of
*regularization* to keep the optimizer in regions which are sufficiently
well-represented by the sample.

Extensions
----------

* **Unknown factors**: Consider the contextual bandit setting (mentioned
  above). Here $p$ and $q$ share an *unknown* factor: the distribution of
  contexts. Luckily, we do not need this factor in order to evaluate any of our
  estimators! This is because they are all based on a *ratio* of probabilities,
  thus the factors in the importances weights cancel out! In contextual bandit
  example, $w_{\theta}(x) = \frac{p_{\theta}(x) }{ q(x) } = \frac{
  p_{\theta}(s,a) }{ q(s,a) } = \frac{p_{\theta}(a|s) p(s) }{ q(a|s) p(s) } =
  \frac{p_{\theta}(a|s) }{ q(a|s) }$. In other words, the unknown factors are
  only needed for *sampling* (not *evaluation*).

* **Unknown $q$**: Often $q$ is an existing complex system, which does not
  accurately record its probabilities. It is possible to use regression to
  estimate $q$ from the samples. Unfortunately, this will result in a biased
  estimator because we're using a 'ratio of expectations' instead of an
  'expectation of ratios' (but it at least it's statistically consistent). This
  is called the **inverse propensity score** (IPS). In the unknown $q$ setting,
  it's better to use the **doubly-robust estimator** (DR) which combines *two*
  estimators: a density estimator for $q$ and a function approximation for
  $r$. A great explanation can be found in
  [Dudík et al. (2011)](https://arxiv.org/abs/1103.4601) The DR estimator is
  also biased, but it has a better bias-variance tradeoff than IPS.

* **What if $q$ is deterministic?**: This is a really hard problem. Essentially,
  this trying to learn without any exploration / experimentation! In general, we
  need exploration to learn. Randomization isn't the only way to perform
  exploration, there are many systematic types of experimentation. It's however,
  more difficult to account for **confounding variables**, which are hidden
  causes that control variation in the data.

    - There are some cases of systematic (non-random) exploration /
      experimentation. We may be able to caracterize these as a type of
      stratified sampling. An extreme example is a contextual bandit where $q$
      assigns contexts to actions deterministically via a hash function. This
      setting is fine because we $q$ as a uniform distribution of actions even
      though it's deterministic.

    - A generic solution might be to apply the doubly-robust estimator, which
      "smooths out" deterministic components (by pretending they are random) and
      accounting for confounds (by explicitly modelling in the density
      estimation, which often requires careful domain knowledge).
