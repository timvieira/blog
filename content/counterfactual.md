title: Counterfactual reasoning and learning from logged data
date: 2016-12-13
comments: true
tags: counterfactual-reasoning, importance-sampling, machine-learning

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
did *not* observe. Rather than getting into 'philosophy' and discussion of all
the inherent difficulties. I will discuss some basic techniques for learning
from logged partial-feedback data.

For the large part, this post is based on
[Bottou et al. (2013)](https://arxiv.org/abs/1209.2355), which is one my
favorite papers. I'm going to try to synthesize some of the concepts in this
post. Here we go!

**Setup**: We're trying to optimize a function of the form,

$$
J(\theta) = \underset{p_\theta}{\mathbb{E}} \left[ r(x) \right] = \sum_{x \in \mathcal{X}} p_\theta(x) r(x).
$$

<br/> But! We only have a fixes sample of size $m$ from a data collection policy
$q$, $\{ (r^{(j)}, x^{(j)} ) \}_{j=1}^m \overset{\text{i.i.d.}}  \sim q.$ This
setup is sometimes called *off-line off-policy optimization*.

Technicalities: To avoid difficulties later, we'll assume that $q$ assigns
positive probability everwhere, $q(x) > 0$ for all $x \in \mathcal{X}$ and that
we know $q(x^{(j)})$ for all $j$. (I've used the notation $r^{(j)}$ instead of
$r(x^{(j)})$ to emphasize that we can't evaluate $r$ at $x$ values other than
those in the sample.)

**Example**: A good motivating example for this setup is the contextual bandit
problem, where $x$ is a context-action pair, $x = (s,a)$, and the distribution
$q$ is factored as, $q(x) = q(s,a) = q(a|s) p(s)$ and $p$ is factored
$p_{\theta}(x) = p_{\theta}(s,a) = p_{\theta}(a|s) p(s)$. Note that $p$ and
$q$ share the distribution over contexts and only differ in the policy
component. (We can even extend it to a full-blown MDP or POMDP by taking $x$ to
be a sequence of state-action pairs, often called "trajectories".)

**TODO**: One of the main challenges of this setting is that we do not get
paired samples, which means that we have high variance. Consider the contextual
bandit setting. This is the main different with (fully) supervised learning.


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

**Optimizing $J(\theta)$**: After you've collected a sample you can optimze
$\hat{J}_{\!\text{IS}}$ using your favorite deterministic optimization algorithm
(e.g., L-BFGS). Of course, we risk overfitting to the sample if we evaluate
$\hat{J}_{\!\text{IS}}$. The problem is that the objective tends to favor
regions of $\theta$, which are not well represented in the sample (e.g., because
they have high probability under $q$). This is because the importance sampling
estimator has high variance in this regions, thus we want some type of
*regularization* to keep the optimizer in regions which are sufficiently
well-represented by the sample.

Here's a visual example, **TODO**.

**A better surrogate**


**TODO**: Adith's paper calls this "propensity overfitting". Levine and Koltun
also describe this problem in their GPS paper.


Extensions
----------

* **Variance reduction**: Mostly rely on control variates, which are correlated
  random variables which known expectations (or at least they are estimated
  separately). **TODO**: there is a great picture and practical example in
  Bottou's paper, which describes a control variate for clicks vs time-of-day.

* **Imputing missing rewards**: **TODO** use a function approx. Similar to
  control variate, but introduces bias unless the function approximator is
  strongly consistent.

* **Unknown factors**: Consider the contextual bandit setting (mentioned
  above). Here $p$ and $q$ share an *unknown* factor: the distribution of
  contexts. Luckily, we do not need this factor in order to apply any of our
  estimators because they are all based on *likelihood ratios*, thus the factors
  in the importances weights cancel out! Some specific examples are given
  below. These factors do influence the estimators because they are crucial in
  *sampling*, they just aren't necessary in *evaluation*.

    - In contextual bandit example, $x$ is a state-action pair, $w_{\theta}(x) =
      \frac{p_{\theta}(x)}{q(x)} = \frac{ p_{\theta}(s,a) }{ q(s,a) } =
      \frac{p_{\theta}(a|s) p(s)}{q(a|s) p(s)} = \frac{p_{\theta}(a|s) }{ q(a|s)
      }$.

    - In a Markov decision process, $x$ is a sequence of state-action pairs,
      $w_{\theta}(x) = \frac{p_{\theta}(x)}{q(x)} = \frac{ p(s_0) \prod_{t=0}^T
      p(s_{t+1}|s_t,a_t) p_\theta(a_t|s_t) } { p(s_0) \prod_{t=0}^T
      p(s_{t+1}|s_t,a_t) q(a_t|s_t) } = \frac{\prod_{t=0}^T \pi_\theta(a_t|s_t)}
      {\prod_{t=0}^T q(a_t|s_t)}.$


* **Unknown $q$**: Often $q$ is an existing complex system, which does not
  accurately record its probabilities. It is possible to use regression to
  estimate $q$ from the samples. Unfortunately, this will result in a biased
  estimator because we're using a 'ratio of expectations' instead of an
  'expectation of ratios' (but it at least it's statistically consistent). This
  is called the **inverse propensity score** (IPS). In the unknown $q$ setting,
  it's better to use the **doubly-robust estimator** (DR) which combines *two*
  estimators: a density estimator for $q$ and a function approximation for
  $r$. A great explanation for the bandit case can be found in
  [Dudík et al. (2011)](https://arxiv.org/abs/1103.4601). The DR estimator is
  also biased, but it has a better bias-variance tradeoff than IPS.

* **What if $q$ doest have support everywhere?**: This is an especially
  important setting because it is often the case that data collection policies
  abide by some safe regulations, which prevent really bad configurations (i.e.,
  terrible $f(x)$).

  TODO: Maybe the best you can do is not to wander in these areas.

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
