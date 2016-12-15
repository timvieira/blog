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


Counterfactual reasoning is about *reasoning about data that you did not
observe*. For example, reasoning about the expected reward of new policy given
data collected from a older one.

Rather than getting into 'philosophy' and discussion of all the inherent
difficulties, I will discuss some basic techniques for learning from logged
partial-feedback data. For the large part, this post is based on things I
learned from [Bottou et al. (2013)](https://arxiv.org/abs/1209.2355) (one my
favorite papers!).

**Setup**: We're trying to optimize a function of the form,

$$
J(\theta) = \underset{p_\theta}{\mathbb{E}} \left[ r(x) \right] = \sum_{x \in \mathcal{X}} p_\theta(x) r(x).
$$

<br/> But! We only have a fixes sample of size $m$ from a data collection policy
$q$, $\{ (r^{(j)}, x^{(j)} ) \}_{j=1}^m \overset{\text{i.i.d.}}  \sim q.$

Technicalities: To avoid difficulties later, we'll assume that $q$ assigns
positive probability everwhere, $q(x) > 0$ for all $x \in \mathcal{X}$. This
means is that the data collection process (sample from $q$ and evaluate $r$,
repeat $m$ times) must be randomized and eventually any possible
configurations. Later, I discuss relaxing this assumption (see 'extensions').

<!--
I'll also assume that we
know $q(x^{(j)})$ for all $j$, or at least the factors in $p_{\theta}$ that
directly depend on $\theta$ (more on this later).
-->

I've used the notation $r^{(j)}$ instead of $r(x^{(j)})$ to emphasize that we
can't evaluate $r$ at $x$ values other than those in the sample. This setup is
sometimes called *off-line off-policy optimization*.

**Example**: A good motivating example for this setup is the contextual bandit
problem, where $x$ is a context-action pair, $x = (s,a)$, and the distribution
$q$ is factored as, $q(x) = q(s,a) = q(a|s) p(s)$ and $p$ is factored
$p_{\theta}(x) = p_{\theta}(s,a) = p_{\theta}(a|s) p(s)$. Note that $p$ and $q$
share the distribution over contexts and only differ in the policy component
(the distribution of over actions given the current state). (We can even extend
it to a full-blown MDP or POMDP by taking $x$ to be a sequence of state-action
pairs, often called "trajectories".)

The main challenges of this setting is *we do not get paired samples* (i.e., we
have incomplete feedback from $r(x)$). This manifests itself in our estimator as
high variance. This is the crucial difference that makes counterfactual learning
more difficult than (fully) supervised learning. Consider the contextual bandit
setting, here we sample a state and get to try a single action, which means we
never get to rollback to that precise state and try an alternative action. This
is an important paradigm for real tasks, such as, medical treatment or ad
placement. In these settings, we never get to turn back time (#tbt) and try
something else #yolo.

**Estimating $J(\theta)$** [V1]: We obtain an unbiased estimator of $J(\theta)$
with
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
i.e., with no knowledge of the sample. We also require that the usual 'support
conditions' for importance sampling conditions ($p_{\theta}(x)>0 \Rightarrow
q(x)>0$ for all $x \in \mathcal{X}$).

**Optimizing $J(\theta)$** [V1]: After you've collected a sample you can optimze
$\hat{J}_{\!\text{IS}}$ using your favorite deterministic optimization algorithm
(e.g., L-BFGS). Of course, we risk overfitting to the sample if we evaluate
$\hat{J}_{\!\text{IS}}$. The problem is that the objective tends to favor
regions of $\theta$, which are not well represented in the sample (e.g., because
they have high probability under $q$). This is because the importance sampling
estimator has high variance in these regions, thus we want some type of
"regularization" to keep the optimizer in regions which are sufficiently
well-represented by the sample.

<!--
**Visual example**: We can visualize this phenomena in a simple example. Let $q
= \mathcal{N}(0, \sigma=5)$, $r(x) = 1 \text{ if } x \in [2, 3], 0.2 \text{
otherwise},$ and $p_\theta = \mathcal{N}(\theta, \sigma=1)$.  This example is
nice because it let's us plot $x$ and $\theta$ in the same space. This is
generally not the case, because $\mathcal{X}$ may have no connection to $\theta$
space, e.g., $\mathcal{X}$ may be discrete.

**TODO** add plot
-->

**Better surrogate** [V2]: There are many ways to improve the variance of the
estimator and *confidently* obtain improvements to the system. One of my
favorites is Bottou et al.'s lower bound on $J$, which we get by clipping
weights, replace $w^{(j)}_{\theta}$ with $\min(R, w^{(j)}_{\theta})$.

**Confidence intervals** [V3]: You can augment the V2 lower bound with (upper
and lower bounding) confidence intervals derived from the empirical Bernstein
bound (EBB). The EBB is a nice method for obtaining a high-probability bound,
which is tighter than those derived by Hoeffding because it accounts for sample
variance. Essentially, the EBB lower bound *penalizes* hypotheses (values of
$\theta$) which have higher sample variance. (Note: a Hoeffding bound wouldn't
change the shape of the objective, but EBB does thanks to the sample variance
penalty.). This version of the bound introduces an additional a "confidence"
hyperparameter $(1-\delta)$. See the original paper for the derivation of these
bounds.

<!--
An important benefit of having upper *and* lower is that the bounds tell
us whether or not we should collect more data
-->

Both V2 and V3 are biased, but we can mitigate the bias by *tuning* the
hyperparameter $R$ on a heldout sample (you can even tune $\delta$, if you
like). Both estimators is also valid when $q$ has limited support since it
prevents importance weights from exploding (of course, the bias can be
arbitrarily bad, but unavoidable given the learning-from-only-logged data
setup).

Extensions
----------

<!--
**Be warned**: This may be considered an idealized setting. Much of the research
in counterfactual and causal reasoning targets (often subtle) deviations from
these assumptions (and some different questions, of course). Some extensions and
discussion appear towards the end of the post.
-->

**Unknown factors**: Consider the contextual bandit setting (mentioned
above). Here $p$ and $q$ share an *unknown* factor: the distribution of
contexts. Luckily, we do not need to know the value of this factor in order to
apply any of our estimators because they are all based on *likelihood ratios*,
thus the factors in the importances weights cancel out! Some specific examples
are given below. These factors do influence the estimators because they are
crucial in *sampling*, they just aren't necessary in *evaluation*.

  - In contextual bandit example, $x$ is a state-action pair, $w_{\theta}(x) =
    \frac{p_{\theta}(x)}{q(x)} = \frac{ p_{\theta}(s,a) }{ q(s,a) } =
    \frac{p_{\theta}(a|s) p(s)}{q(a|s) p(s)} = \frac{p_{\theta}(a|s) }{ q(a|s)
    }$.

  - In a Markov decision process, $x$ is a sequence of state-action pairs,
    $w_{\theta}(x) = \frac{p_{\theta}(x)}{q(x)} = \frac{ p(s_0) \prod_{t=0}^T
    p(s_{t+1}|s_t,a_t) p_\theta(a_t|s_t) } { p(s_0) \prod_{t=0}^T
    p(s_{t+1}|s_t,a_t) q(a_t|s_t) } = \frac{\prod_{t=0}^T \pi_\theta(a_t|s_t)}
    {\prod_{t=0}^T q(a_t|s_t)}.$

**Variance reduction**: Mostly rely on
[control variates](https://en.wikipedia.org/wiki/Control_variates) of which
*baseline functions* are a special case. These other variables (covariates) that
are correlated with $r(x)$ for which we know their expectations (or at least
they are estimated separately). A great example is how ad click depend strongly
on time-of-day (fewer people are online late at night so you get fewer clicks),
thus the time-of-day covariate explains a large part of the variation in $r(x)$.

**Unknown $q$**: Often $q$ is an existing complex system, which does not
accurately record its probabilities (if at all). It is possible to use
regression to estimate $q$ from the samples. Unfortunately, this will result in
a biased estimator because we're using a 'ratio of expectations' instead of an
'expectation of ratios' (but it at least it's statistically consistent). This is
called the **inverse propensity score** (IPS). In the unknown $q$ setting, it's
better to use the **doubly-robust estimator** (DR) which combines *two*
estimators: a density estimator for $q$ and a function approximation for $r$. A
great explanation for the bandit case is in
[Dudík et al. (2011)](https://arxiv.org/abs/1103.4601). The DR estimator is also
biased, but it has a better bias-variance tradeoff than IPS.

**What if $q$ doesn't have support everywhere?** This is an especially important
setting because it is often the case that data collection policies abide by some
**safety regulations**, which prevent bad configurations. In many situations,
evaluating $r(x)$ corresponds to execution a policy ($q$) in the real world and
as terrible outcomes could occur, such as, breaking a system, giving a patient a
bad treatment, or losing money. V1 is ok to use as long as we satisfy the
importance sampling support conditions, which might mean rejecting certain
values for $\theta$ and consequently finding a less-optimal policy (this might
be non-trival to enforce). V2 and V3 are ok to use without an explicit
constraint, but additional care may be needed to ensure specific safety
constraints are satisfied (if that's desired).

**What if $q$ is deterministic?** This is related to the point above. This is a
really hard problem. Essentially, this trying to learn without any exploration /
experimentation! In general, we need exploration to learn. Randomization isn't
the only way to perform exploration, there are many systematic types of
experimentation. It's however, more difficult to account for **confounding
variables**, which are hidden causes that control variation in the data. Failing
to account for confounding variables may lead to
[incorrect conclusions](https://en.wikipedia.org/wiki/Simpson's_paradox).

  - There are some cases of systematic (non-random) exploration /
    experimentation. For example, enumerating all elements of $\mathcal{X}$,
    which almost certainly infeasible. An extreme example is a contextual bandit
    where $q$ assigns contexts to actions deterministically via a hash
    function. This setting is fine because we $q$ as a uniform distribution of
    actions even though it's deterministic. In other special cases, we may be
    able to caracterize systematic exploration as *stratified sampling*.

  - A generic solution might be to apply the doubly-robust estimator, which
    "smooths out" deterministic components (by pretending they are random) and
    accounting for confounds (by explicitly modeling them in the inverse
    propensity estimate, which often requires careful domain knowledge).

**What if we control data collection ($q$)?** This is an interesting
setting. Essentially, it asks "how do we explore/experiment optimally (and
safely)?". In general, this is an open question and depends on many
considerations, such as, exploration cost (safety constraints) and prior
knowledge (of $r$ and unknown factors in the environment). I've seen some papers
cleverly design $q$. The first that comes to mind is
[Levine & Koltun (2013)](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf). Another
setting is *online* contextual bandits, in which case there is
[EXP4](http://jmlr.org/proceedings/papers/v15/beygelzimer11a/beygelzimer11a.pdf). There's
a fair amount of work in choosing importance sampling proposal distributions to
reduce variance, which may apply.


Further reading
---------------

> Léon Bottou, Jonas Peters, Joaquin Quiñonero-Candela, Denis X. Charles, D. Max
> Chickering, Elon Portugaly, Dipankar Ray, Patrice Simard, Ed Snelson.
> [Counterfactual reasoning in learning systems](https://arxiv.org/abs/1209.2355).
> JMLR 2013.

The source for the majority of this post. It includes many other interesting
ideas and goes more in depth into some of the details.

> Miroslav Dudík, John Langford, Lihong Li.
> [Doubly robust policy evaluation and learning](https://arxiv.org/abs/1103.4601).
> ICML 2011.

Discussed in extensions section.

> Philip S. Thomas.
> [Safe reinforcement learning](http://psthomas.com/papers/Thomas2015c.pdf).
> PhD Thesis 2015.

Covers many of the same policy evalation via importance sampling bounds as
Bottou et al., but also covers learning algorithms for RL with safety guarantees
(e.g., so you don't break the robot).

> Peshkin and Shelton.
> [Learning from scarce experience](http://www.cs.ucr.edu/~cshelton/papers/docs/icml02.pdf).
> ICML 2002.

An older RL paper, which covers learning from logged data. This is one of the
earliest papers on learning from logged data that I could find.

> Levine and Koltun.
> [Guided policy search](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf).
> ICML 2013.

Discusses clever choices for $q$ to better-guide learning in the RL setting.

> Corinna Cortes, Yishay Mansour, Mehryar Mohri.
> [Learning bounds for importance weighting](https://papers.nips.cc/paper/4156-learning-bounds-for-importance-weighting.pdf).
> NIPS 2010.

Discusses *generalization bounds* for the counterfactual objective. Includes an
alternative weighting scheme to keep importance weights from exploding.
