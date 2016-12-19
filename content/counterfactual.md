title: Counterfactual reasoning and learning from logged data
date: 2016-12-19
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


Counterfactual reasoning is *reasoning about data that we did not observe*. For
example, reasoning about the expected reward of new policy given data collected
from a older one.

In this post, I'll discuss some basic techniques for learning from logged
data. For the large part, this post is based on things I learned from
[Peshkin & Shelton (2002)](http://www.cs.ucr.edu/~cshelton/papers/docs/icml02.pdf)
and [Bottou et al. (2013)](https://arxiv.org/abs/1209.2355) (two of all-time
favorite papers).

After reading this post, have a look at the
[Jupyter notebook](https://gist.github.com/timvieira/788c2c25c94663c49abada60f2e107e9)
accompanying this post!

**Setup** (*off-line off-policy optimization*): We're trying to optimize a
function of the form,

$$
J(\theta) = \underset{p_\theta}{\mathbb{E}} \left[ r(x) \right] = \sum_{x \in \mathcal{X}} p_\theta(x) r(x).
$$

<br/> But! We only have a *fixed* sample of size $m$ from a data collection
policy $q$, $\{ (r^{(j)}, x^{(j)} ) \}_{j=1}^m \overset{\text{i.i.d.}} \sim q.$

* Although, it's not *necessarily* the case, you can think of $q = p_{\theta'}$
  for a *fixed* value $\theta'.$

* $\mathcal{X}$ is an arbitrary multivariate space, which permits a mix of
  continuous and discrete components, with appropriate densities, $p_{\theta}$
  and $q$ defined over it.

* $r: \mathcal{X} \mapsto \mathbb{R}$ is a black box that outputs a scalar
  score.

* I've used the notation $r^{(j)}$ instead of $r(x^{(j)})$ to emphasize that we
  can't evaluate $r$ at $x$ values other than those in the sample.

* We'll assume that $q$ assigns positive probability everywhere, $q(x) > 0$ for
  all $x \in \mathcal{X}$. This means is that the data collection process must
  be randomized and eventually sample all possible configurations. Later, I
  discuss relaxing this assumption.


Each distribution is a product of one or more factors of the following types:
**policy factors** (at least one), which directly depend on $\theta$, and
**environment factors** (possibly none), which do not depend directly on
$\theta$. Note that environment factors are *only* accessible via sampling
(i.e., we don't know the *value* they assign to a sample). For example, a
*contextual bandit problem*, where $x$ is a context-action pair, $x =
(s,a)$. Here $q(x) = q(a|s) p(s)$ and $p_{\theta}(x) = p_{\theta}(a|s)
p(s)$. Note that $p_{\theta}$ and $q$ share the environment factor $p(s)$, the
distribution over contexts, and only differ in the action-given-context
factor. For now, assume that we can evaluate all environment factors; later,
I'll discuss how we cleverly work around it.


<!--
(We can
even extend it to a full-blown MDP or POMDP by taking $x$ to be a sequence of
state-action pairs, often called "trajectories".)
-->

**The main challenge** of this setting is that we don't have controlled
experiments to learn from because $q$ is not (completely) in our control. This
manifests itself as high variance ("noise") in estimating $J(\theta)$. Consider
the contextual bandit setting, we receive a context $s$ and execute single
action; we never get to rollback to that precise context and try an alternative
action (to get a paired sample #yolo) because we do not control $p(s)$. This is
an important paradigm for many 'real world' problems, e.g., predicting medical
treatments or ad selection.

<!--
This is the crucial difference that makes counterfactual learning
more difficult than (fully) supervised learning.
-->

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
i.e., with no knowledge of the sample.

<!--
We also require that the usual 'support
conditions' for importance sampling conditions ($p_{\theta}(x)>0 \Rightarrow
q(x)>0$ for all $x \in \mathcal{X}$, which is why we made assumption A1.
-->

After we've collected a (large) sample it's possible to optimize
$\hat{J}_{\!\text{IS}}$ using any optimization algorithm (e.g., L-BFGS). Of
course, we risk overfitting to the sample if we evaluate
$\hat{J}_{\!\text{IS}}$. Actually, it's a bit worse: this objective tends to
favor regions of $\theta$, which are not well represented in the sample because
the importance sampling estimator has high variance in these regions resulting
from large importance weights (when $q(x)$ is small and $p_{\theta}(x)$ is
large, $w(x)$ is large and consequently so is $\hat{J}_{\text{IS}}$ regardless of
whether $r(x)$ is high!). Thus, we want some type of "regularization" to keep
the optimizer in regions which are sufficiently well-represented by the sample.

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
favorites is Bottou et al.'s lower bound on $J(\theta)$, which we get by
clipping importance weights, replace $w^{(j)}_{\theta}$ with $\min(R,
w^{(j)}_{\theta})$.

**Confidence intervals** [V3]: We can augment the V2 lower bound with confidence
intervals derived from the empirical Bernstein bound (EBB). We'll require that
$r$ is bounded and that we know its max/min values. The EBB *penalizes*
hypotheses (values of $\theta$) which have higher sample variance. (Note: a
Hoeffding bound wouldn't change the *shape* of the objective, but EBB does
thanks to the sample variance penalty. EBB tends to be tighter.). The EBB
introduces an additional "confidence" hyperparameter, $(1-\delta)$. Bottou et
al. recommend maximizing the lower bound as it provides safer improvements. See
the original paper for the derivation.

<!--
An important benefit of having upper *and* lower is that the bounds tell
us whether or not we should collect more data
-->

Both V2 and V3 are *biased* (as they are lower bounds), but we can mitigate the
bias by *tuning* the hyperparameter $R$ on a heldout sample (we can even tune
$\delta$, if desired). Additionally, V2 and V3 are 'valid' when $q$ has limited
support since they prevent the importance weights from exploding (of course, the
bias can be arbitrarily bad, but probably unavoidable given the
learning-from-only-logged data setup).

Extensions
----------

<!--
**Be warned**: This may be considered an idealized setting. Much of the research
in counterfactual and causal reasoning targets (often subtle) deviations from
these assumptions (and some different questions, of course). Some extensions and
discussion appear towards the end of the post.
-->

**Unknown environment factors**: Consider the contextual bandit setting
(mentioned above). Here $p$ and $q$ share an *unknown* environment factor: the
distribution of contexts. Luckily, we do not need to know the value of this
factor in order to apply any of our estimators because they are all based on
likelihood *ratios*, thus the shared unknown factors cancel out!  Some specific
examples are given below. Of course, these factors do influence the estimators
because they are crucial in *sampling*, they just aren't necessary in
*evaluation*.

  - In contextual bandit example, $x$ is a state-action pair, $w_{\theta}(x) =
    \frac{p_{\theta}(x)}{q(x)} = \frac{ p_{\theta}(s,a) }{ q(s,a) } =
    \frac{p_{\theta}(a|s) p(s)}{q(a|s) p(s)} = \frac{p_{\theta}(a|s) }{ q(a|s)
    }$.

  - In a Markov decision process, $x$ is a sequence of state-action pairs,
    $w_{\theta}(x) = \frac{p_{\theta}(x)}{q(x)} = \frac{ p(s_0) \prod_{t=0}^T
    p(s_{t+1}|s_t,a_t) p_\theta(a_t|s_t) } { p(s_0) \prod_{t=0}^T
    p(s_{t+1}|s_t,a_t) q(a_t|s_t) } = \frac{\prod_{t=0}^T \pi_\theta(a_t|s_t)}
    {\prod_{t=0}^T q(a_t|s_t)}.$

**Variance reduction**: These estimators can all be improved with variance
reduction techniques. Probably the most effective technique is using
[control variates](https://en.wikipedia.org/wiki/Control_variates) (of which
baseline functions are a special case). These are random variables correlated
with $r(x)$ for which we know their expectations (or at least they are estimated
separately). A great example is how ad clicks depend strongly on time-of-day
(fewer people are online late at night so we get fewer clicks), thus the
time-of-day covariate explains a large part of the variation in $r(x)$.

**Estimation instead of optimization**: You can use this general setup for
estimation instead of optimization, in which case it's fine to let $r$ have
real-valued multivariate output. The confidence intervals are probably useful in
that setting too.

**Unknown $q$**: Often $q$ is an existing complex system, which does not record
its probabilities. It is possible to use regression to estimate $q$ from the
samples, which is called the **propensity score** (PS). PS attempts to account
for **confounding variables**, which are hidden causes that control variation in
the data. Failing to account for confounding variables may lead to
[incorrect conclusions](https://en.wikipedia.org/wiki/Simpson's_paradox). Unfortunately,
PS results in a biased estimator because we're using a 'ratio of expectations'
(we'll divide by the PS estimate) instead of an 'expectation of ratios'. PS is
only statistically consistent in the (unlikely) event that the density estimate
is correctly specified (i.e., we can eventually get $q$ correct). In the unknown
$q$ setting, it's often better to use the **doubly-robust estimator** (DR) which
combines *two* estimators: a density estimator for $q$ and a function
approximation for $r$. A great explanation for the bandit case is in
[Dudík et al. (2011)](https://arxiv.org/abs/1103.4601). The DR estimator is also
biased, but it has a better bias-variance tradeoff than PS.

**What if $q$ doesn't have support everywhere?** This is an especially important
setting because it is often the case that data collection policies abide by some
**safety regulations**, which prevent known bad configurations. In many
situations, evaluating $r(x)$ corresponds to executing an action $x$ in the real
world so terrible outcomes could occur, such as, breaking a system, giving a
patient a bad treatment, or losing money. V1 is ok to use as long as we satisfy
the importance sampling support conditions, which might mean rejecting certain
values for $\theta$ (might be non-trivial to enforce) and consequently finding a
less-optimal policy. V2 and V3 are ok to use without an explicit constraint, but
additional care may be needed to ensure specific safety constraints are
satisfied by the learned policy.

**What if $q$ is deterministic?** This is related to the point above. This is a
hard problem. Essentially, this trying to learn without any exploration /
experimentation! In general, we need exploration to learn. Randomization isn't
the only way to perform exploration, there are many systematic types of
experimentation.

  - There are some cases of systematic experimentation that are ok. For example,
    enumerating all elements of $\mathcal{X}$ (almost certainly
    infeasible). Another example is a contextual bandit where $q$ assigns
    actions to contexts deterministically via a hash function (this setting is
    fine because $q$ is essentially a uniform distribution over actions, which
    is independent of the state). In other special cases, we *may* be able to
    characterize systematic exploration as
    [stratified sampling](https://en.wikipedia.org/wiki/Stratified_sampling).

  - A generic solution might be to apply the doubly-robust estimator, which
    "smooths out" deterministic components (by pretending they are random) and
    accounting for confounds (by explicitly modeling them in the propensity
    score).

**What if we control data collection ($q$)?** This is an interesting
setting. Essentially, it asks "how do we explore/experiment optimally (and
safely)?". In general, this is an open question and depends on many
considerations, such as, how much control, exploration cost (safety constraints)
and prior knowledge (of $r$ and unknown factors in the environment). I've seen
some papers cleverly design $q$. The first that comes to mind is
[Levine & Koltun (2013)](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf). Another
setting is *online* contextual bandits, in which algorithms like
[EXP4](http://jmlr.org/proceedings/papers/v15/beygelzimer11a/beygelzimer11a.pdf)
and
[Thompson sampling](http://www.research.rutgers.edu/~lihong/pub/Chapelle12Empirical.pdf),
prescribe certain types of exploration and work interactively (i.e., they don't
have a fixed training sample). Lastly, I'll mention that there are many
techniques for variance reduction by importance sampling, which may apply.


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

Covers confidence intervals for policy evaluation similar to Bottou et al., as
well as learning algorithms for RL with safety guarantees (e.g., so we don't
break the robot).

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
