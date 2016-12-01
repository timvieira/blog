title: Likelihood-ratio gradient
date: 2016-06-09
comments: true
status: draft
tags: math, optimization, rl, policy-gradient


**Setup**: We're trying to optimize a function of the form

$$
J(\theta) = \underset{p_\theta}{\mathbb{E}} \left[ r(x) \right] = \sum_{x \in \mathcal{X}} p_\theta(x) r(x).
$$

The problem is that it's too costly to evaluate $r(x)$ for all $\mathcal{X}$ and
that $p$ times $r$ has no structure which we can exploit (e.g., for dynamic
programming). We also do not assume full knowledge of $r(x)$; sometimes called
"bandit feedback."

Suppose we can sample $x^{(j)} \sim p_\theta$. This opens up the following
(unbiased) Monte Carlo estimators for $J$ and its gradient,

$$
J(\theta) \approx \frac{1}{m} \sum_{j=1}^m r(x^{(j)})
$$

$$
\nabla_{\!\theta} J(\theta) \approx \frac{1}{m} \sum_{j=1}^m r(x^{(j)}) \nabla_{\!\theta} \log p_{\theta}(x^{(j)}).
$$


I'm going to work with a more general version based on
[importance sampling](http://timvieira.github.io/blog/post/2014/12/21/importance-sampling/)
because it will give us some interesting freedom later on. Here $x^{(j)} \sim q$
instead of $p_\theta$. Note that we need the following condition on $q$ to hold
for all $x$, $p(x) > 0 \Rightarrow q(x) > 0$.

$$
J(\theta) \approx \frac{1}{m} \sum_{j=1}^m w^{(j)}_{\theta} r(x^{(j)})
$$

$$
\nabla_{\!\theta} J(\theta) \approx \frac{1}{m} \sum_{j=1}^m w^{(j)}_{\theta} r(x^{(j)}) \nabla_{\!\theta} \log p_{\theta}(x^{(j)}).
$$

where $w^{(j)}_{\theta} = p_{\theta}(x^{(j)}) / q(x^{(j)})$


<style>
.toggle-button {
    background-color: #555555;
    border: none;
    color: white;
    padding: 10px 15px;
    border-radius: 6px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    cursor: pointer;
}
.derivation {
  background-color: #f2f2f2;
  border: thin solid #ddd;
  padding: 10px;
  margin-bottom: 10px;
}
</style>

<script>
// workaround for when markdown/mathjax gets confused by the
// javascript dollar function.
function toggle(x) { $(x).toggle(); }
</script>

<button class="toggle-button" onclick="toggle('#likelihood-ratio-derivation');">Derivation</button>
<div id="likelihood-ratio-derivation" class="derivation">
The derivation is pretty simple
$$
\begin{eqnarray}
  \nabla_{\!\theta} \, \underset{p_\theta}{\mathbb{E}}\left[ r(x) \right]
  &=& \nabla_{\!\theta} \left[ \sum_x p_{\theta}(x) r(x) \right] \\
  &=& \sum_x \nabla_{\!\theta} \left[ p_{\theta}(x) \right] r(x) \\
  &=& \sum_x p_{\theta}(x) \frac{\nabla_{\!\theta} \left[ p_{\theta}(x) \right] }{ p_{\theta}(x) } r(x) \\
  &=& \underset{p}{\mathbb{E}}\left[ r(x) \nabla_{\!\theta} \log p_\theta(x) \right]
\end{eqnarray}
$$

<br/>
We use the identity $\nabla g = g\, \nabla \log g$, assuming $g > 0$.

Importance-weighted version:
$$
\begin{eqnarray}
  \nabla_{\!\theta} \, \underset{p_\theta}{\mathbb{E}}\left[ r(x) \right]
  &=& \underset{p}{\mathbb{E}}\left[ r(x) \nabla_{\!\theta} \log p_\theta(x) \right] \\
  &=& \sum_x p_{\theta}(x) r(x) \nabla_{\!\theta} \log p_\theta(x) \\
  &=& \sum_x \frac{q(x)}{q(x)} p_{\theta}(x) r(x) \nabla_{\!\theta} \log p_\theta(x) \\
  &=& \underset{q}{\mathbb{E}}\left[ \frac{p_{\theta}(x)}{q(x)} r(x) \nabla_{\!\theta} \log p_\theta(x) \right]
\end{eqnarray}
$$
</div>

Two use this estimator we need two things (1) the ability to sample a joint $x
\sim p_{\theta}$, (2) the ability to compute the probability of a sampled value,
i.e., evaluate $\log p_{\theta}(x)$.

The real power of this method is when you have the ability to sample $x$, but
*not* the ability to compute probability of all factors of the joint probability
of $x$ (i.e., you can't compute the complete score $p_{\theta}(x)$). In other
words, some components of the joint probability's *generative process* might
pass through factors which are *only accessible through sampling*, e.g., because
they require performing *actual experiments* in the real world or a complex
simulation!

The factors that we can only sample from are what make this a true stochastic
optimization problem.

A classic example is a Markov decision process (MDP). In this context, the
random variable $x$ is an alternating sequence of states and actions, $x = (s_0,
a_0, s_1, a_1, \ldots a_{T-1}, s_T)$ and the generative process consists of an
unknown transition function $p(s_{t+1}|s_t,a_t)$ that is only accessible through
sampling and a policy $p_{\theta}(a_t|s_t)$ which we in control of. So the
probability of an entire sequence in an MDP is $p_{\theta}(x) = p(s_0)
\prod_{t=0}^T p(s_{t+1}|s_t,a_t) \pi_\theta(a_t|s_t)$. The likelihood-ratio
method can be used to derive several "policy gradient" methods, which compute
unbiased gradient estimates with no knowledge of the transition distribution.

> The beauty of the likelihood ratio is the cancellation of unknown terms.

This wonderful cancellation occurs in many contexts, including the
Metropolis-Hastings accept-reject criteria.

To make this explicit, let's consider the importance weight, $w(\tau)$.
\begin{eqnarray}
w(\tau)
= \frac{p(\tau|\pi_\theta)}{q(\tau)}
= \frac{ p(s_0) \prod_{t=0}^T p(s_{t+1}|s_t,a_t) \pi_\theta(a_t|s_t) }
       { p(s_0) \prod_{t=0}^T p(s_{t+1}|s_t,a_t) q(a_t|s_t) }
= \frac{\prod_{t=0}^T \pi_\theta(a_t|s_t)}
       {\prod_{t=0}^T q(a_t|s_t)}
\end{eqnarray}

<br/>
Common terms cancel! This implies that we don't need to compute them.


But that's only talking about the importance-weighted version... Well, another
interpretation of the log-probability is that related to cancellation.

Also, note that the component $\nabla_{\!\theta} \log p_{\theta}(x)$ also
simplifies because terms that do not depend on $\theta$ also disappear. Leaving
you with just a sum of log-gradient terms you know.

The general framework (along with a bunch of tricks and extensions) is presented
in [Bottou et. al (2013)](http://arxiv.org/abs/1209.2355)). This is one of the
best papers I have ever read. It took mean about a year and several reads to
really grokk it.

<hr/>
<hr/>
<hr/>

You can improve your data efficiency and algorithm stability using off-line
optimization.

* **Off-line optimization**: After you've collected a large sample (big $m$) you
  can optimze $\hat{J}$ using your favorite deterministic optimization algorithm
  (e.g., L-BFGS). You'll definitely want some type of "regularization" which
  prefers policies in places with sufficient samples. You can measure this type
  of thing in many ways (e.g., Bottou; Levine & Koltun, ; Philip Thomas, ; Tang &
  Abbeel, 2010). The original paper on this topic is (probably) "Learning from
  scarce experience" (Peshkin & Shelton, 2002) or Shelton's thesis. A similar
  deterministic approximation appears in PEGASUS (Ng & Jordan, 2000).



Another case where its tempting to apply policy gradient is in minimum risk
training of structured prediction models. Unfortunately, the likelihood-ratio
trick doesn't help us with the usual computation problems in structure
prediction, which have to do with computing normalization constants, but
assuming we can obtain good sample&mdash;preferably exact samples, but MCMC
samples might be ok&mdash;the likelihood ratio can help us with complicated
blackbox cost functions like human annotators or impenetrable perl scripts. I
had this idea back in 2012, but never got around to pushing it out. There appear
to be some recent papers that picked up on, including
[Sokolov+,2016](http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ACL2016.pdf)
and a few papers using it for variational inference.


I'd like to stress an important point. Although the likelihood-ratio gives us an
unbiased estimate of the gradient. It can even give us a surrogate objective to
optimize offline.

Don't be fooled. Just because there is a (stochastic) gradient used in the
method, this does not mean that you get the convergence rate that you might be
used to with your stochastic gradient method! In particular, the gradient
estimates will have high variance as it depends on variation in $r(x)$, which
might be crazy-large if, for example, $r(x)$ is sparse corresponding to winning
the lottery, completing a maze, winning at Go, finding some poor sap to click on
your ad. Compare this to the benign amount of noisy you get from subsampling the
data, which is the sgd that people are familiar with.

This gradient estimate is "zero order" it is essentially probing the function in
$x$ space, which might be higher dimensional that $\theta$. As a result, you
might be better off with gradient estimators that are based on perturbing
$\theta$ directly, e.g., so called "direct search" methods (or zeroth-order
optimization methods) like Nelder-Mead, FDSA, SPSA, and CMA-ES.


**Remarks**

 * **Relaxing discrete actions into stochastic ones**: A common way to handle
   discrete decisions is to put a *differentiable* parametric density (like
   $p_\theta$) over the space of possible executions (paths $x$). (Note: this
   shouldn't be surprising -- it's what we do in structured prediction,
   e.g. with CRFs!)  The likelihood-ratio method describe here can be used to
   estimate gradients in this setting. In a number of settings the $p_\theta$
   and $r$ decompose algebraically into a nice structure that is amenable to
   dynamic programming.

 * **Unknown environments**: Dynamics cancel out! Note that we need to be able
   to get samples from the unknown factors in the environment/model.

 * **Bandit feedback**: Learning under nondecomposable reward functions Policy
   gradient naturally handles "bandit feedback" (i.e., you only see the values
   of trajectories that you sample). In contrast with "full information" which
   tells you the reward of all possible trajectories.


**Take home messages**:

 * If you can evaluate it, then you can take the gradient of it (assuming it
   exists). This even holds if the evaluation is based on Monte Carlo.

 * The likelihood-ratio shows up all over the place, not just RL. It even shows
   up in counterfactual / causal reasoning.

 * We described a general way to learn from watching someone else act in a world
   we don't understand. The only catch is that in order for us to learn from
   them we need them to do a little bit of "exploration" (and tell us their
   action probabilities).

 * Policy gradient is useful in many domains, but usually doesn't work out of
   the box. It's an interesting set of math tricks nonetheless.
