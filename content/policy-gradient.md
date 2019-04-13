title: The likelihood-ratio gradient
date: 2019-04-20
comments: true
tags: optimization, rl, machine-learning

**Setup**: We're trying to optimize a function of the form

$$
J(\theta) = \underset{p_\theta}{\mathbb{E}} \left[ r(x) \right] = \sum_{x \in \mathcal{X}} p_\theta(x) r(x).
$$

The problem is that we can't just evaluate each $x \in \mathcal{X}$ because we
don't have complete knowledge of $p_\theta$.  For example, it is a mix of
factors that are known and under our control via $\theta$ (policy factors) and
factors that are not known (environment factors).

The likelihood-ratio gradient estimator is an approach for solving such a
problem.  It appears in policy gradient methods for reinforcement learning
(e.g.,
[Sutton et al. 1999](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)),
black-box optimization, and causal reasoning. There are two main ideas in the
trick: (1) the "score function" estimator and (2) the cancelation of
complicating factors.


#### Part 1: The score function gradient estimator

Suppose we can sample $x^{(j)} \sim p_\theta$. This opens up the following
(unbiased) Monte Carlo estimators for $J$ and its gradient,

$$
J(\theta) \approx \frac{1}{m} \sum_{j=1}^m r(x^{(j)})
$$

$$
\nabla_{\!\theta} J(\theta) \approx \frac{1}{m} \sum_{j=1}^m r(x^{(j)}) \nabla_{\!\theta} \log p_{\theta}(x^{(j)}).
$$


The derivation is pretty simple
$$
\begin{eqnarray*}
  \nabla_{\!\theta} \, \underset{p_\theta}{\mathbb{E}}\left[ r(x) \right]
  &=& \nabla_{\!\theta} \left[ \sum_x p_{\theta}(x) r(x) \right] \\
  &=& \sum_x \nabla_{\!\theta} \left[ p_{\theta}(x) \right] r(x) \\
  &=& \sum_x p_{\theta}(x) \frac{\nabla_{\!\theta} \left[ p_{\theta}(x) \right] }{ p_{\theta}(x) } r(x) \\
  &=& \underset{p}{\mathbb{E}}\left[ r(x) \nabla_{\!\theta} \log p_\theta(x) \right]
\end{eqnarray*}
$$

<br/>
We use the identity $\nabla f = f\, \nabla \log f$, assuming $f > 0$.

To use this estimator, we only need two things (1) the ability to sample
$x^{(j)} \sim p_{\theta}$, (2) the ability to evaluate $\log
p_{\theta}(x^{(j)})$ and $r(x^{(j)})$ for each sampled value.

This isn't even the entire method, but we can already use it to do some neat
things.  For example, minimum risk training of structured prediction models.
Assuming we can obtain good samples&mdash;preferably exact samples, but MCMC
samples might be ok&mdash;the likelihood ratio can help us learning even with
complicated blackbox cost functions (sometimes called "nondecomposable loss
functions") like human annotators or impenetrable perl scripts. I had this idea
back in 2012, but never got around to pushing it out. There appear to be some
papers that picked up on this idea, including
[Sokolov et al. (2016)](http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ACL2016.pdf)
and [Norouzi et al. (2016)](https://arxiv.org/abs/1609.00150) and even a few
papers using it for "black box" variational inference
[(Ranganath et al., 2013)](https://arxiv.org/abs/1401.0118).

*Remarks:*

 * **Relaxing discrete actions into stochastic ones**: A common way to handle
   discrete decisions is to put a *differentiable* parametric density (like
   $p_\theta$) over the space of possible executions (paths $x$). (Note: this
   shouldn't be surprising&mdash;it's already what we do in structured
   prediction methods like conditional random fields!)  The likelihood-ratio
   method can be used to estimate gradients in such settings.

 * **Bandit feedback**: This approach naturally handles "bandit feedback"
   (partial information about $r$): you only see the values of only the
   trajectories that you actually sample. In contrast with "full information",
   which tells you the reward of all possible trajectories.


##### The off-policy estimator

Let's generalize this estimator to allow off-policy actions
[importance-weighted estimator](https://timvieira.github.io/blog/post/2014/12/21/importance-sampling/). Here
$q$ is a distribution over the same space as $p$ with support at least
everywhere $p$ has support.  $$ \begin{eqnarray*} \nabla_{\!\theta} \,
\underset{p_\theta}{\mathbb{E}}\left[ r(x) \right] &=&
\underset{p}{\mathbb{E}}\left[ r(x) \nabla_{\!\theta} \log p_\theta(x) \right]
\\ &=& \sum_x p_{\theta}(x) r(x) \nabla_{\!\theta} \log p_\theta(x) \\ &=&
\sum_x \frac{q(x)}{q(x)} p_{\theta}(x) r(x) \nabla_{\!\theta} \log p_\theta(x)
\\ &=&
\underset{q}{\mathbb{E}}\left[ \frac{p_{\theta}(x)}{q(x)} r(x) \nabla_{\!\theta} \log p_\theta(x) \right]
\\ &\approx& \frac{1}{n} \sum_{i=1}^n \frac{p_{\theta}(x^{(i)})}{q(x^{(i)})} r(x^{(i)}) \nabla_{\!\theta} \log p_\theta(x^{(i)})\quad \text{ where } x^{(i)} \sim q
\end{eqnarray*} $$

Note that we recover the original estimator when $q=p$.


#### Part 2: The convenient cancelation of complicating components

The real power of the *likelihood-ratio* part of this method comes when you have
the ability to sample $x$, but *not* the ability to compute the probability of
*all* factors of the joint probability of $x$ (i.e., you can't compute the
*complete* score $p_{\theta}(x)$). In other words, some components of the joint
probability's *generative process* might pass through factors which are *only
accessible through sampling*, e.g., because they require performing *actual
experiments* in the real world or a complex simulation! The factors that we can
only sample from are what make this a true stochastic optimization problem.

Let's be a little more concrete by looking at a classic example from
reinforcement learning: the Markov decision process (MDP). In this context, the
random variable $x$ is an alternating sequence of states and actions, $x =
\langle s_0, a_0, s_1, a_1, \ldots a_{T-1}, s_T \rangle$ and the generative
process consists of an unknown transition function $p(s_{t+1}|s_t,a_t)$ that is
only accessible through sampling and a policy $p_{\theta}(a_t|s_t)$ which we in
control of. So the probability of an entire sequence in an MDP $p_{\theta}(x)$
is $p(s_0) \prod_{t=0}^T p(s_{t+1}|s_t,a_t) \pi_\theta(a_t|s_t)$. The
likelihood-ratio method can be used to derive several "policy gradient" methods,
which compute unbiased gradient estimates with no knowledge of the transition
distribution.

> The beauty of the likelihood ratio is the cancellation of unknown terms.

Aside: This fortunate cancellation occurs in many other contexts, e.g. the
Metropolis-Hastings accept-reject criteria.

To make this explicit, let's consider the importance weight, $p/q$.
\begin{eqnarray}
\frac{p_\theta(x)}{q(x)}
= \frac{ {\color{red}{ p(s_0) }} \prod_{t=0}^T {\color{red}{ p(s_{t+1}|s_t,a_t) }} \pi_\theta(a_t|s_t) }
       { {\color{red}{ p(s_0) }} \prod_{t=0}^T {\color{red}{ p(s_{t+1}|s_t,a_t) }}  q(a_t|s_t) }
= \frac{\prod_{t=0}^T \pi_\theta(a_t|s_t)}
       {\prod_{t=0}^T q(a_t|s_t)}
\end{eqnarray}

<br/>
Common terms cancel! This implies that we don't need to compute them.

Those component cancel in $\nabla_{\!\theta} \log p_{\theta}(x)$ because terms
that do not depend on $\theta$ also disappear. Leaving you with just a sum of
log-gradient terms that you *do* know because they are part of the model you're
tuning.

$$
\begin{eqnarray*}
\nabla \log p(x)
&=& \nabla \log \left( p(s_0) \prod_{t=0}^T p(s_{t+1}|s_t,a_t) \pi_\theta(a_t|s_t) \right) \\
&=& \nabla \left(\log p(s_0) + \sum_{t=0}^T \log p(s_{t+1}|s_t,a_t)
  + \log \pi_\theta(a_t|s_t) \right) \\
&=& \sum_{t=0}^T \nabla \log \pi_\theta(a_t|s_t)
\end{eqnarray*}
$$

#### The baseline trick

These estimators should always be used in conjunction with a baseline function
or more generally a control variate. There are many options for deriving control
variates, which will depend on the specific structure of $x$.  For example, in
the MDP case, we can use any function that depends on $s_t$.

However, even without special structure, we can an always should use (at a
minimum) a constant baseline,
$$
\mathbb{E}_{x \sim q} \left[
\frac{p_{\theta}(x)}{q(x)}
r(x)
\nabla_{\!\theta} \log p_\theta(x)
\right]
=
\mathbb{E}_{x \sim q} \left[
\frac{p_{\theta}(x)}{q(x)}
(r(x) - {\color{red}{b}})
\nabla_{\!\theta} \log p_\theta(x)
\right]
\text{for all } {\color{red}{b} \in \mathbb{R}}
$$
The minimum variance choice for b is
$$
b = \frac{\sum_k \mathrm{Cov}(r, \nabla_{\theta_k} \log p) }{\sum_k \mathrm{Var}(\nabla_{\theta_k} \log p) }
$$
which we can compute with sampling-based estimators of the quantities.

Some folks use an estimate of $J$, which is better than nothing.

#### Important points

 * Always use a baseline.

 * This gradient estimate is "zero order" it is essentially probing the function
   in $x$ space, which might be higher dimensional than $\theta$. As a result,
   you might be better off with gradient estimators that are based on perturbing
   $\theta$ directly, e.g., zeroth-order methods (sometimes called *direct
   search* or *gradient-free* optimization methods) like Nelder-Mead simplex,
   FDSA, SPSA, and CMA-ES.

 * Often there is almost no signal. Consider the example of trying to solve a
   maze by randomly running around in it.  In this case, it's very unlikely that
   a random path will lead to a positive outcome.  Therefore, the gradient
   really is essentially zero. So even with access to the *true* gradient (i.e.,
   no variance), optimization would have a lot of trouble finding a good
   optimum.  Add to that some variance and you have useless on top of noisy.

 * Although the likelihood-ratio gives us an unbiased estimate of the gradient,
   don't be fooled. The particular gradient estimate used in the
   likelihood-ratio method has an impractical signal-to-noise ratio, which makes
   it very hard use in optimization.  There are countless papers on tricks to
   reduce the variance of the estimator.

 * You can improve your data efficiency and algorithm stability using off-line
   optimization (with your favorite deterministic optimization algorithm).  I
   have written a long article about offline optimization
   [here]((https://timvieira.github.io/blog/post/2016/12/19/counterfactual-reasoning-and-learning-from-logged-data/)).


## Summary

There is still a lot to say about likelihood-ratio methods.  I didn't talk about
control variates or "baseline" functions, which are very important to making
things work.  I'll try to post my notes on those ideas soon.

**Take home messages**:

 * If you can evaluate it, then you can take the gradient of it (assuming it
   exists). This even holds if the evaluation is based on Monte Carlo.

 * The likelihood-ratio shows up all over the place, not just RL. It shows up in
   [causal reasoning](https://timvieira.github.io/blog/post/2016/12/19/counterfactual-reasoning-and-learning-from-logged-data/)
   more generally.

 * We described a general way to learn from watching someone else act in a world
   we don't understand (i.e., off-policy learning with no knowledge of the
   environment just samples!). The only catch is that in order for us to learn
   from them we need them to do a little bit of "exploration" (i.e., be a
   stochastic policy that has support everywhere we do) and tell us their action
   probabilities (so that we can important weight against our policy).
