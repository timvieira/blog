title: Likelihood-ratio gradient
date: 2016-06-09
comments: true
status: draft
tags: math, optimization, rl, policy-gradient


**Setup**: We're trying to optimize a function of the form

$$
J(\theta) = \underset{p_\theta}{\mathbb{E}} \left[ r(x) \right] = \sum_{x \in \mathcal{X}} p_\theta(x) r(x).
$$

The problem is that we can't enumerate $\mathcal{X}$ and $p$ times $r$ has no
structure over $\mathcal{X}$ which we can exploit (e.g., for dynamic
programming).

Suppose we can sample $x^{(j)} \sim p_\theta$. This opens up the following
(unbiased) Monte Carlo estimators for $J$ and its gradient,

$$
J(\theta) \approx \frac{1}{m} \sum_{j=1}^m r(x^{(j)})
$$

$$
\nabla_{\!\theta} J(\theta) \approx \frac{1}{m} \sum_{j=1}^m r(x^{(j)}) \nabla_{\!\theta} \log p_{\theta}(x^{(j)}).
$$


I'm going to work with a more general version based on
[importance Sampling](http://timvieira.github.io/blog/post/2014/12/21/importance-sampling/)
because it will give us some interesting freedom later on. Here $x^{(j)} \sim q$
instead of $p_\theta$. Note that we need the follow condition on $q$ to hold for
all $x$, $p(x) > 0 \Rightarrow q(x) > 0$.

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

The real power of this method is when you have the ability to sample $x$, but
*not* the ability to compute all factors of the joint probability of $x$. For
example, some components of the joint probability's *generative process* might
pass through factors which are *only accessible through sampling*, e.g., because
they require performing *actual experiments* in the real world or a complex
simulation!

So, let's think about being two types of conditional probability distributions
this graphical model: the ones we control via $\theta$ and those that are
independent of $\theta$.

A classic example is a Markov decision process (MDP). The random variable $x$ in
this context is a sequence of states and actions, $(s_0, a_0, s_1, a_1, \ldots
a_{T-1}, s_T)$ and the "generative process" consists of an unknown transition
function $p(s_{t+1}|s_t,a_t)$ that is only accessible through sampling and a
policy $p_{\theta}(a_t|s_t)$ which we in control of.


> The beauty of the likelihood ratio is the cancellation of unknown terms.


This wonderful cancellation occurs in many contexts, including the
Metropolis-Hastings accept-reject criteria.


<hr/>
<hr/>
<hr/>

**Deriving the policy gradient**: Here we describe the a basic method for
estimating $\nabla_\theta J(\pi_\theta)$, known as the likelihood-ratio
gradient.

Let $q(\tau)$ be distribution over trajectories, which comes from following a
policy $q$. (See my earlier post on
[Importance Sampling](http://timvieira.github.io/blog/post/2014/12/21/importance-sampling/).)


Given the MDP conditional independence assumptions, we can further simplify the
computation of the ratio term $\nabla_\theta p(\tau|\pi_\theta)/q(\tau),$
\begin{eqnarray}
\frac{\nabla_\theta p(\tau|\pi_\theta)}{{q(\tau)}}
&=& \frac{p(\tau|\pi_\theta)}{q(\tau)} \frac{\nabla_\theta p(\tau|\pi_\theta)}{p(\tau|\pi_\theta)} \\
&=& \frac{p(\tau|\pi_\theta)}{q(\tau)} \nabla_\theta \log p(\tau|\pi_\theta) \\
&=& \frac{p(\tau|\pi_\theta)}{q(\tau)} \nabla_\theta \log \left( p(s_0) \prod_{t=0}^H p(s_{t+1}|s_t,a_t) \cdot \pi_\theta(a_t|s_t)\right) \\
&=& w(\tau) \cdot \nabla_\theta \left( \log p(s_0) + \sum_{t=0}^H \log p(s_{t+1}|s_t,a_t) + \log \pi_\theta(a_t|s_t) \right) \\
&=& w(\tau) \cdot \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t|s_t) \label{eq:missingweights}.
\end{eqnarray}

Now, we expand the importance weight, $w(\tau)$.
\begin{eqnarray}
w(\tau)
= \frac{p(\tau|\pi_\theta)}{q(\tau)}
= \frac{ p(s_0) \prod_{t=0}^H p(s_{t+1}|s_t,a_t) \pi_\theta(a_t|s_t) }
       { p(s_0) \prod_{t=0}^H p(s_{t+1}|s_t,a_t) q(a_t|s_t) }
= \frac{\prod_{t=0}^H \pi_\theta(a_t|s_t)}
       {\prod_{t=0}^H q(a_t|s_t)}
\end{eqnarray}

Now, plugging our simplified expression for $w(\tau)$ back into
(\ref{eq:missingweights}) and (\ref{eq:likelihoodratio-sum-version}),

\begin{equation}\label{eq:LR-MDP}
 \nabla_\theta J(\theta) = \sum_{\tau} q(\tau)
      \left( \prod_{t=0}^H \frac{\pi_\theta(a_t|s_t)}{q(a_t|s_t)} \right)
      \cdot \left( \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t|s_t) \right)
      \cdot \left( \sum_{t=0}^H R(s_t) \right).
\end{equation}

This equation shows that *no knowledge of the dynamics are needed to estimate
the gradient* as long as we can sample from $q(\tau)$. This method gets the name
**likelihood-ratio** because of the division in the importance weight. It is
applicable beyond RL! Specifically, in cases where there are factors in a joint
probability model that we want to cancel out because we can't evaluate the
probability directly (e.g., because it is the real world not a simulation), but
we can sample from it (e.g., [Bottou+,13](http://arxiv.org/abs/1209.2355)).

**Computing the gradient**: If the state space is managable in size and we know
the parameters of the MDP, we can use dynamic programming to compute
$\nabla_\theta J(\pi_\theta)$ *exactly*. This is because the sum over
trajectories in Eq \ref{eq:LR-MDP} factors nicely due to the Markov assumptions.
Evaluting $J(\pi_\theta)$ is equivalent to computing the value of a Markov
reward process (an MDP with the policy fixed is a **Markov reward process**). To
get the gradient, you can simply backpropagate through the policy evaluation
procedure.

**Estimating the gradient**: In most cases, however, the state space is too big
*and* we don't know the dynamics or rewards, thus dynamic programming is
infeasible. So we use a **Monte Carlo estimate**. We can get an unbiased
estimates by sampling a bunch of trajectories, $\tau^{(1)} \dots \tau^{(m)}
\overset{\text{i.i.d.}}{\sim} q$ and computing

\begin{eqnarray}
\hat{J}(\pi_\theta)
&=& \frac{1}{m} \sum_{j=1}^m
  w^{(j)}
  \cdot \left( \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a^{(j)}_t|s^{(j)}_t) \right)
  \cdot \left( \sum_{t=0}^H R(s^{(j)}_t) \right). \label{eq:LRMCestimate}
\end{eqnarray}

where

$$
w^{(j)} = \frac{p(\tau^{(j)}|\pi_\theta)}{q(\tau^{(j)})} = \prod_{t=0}^H \frac{\pi_\theta(a^{(j)}_t|s^{(j)}_t)}{q(a^{(j)}_t|s^{(j)}_t)}
$$

This Monte Carlo estimate is unbiased, but is it any good?  Does it help us
estimate a good policy? One might imagine that if the state spaces or action
spaces is very large, far too many sample trajectories will be needed in order
to obtain a precise estimate; this notion can be described mathematically as the
variance of the estimate. Furthermore, if the sampling policy $q$ is too
dissimilar to $\pi_\theta$ the samples will be a weak estimate of the true
gradient.


**When does policy gradient fail**:

 * We need a strong signal from beginning to end when using policy
   gradient. Thus, policy gradient works best when used with aggressive **reward
   shaping**, that reward functions which aren't super "hands off" and "push
   back" the reward signal to earlier states. This is not always easy to do.

 * Stochastic policies tend to learn more conservative policies. (Although it is
   the case that deterministic policies are a special case of stochastic
   policies.) The classic example is Sutton and Barto's cliff problem where the
   optimal policy walks a straight line across the edge of a cliff, but
   stochastic policies tend to learn to move away from edge of the cliff because
   during learning it tends to fall off.

 * [Kakade & Langford (2002)](http://hunch.net/~jl/projects/aoarl/Final.ps)'s
   long corridor problem (I prefer "walking a tight rope" since it's more like
   the cliff problem from Sutton & Barto's book, which is also relevant to
   stochastic policies). In this case, it's difficult to reach final state and
   thus we never get a reward. It may take exponential time in $H$ to reach the
   final state even once when we're just sampling random stuff.

 * Long trajectories (for example the visual attention paper uses $5$
   steps. Jacob Andreas' recent best paper at NAACL uses $H=1$). Of course,
   there is a tradeoff between number of actions and trajectory length
   (otherwise we could say trivially say that we just have each trajectory as a
   single action making $H=1$ (in a trivial sense).

 * State spaces with sparse rewards. We need a strong reward signal to lead
   policy gradient in the right direction.

 * Another case, where its tempting to apply policy gradient is in structured
   predication where you try to minimize the risk (i.e., the expected cost of
   samples from the model). Likelihood ratio doesn't help us compute the "usual
   problems" in structure prediction, which have to do with computing
   normalization constants, but assuming we can obtain good
   sample&mdash;preferably exact samples, but MCMC samples might be ok&mdash;the
   likelihood ratio can help us with complicated blackbox cost functions like
   human annotators or impenetrable perl scripts. I had this idea back in 2012,
   but never got around to pushing it out. There appear to be some recent papers
   that picked up on, including
   [Sokolov+,2016](http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ACL2016.pdf)
   and a few papers using it for variational inference.


**Misc tricks**:

* **How big should we set $m$?** I like to set the minibatch size $m$ in any
  finicky SGD setup to ensure that the inner product between the current
  gradient and previous gradient is positive. This prevents excessive
  oscillation. Monitoring this quantity is pretty easy.

* **Gradient clipping** is sometimes useful because we may have exploding
  gradients. (It seems like every one uses the magic value $5$. I'd prefer to
  use something with appropriate units for the given problem. Maybe something
  based on the norms of previous gradients.)

* **Reward normalization**: I'm not crazy about normalizing rewards (e.g.,
  subtract mean and divide by variance; or possibly the rank transform used in
  CMA-ES).

* **Natural gradient**: Seems to work if the additional computation is faster
  than collecting more samples. I recommend analytic Fisher matrix-vector
  products instead of the empirical Fisher matrix and the truncated conjugate
  gradient trick to avoid ever materializing the Fisher matrix. Natural gradient
  is (approximately) parameterization invariant.

* **Self-normalized importance sampling** (divide by $\sum_{j=1}^m w^{(j)}$
  instead of $m$ in the MC estimate). Introduces bias which decreases with $m$
  (eventually the bias vanishes). Often works better than the vaniila REINFORCE
  algorithm.

* **Off-line optimization**: After you've collected a large sample (big $m$) you
  can optimze $\hat{J}$ using your favorite deterministic optimization algorithm
  (e.g., L-BFGS). You'll definitely want some type of "regularization" which
  prefers policies in places with sufficient samples. You can measure this type
  of thing in many ways (e.g., Bottou, Levine & Koltun, ; Philip Thomas; Tang &
  Abbeel, 2010). The original paper on this topic is (probably) "learning from
  scarce experience" (Peshkin & Shelton, 2002) or Shelton's thesis. A similar
  deterministic approximation appears in PEGASUS (Ng & Jordan, 2000)

* **Variance reduction**

  - The baseline trick: always recommended, not described here.

    **TODO**: Explain why the baseline "is a thing"... it looked totally
    arbitrary to me when I first saw it.

  - Control variate: generalization of optimal baseline to other available
    quantities with known exact expectations.

    (The optimal baseline uses the gradient-of-log-policy as a control
    variate. It's correct because the expected value of this quantity is
    zero. It can't hurt variance unless the coefficients are poorly estimated
    (actually, there are more precise conditions).)

  - Actor-critic (i.e., using a value function approximator in addition to
    policy learning)

    Introduces bias (unless we have a "compatible" parameterization between the
    policy and value function).

    There are many bias-variance tradeoffs available under this general scheme.

  - Past rewards are independent of future actions. This let's us rewrite the
    MC estimate to obtain a variance reduction. (not described here)

**Remarks**

 * **Relaxing discrete actions into stochastic ones**: A common way to handle
   discrete (i.e., nondifferentiable) structures/decisions is to put a
   differentiable density over the structure. (Note: this shouldn't be
   surprising -- it's what we do in structured prediction, e.g. with CRFs!)

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
