title: Policy gradient reinforcement learning
date: 2016-06-09
comments: true
tags: math, optimization, rl, policy-gradient

Policy search methods are a popular approach to reinforcement learning (RL)
problems due to their ability to scale to large state spaces including
continuous and partially observed domains. Policy search methods operate by
directly searching the space of policies, as opposed to indirect methods, which
learn a model of the environment and derive a policy from it. The most common
policy search methods are based on stochastic gradient ascent, termed policy
gradient methods
([Williams,92](http://incompleteideas.net/sutton/williams-92.pdf);
[Sutton+,99](https://webdocs.cs.ualberta.ca/~sutton/papers/SMSM-NIPS99.pdf)).

Since we are in the RL setting, we will be searching for a good policy *without
a priori knowledge of the dynamics model or the reward function*.

A **Markov Decision Process** (MDP) over a state set $\mathcal{S}$ and action
set $\mathcal{A}$ consists of:

  * $p(s_{t+1}|s_t,a_t)$: A **dynamics model** that specifies the probability of
    transition to each state $s_{t+1}$ upon execution of action $a_t$ in state
    $s_t$.

  * $R(s_t)$: A **reward function** that specifies a scalar value for being in
    state $s_t$.

We'll assume that periodically (after every $H$ steps) the environment will
terminate the current **episode** and drop us in an initial state $s_0$.

Our goal is to attain a policy, $\pi$, a mapping from states to action, such
that the **utility** $J(\pi)$ is maximized,

\begin{equation}\label{def:policyperformance}
  J(\pi) = \mathbb{E}\left[ R(s_0) + R(s_1) + \dots + R(s_H) \Biggr| \pi \right].
\end{equation}

Given this definition it is natural to want to try to optimize $J(\pi)$ by
gradient ascent. We'll get there in a minute; let's talk a little bit more about
notation and assumptions.

Policy gradient assumes that $\pi$ is a **stochastic policy**, i.e., actions are
drawn randomly $a \sim \pi(\cdot|s)$. Additionally, $\pi$ is parametrized by a
vector, $\theta \in \mathbb{R}^d$, and $\frac{\partial
\pi_\theta}{\partial\theta}$ exists.

A **trajectory** in an MDP, is a sequence of random variables representing
states and actions drawn from environment $\tau=[s_0,a_0,s_1,a_1\dots,s_H,a_H]$
by following a policy, $\pi_\theta$ in the environment for $H$ steps ($a_H$ is a
dummy action so we don't count it towards the trajectory length). Let
$p(\tau|\pi_\theta)$ denote the distribution over trajectories in the
environment given that we are following a particular policy.

The conditional independence assumptions of the MDP imply the following
factorization of $p(\tau|\pi_\theta)$,

\begin{equation}\label{eq:mdp-prob}
p(\tau|\pi_\theta) = p(s_0) \prod_{t=0}^H p(s_{t+1}|s_t,a_t) \pi_\theta(a_t|s_t)
\end{equation}


**Deriving the policy gradient**: Here we describe the a basic method for
estimating $\nabla_\theta J(\pi_\theta)$, known as the likelihood-ratio
gradient.

Let $q(\tau)$ be distribution over trajectories, which comes from following a
policy $q$. (See my earlier post on
[Importance Sampling](http://timvieira.github.io/blog/post/2014/12/21/importance-sampling/).)

\begin{eqnarray}
  \nabla_\theta J(\theta)
  &=& \nabla_\theta \mathbb{E}_{\tau \sim p(\cdot|\pi_\theta) }\left[ \sum_{t=0}^H R(s_t) \Biggr| \pi_\theta \right]  \\
  &=& \nabla_\theta \sum_\tau p(\tau|\pi_\theta) R(\tau) \\
  &=& \sum_\tau \nabla_\theta p(\tau|\pi_\theta) R(\tau) \label{eq:requires-interchange} \\
  &=& \sum_\tau q(\tau) \frac{\nabla_\theta p(\tau|\pi_\theta)}{q(\tau)} R(\tau) \label{eq:likelihoodratio-sum-version} \\
  &=& \mathbb{E}_{\tau \sim q}\left[ \frac{\nabla_\theta p(\tau|\pi_\theta)}{q(\tau)} R(\tau) \Biggr| \pi_\theta \right] \label{eq:likelihoodratio}
\end{eqnarray}

A few notes on the derivation:

 * Note that even if $q = p_\theta$, we still need to "correct" for the bias
   (via importance weights, i.e., we divide by $q$ in our sampled gradients even
   if $q=p_\theta$.

 * Technical note: Line \ref{eq:requires-interchange} requires an interchange of
   derivative and integral, which can be troublesome if any bits of the math can
   explode. We're not going to worry about it.

 * It's a very interesting fact that we can obtain unbiased gradient estimates
   for $\pi_\theta$ by evaluating on samples from a completely different policy!
   For example, $q(\tau)=p(\tau|\pi_\theta)$ is a valid choice, as is
   $p(\tau|\pi_{\theta'})$ for a different set of parameters. In fact, *any*
   distribution over trajectories, which has support everywhere
   $p(\tau|\pi_\theta)$, is valid (a general requirement in importance
   sampling).

Given the MDP conditional independence assumptions, we can further simplify the
computation of the ratio term $\nabla_\theta p(\tau|\pi_\theta)/q(\tau)$,
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

**Computating the gradient**: If the state space is managable in size and we
know the parameters of the MDP, we can use dynamic programming to compute
$\nabla_\theta J(\pi_\theta)$ *exactly*. This is because the sum over
trajectories in Eq \ref{eq:LR-MDP} factors nicely due to the Markov assumptions
Evaluting $J(\pi_\theta)$ is equivalent to computing the value of a Markov
reward process (an MDP with the policy fixed is a Markov reward process; you see
this as a subroutine in algorithms like policy iteration). To get the gradient,
simply backpropagate through your favorite Markov reward process value
estimation procedure.

In most cases, however, the state space is too big *and* we don't know the
dynamics or rewards, thus dynamic programming is infeasible. So we use a **Monte
Carlo Approximation**. We can get an unbiased estimates by sampling a bunch of
trajectories, $\tau^{(1)} \dots \tau^{(m)} \overset{\text{i.i.d.}}{\sim} q$ and
computing

\begin{eqnarray}
\hat{J}(\pi_\theta) &=& \frac{1}{m} \sum_{j=1}^m \frac{\nabla_\theta p(\tau^{(j)}|\pi_\theta)}{q(\tau^{(j)})} R(\tau^{(j)}) \\
&=& \frac{1}{m} \sum_{j=1}^m
  \left( \prod_{t=0}^H \frac{\pi_\theta(a^{(j)}_t|s^{(j)}_t)}{q(a^{(j)}_t|s^{(j)}_t)} \right)
  \cdot \left( \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a^{(j)}_t|s^{(j)}_t) \right)
  \cdot \left( \sum_{t=0}^H R(s^{(j)}_t) \right). \label{eq:LRMCestimate}
\end{eqnarray}

This Monte Carlo estimate is unbiased, but is it any good?  Does it help us
estimate a good policy? One might imagine that if the state spaces or action
spaces is very large, far too many sample trajectories will be needed in order
to obtain a precise estimate; this notion can be described mathematically as the
variance of the estimate. Furthermore, if the sampling policy $q$ is too
dissimilar to $\pi_\theta$ the samples will be a weak estimate of the true
gradient.


**Other stuff**

 * baseline - control variate - actor-critic
   (compatible function)

 * normalized version

 * relaxing discrete decisions to stochastic ones. Learning under complicated
   reward functions in complex environments.

**When does policy gradient fail**:

 * We need a strong signal from beginning to end when using policy
   gradient. Thus, policy gradient works best when used with aggressive **reward
   shaping**, that reward functions which aren't super "hands off" and "push
   back" the reward signal to earlier states. This is not always easy to do.

 * Kakade & Langford (2002): the long corridor problem

 * Cliff problem from Sutton and Barto
