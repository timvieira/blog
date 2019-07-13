title: Value functions as Lagrange multiplier estimates
date: 2018-03-16
comments: true
tags: rl, calculus, Lagrange-multipliers
status: draft

[Value functions](https://en.wikipedia.org/wiki/Bellman_equation), or some
variant thereof, are key concept in sequential decision-making tasks (e.g.,
reinforcement learning, planning under uncertaintly, and optimal control).  They
generally regarded as (somewhat intuitive) definitions that seem to help solve
the decision-making problem.  In this post, I will give an account of value
functions as Lagrange multiplier estimates for a specific formulation of the
policy-search problem in reinforcement learning - I believe this story
generalize to the other cases, but I haven't worked it out formally.  This
connection is pretty cool and is closely related to my previous post on
[backpropagation and Lagrangians](http://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/).

<!--
They are a mathematical operationalization of the idea that

> “Life can only be understood backwards; but it must be lived forwards.”
>
> ― [Søren Kierkegaard](https://www.goodreads.com/quotes/6812-life-can-only-be-understood-backwards-but-it-must-be)
-->

**The setup**: Let $M = \langle S, A, p_0(s), p(s' \mid s, a), r(s,a), \gamma \rangle$
be a Markov decision process over states $S$ and actions $A$ with $0 < \gamma < 1$.  Let $p_0(s)$ be a
distribution over initial states and $p(s' \mid s, a)$ be a transition kernel
that describes how the next state $s'$ evolve from current state $s$ given an action $a$
from a policy $\pi(a \mid s)$.  Lastly, let $r(s,a)$ denote the immediate reward of taking action $a$ in state $s$. The utility of a policy $\pi$ is typically measure as the long-term,
$\gamma$-discounted reward
$$
U(\pi) \overset{\text{def}}{=} \mathbb{E}\left[  \sum_{t=0}^\infty \gamma^t \!\cdot\! r(s,a) \right]
$$

where the randomness in the expecation is take over trajectories, $\langle
\langle s_t, a_t, r_t \rangle \rangle_{t=0}^\infty$.  Don't worry too much about
the infinite summation, we will now replace it with something much more
computation friendly.

**The policy search problem:** Here we write the policy search problem as a
constrained optimization problem.  We exploit an important rewrite of $U(\pi)$
as the expectation an expectation over an occupancy measure over states rather
than an expectation over infinitely long trajectories.

$$
\underset{\pi, \delta}{\textrm{maximize }} U = \frac{1}{1-\gamma}\sum_{s,a} r(s,a) \cdot \delta(s) \cdot \pi(a|s)
$$

*subject to*

 - $\pi$ is a valid conditional probability distribution

$$
\textstyle\sum_a \pi(a | s) = 1 \quad\text{for all } s \in S
$$
$$
\pi(a | s) \ge 0 \quad\text{for all } s \in S, a \in A
$$

 - $\delta$ is $\pi$'s occupancy measure (**footnote**: Alternatively, we can
   think of $\delta$ as $\pi$'s stationary distribution if we regard
   $(1-\gamma)$ as the probability of restarting the Markov chain (i.e.,
   sampling the next state from a mixture of $p_0(s')$ with probability
   $(1-\gamma)$ and from $p(s' \mid s,a)$ with probability $\gamma$).  This is
   an equivalent view for expectations, but not higher-order momements, e.g.,
   it's not equivalent in distribution.)

$$
\delta(s') = (1-\gamma) \cdot p_0(s') + \gamma \cdot \sum_{s,a} \delta(s) \cdot \pi(a|s) \cdot p(s'|s,a)\quad\text{for all }s' \in S
$$
$$
\sum_{s} \delta(s) = 1
$$
$$
\delta(s) \ge 0 \quad\text{for all }s
$$


Written as above, this optimization problem is a quadratic program with
quadractic equality constraints.  Quadratic equality constraints can encode
nonconvex constraints, which are generally NP-Hard to solve.  (**footnote**: To
see why, consider the constraint $x \cdot (x - 1) = 0$, this constraint has
exactly two solutions $x \in \{0, 1\}$.  In other words, it is an encoding of
binary variables, and thus can be used to solve NP-hard problems.)


## Lagrangian
We can write out the Lagrangian for this optimization problem

$$
\begin{eqnarray*}
\mathcal{L}(\delta, \pi, \lambda, \sigma, \zeta, \eta)
&=& \sum_{s,a} r(s,a) \delta(s) \pi(a|s)\\
&& + \sum_{s'} \lambda(s') \delta(s')  - \sum_{s,a} \lambda(s') \delta(s) \pi(a|s) p(s'|s,a)\\
&& + \sum_s \sigma(s)  - \sum_a \sigma(s) \pi(a | s)\\
&& + \sum_{s,a} \zeta(s,a) \pi(a | s) \\
&& + \sum_{s} \eta(s) \delta(a | s)
\end{eqnarray*}
$$


TODO: What does it mean to be a Lagrange multiplier *estimate*

Now, let's solve for $\nabla \mathcal{L} = 0$ under each chunk of parameters.

$$
\begin{eqnarray*}
\frac{\partial \mathcal{L}}{\partial \delta(s^*)}
&=&
\sum_a r(s^*,a) \pi(a \mid s^*)
+ \sum_{s'} \lambda(s') (1(s' = s^*)
- \sum_{a} \delta(s^*) \pi(a \mid s^*) p(s' \mid s^*, a))\\
\end{eqnarray*}
$$


$$
\begin{eqnarray*}
\frac{\partial \mathcal{L}}{\partial \pi(a^* \mid s^*)}
&=&
r(s^*,a^*) \delta(s^*)
- \sum_{s'} \lambda(s') \delta(s^*) p(s'|s^*,a^*)
 - \sum_a \sigma(s^*)
 + \zeta(s^*,a^*)
\end{eqnarray*}
$$


## Linear programming re-formulation

Manne (1960), came up with a very clever trick which reformulates this
optimization into a linear programming problem.  The trick is to "flatten" the
optimization of $\pi$ and $\delta$, by optimizing over their product $\mu(s,a)
\overset{\text{def}}{=} \pi(a \mid s) \cdot \delta(s)$ instead.  We can recover
our original variables as $\delta(s) = \sum_a \mu(s,a)$ and $\pi(a \mid s) =
\mu(s,a) / \delta(s)$.  If every $\delta(s)=0$ the choice for $\pi(a \mid s)$ is
an arbitrary distribution over $a$.

$$
\underset{\pi}{\textrm{maximize }} \frac{1}{1-\gamma}\sum_{s,a} r(s,a) \cdot \mu(s, a)
$$

*subject to*

 - $\mu$ is a valid occupancy measure over state-actions pairs

$$
\sum_{a'} \mu(s',a') = (1-\gamma) \cdot p_0(s') + \gamma \cdot \sum_{s,a} \mu(s,a) \cdot p(s'|s,a)\quad\text{for all }s' \in S
$$
$$
\mu(s, a) \ge 0 \quad\text{for all }s, a
$$

It turns out that explicit sum-to-one constraint (i.e., $\sum_{s,a} \mu(s,a) =
1$) is not necessary because solutions to the linear constraints will are
already normalized.

**TODO** Formulate the Lagrangian.  Take its derivatives.

**TODO** Take the LP dual, note that it is exactly the optimal value function
  problem.

**TODO** Block-coordinate method - is policy iteration.  Just like my backprop
  post, we can take any fixed policy and estimate the Lagrange mulipliers by
  setting the gradient of the Lagrangian wrt $V$ equal to zero given what every
  $\pi$ is.

**TODO** The successor representation is the dual of this connection -- They are
Lagrange multiplier estimates of the in the primal problem.


## Connections in graphical models

The concept of a value function is not limited to RL: value functions arise in
the dynamic programming solutions to many other problems.

Building on
[Vieira (2017)](http://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/),
the gradient estimates provided by backpropagation can be viewed as Lagrange
muliplier estimates in a particular formulation of an optimization problem with
intermediate variables that are defined by equality constraints on intermediate
quantities.

Consider the case of the forward-backward algorithm for linear-chain CRFs or
HMMs.  As detailed in
[Eisner (2016)](https://www.cs.jhu.edu/~jason/papers/eisner.spnlp16.pdf), the
backward algorithm and the outside algorithm are *precisely* the result of
backpropgation on the forward algorithm and inside algorithm respectively.

Much like the MDP setting, the backward algorithm and outside algorithm are
often viewed as some useful quantities.  However, when viewed as the result of
backpropagation it not only deepens our understanding of the connection, but
also establishes a bridge to the rich theory that underlies automatic
differentiation.  This connection tells us important things about the time and
space complexity of algorithms for computing these quantities and even better
gives us a recipe for efficiently computing these quantities backward/outside
quantities.

This connection extends to marginal inference in Bayesian networks more
generally [Darwiche (2003)](https://dl.acm.org/citation.cfm?id=765570).

> Adnan Darwiche. [A differential approach to inference in Bayesian networks](https://dl.acm.org/citation.cfm?id=765570). Journal of the Association for Computing Machinery, 50(3):280–305, 2003.
