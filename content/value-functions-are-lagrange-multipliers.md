title: Value functions as Lagrange multiplier estimates
date: 2019-07-17
comments: true
tags: rl, calculus, Lagrange-multipliers
status: draft

[Value functions](https://en.wikipedia.org/wiki/Bellman_equation), or some
variant thereof, are key concept in sequential decision-making tasks (e.g.,
reinforcement learning, planning under uncertaintly, and optimal control).  They
generally regarded as (somewhat intuitive) definitions that seem to help solve
the decision-making problem.  In this post, I will give an account of value
functions as Lagrange multiplier estimates for a specific formulation of the
policy-search problem in reinforcement learning.  This connection is pretty cool
and is closely related to my previous post on
[backpropagation and Lagrangians](http://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/).

<div style="border: thin solid black; padding: 10px; background-color: #ffffcc;
margin-bottom: 1.5em;">

The ideal version of this work would be a completed
commutation diagram of the primal and dual views of the MDP optimization problem
much like the diagrams in consumer theory.[^LM04]

I am confident that every quantity will have many connections (each one
analogous to their "cousin" in economics).  This diagram will be an elegant,
unifying view of lots of concepts in MDPs.

Making the implicit functions explicit will make a lot of shortcuts simplier as
well as providing calculus views of policy gradients (for example) than the
usual expectations views.

I am confident that all connections are "interesting" in one way or another.
I am also confident that if we discover a new one, then that's even more interesting.

These connections should even cover policy gradients and successor
representations!
Any properties of the mappings will be of interest as well.
For example, the nonconvexity of the VF polytope as the shadow of the optimal
value function polytope.  It is also interesting that the mapping is not
generally convex.

What are the interesting plots for MDPs?  (analogues of income-consumption
curves) The VF-polytope "line theorem" is like a econ theorem.
</div>

<!--
They are a mathematical operationalization of the idea that

> “Life can only be understood backwards; but it must be lived forwards.”
>
> ― [Søren Kierkegaard](https://www.goodreads.com/quotes/6812-life-can-only-be-understood-backwards-but-it-must-be)
-->

Let $M = \langle \mathcal{S}, \mathcal{A}, p_0(s), p(s' \mid s, a), r(s,a), \gamma \rangle$
be a Markov decision process over states $S$ and actions $A$ with $0 < \gamma < 1$.  Let $p_0(s)$ be a
distribution over initial states and $p(s' \mid s, a)$ be a transition kernel
that describes how the next state $s'$ evolve from current state $s$ given an action $a$
from a policy $\pi(a \mid s)$.  Lastly, let $r(s,a)$ denote the immediate reward of taking action $a$ in state $s$.

The utility of a policy $\pi$ is typically measured as the long-term, $\gamma$-discounted reward
$$
U(\pi) \overset{\text{def}}{=} \mathbb{E}\left[  \sum_{t=0}^\infty \gamma^t \!\cdot\! r(s,a) \right]
$$

where the randomness in the expecation is taken over trajectories,
$\langle \langle s_t, a_t, r_t \rangle \rangle_{t=0}^\infty$.
The next step is rewrite $U(\pi)$ as an expectation over state-visitation fequencies $\delta(s)$, also know as the occupancy measure, rather than an expectation over infinitely long trajectories.[^unnecessary-constraints]

$$
(1-P_{\pi})^{-1} = \lim_{T \rightarrow \infty} \sum_{t=1}^T P_{\pi}^{t}
$$
XXX: this explanation is incomplete. Need to define $P_{\pi}$.


$$
\begin{cases}
\underset{\pi, \delta}{\textrm{maximize }} & \frac{1}{1-\gamma} \sum_{s,a} r(s,a) \cdot \delta(s) \pi(a|s) \\
\text{subject to } &
\sum_{a'} \delta(s') \pi(a' \mid s') = (1-\gamma) p_0(s') + \gamma \sum_{s,a} \delta(s) \pi(a|s) \cdot P(s'|s,a)\quad\text{for all }s' \in \mathcal{S}
\label{eq:balance} \\
& \delta(s) \pi(a \mid s) \ge 0 \quad\text{for } s \in \mathcal{S}, a \in \mathcal{A}
\end{cases}
$$


Written as above, this optimization problem is a quadratic program with
quadractic equality constraints.[^qp] Lucky for us, this optimization
problem&mdash;as a function of $\mu(s,a) \overset{\text{def}}{=} \delta(s)\pi(a
\mid s)$&mdash;is a linear program!  Performing the substitution gives the
following optimization problem,

$$
\begin{cases}
\underset{\mu}{\textrm{maximize }} & \sum_{s,a} r(s,a) \cdot \mu(s, a) \\
\text{subject to} &
\sum_{a'} \mu(s',a') = \sum_{s,a} \mu(s,a) \cdot P(s'|s,a)\quad\text{for all }s' \in \mathcal{S} \\
& \mu(s, a) \ge 0 \quad\text{for all }s \in \mathcal{S}, a \in \mathcal{A}
\end{cases}
$$

The linear programming version formulation suggests allows us to use a number of
efficient (polytime) algorithms.  Once solved, we can recover our original
variables as $\delta(s) \mapsto \sum_a \mu(s,a)$ and $\pi(a \mid s) \mapsto
\frac{\mu(s,a)}{\delta(s)}$.[^ties] The linear programming formulation is
attributed to Manne (1960); however, the best resource on the topic is Wang et
al. (2008).[^W08]

[^unnecessary-constraints]: We also want $\delta(s) \pi(a \mid s)$ to be a valid
  joint state-action distribution and for $\pi$ to be a valid conditional
  distribution.
  $$
  \begin{cases}
  &\sum_{s,a} \delta(s) \pi(a \mid s) = 1
  &\textstyle\sum_a \pi(a | s) = 1 \quad\text{for all } s \in \mathcal{S} \\
  &\pi(a | s) \ge 0 \quad\text{for all } s \in \mathcal{S}, a \in \mathcal{A}
  \end{cases}
  $$
  It turns out that we don't need to explicitly add these constraints because
  they are implied by the existing balance constraints and assumptions that $P$
  is a stochastic matrix.

  TODO: need to prove that this is true in the general.

  Lemma 1 says the solution to the balance equation are always normalized.

  The average-reward case, on the other hand, needs an explicit sum-to-one
  constraint.  This explains why the average-reward settings value functions are
  different from the discounted case.  Note that the average-reward formulation
  on a $\gamma$-resetting transition model is not equivalent.  (TODO: double
  check)

[^ties]: If ever $\delta(s)=0$, the choice for $\pi(a \mid s)$ is any valid
  distribution over $a$.

[^qp]: Optimization with quadratic equality constraints are generally NP-Hard to
  solve.  To see why, consider the constraint $x \cdot (x
  - 1) = 0$, this constraint has exactly two solutions $x \in \{0, 1\}$.  In
  other words, it is an encoding of binary variables, and thus can be used to
  solve NP-hard integer programming problems.


### The Lagrangian

<!--
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
--->


$$
\begin{align*}
\mathcal{L}(\delta, \pi, \lambda) \overset{\text{def}}{=}
& \sum_{s,a} r(s,a) \cdot \delta(s) \pi(a|s) \\
& + \sum_{s'} \lambda(s') \cdot \left( (1-\gamma) p_0(s') + \gamma \sum_{s,a} \delta(s) \pi(a|s) \cdot P(s'|s,a) - \sum_{a'} \delta(s') \pi(a' \mid s')\right) \\
& + \sum_{s, a} \eta(s,a) \cdot \delta(s) \pi(a \mid s)
\end{align*}
$$


If we differentiate wrt $\delta(s) \pi(a \mid s)$

$$
\newcommand{\ggg}[0]{\frac{\partial \mathcal{L}}{\partial \delta(s^*) \pi(a^* \mid s^*)}}
\begin{align*}
\nabla \mathcal{L}
=\,
 \nabla\Biggr[
& \sum_{s,a} r(s,a) \cdot \delta(s) \pi(a \mid s) \\
& + \sum_{s'} \lambda(s') \cdot \left( (1-\gamma) p_0(s') + \gamma \sum_{s,a} \delta(s) \pi(a|s) \cdot P(s'|s,a) - \sum_{a'} \delta(s') \pi(a' \mid s')\right) \\
& + \sum_{s, a} \eta(s,a) \cdot \delta(s) \pi(a \mid s) \Biggr] \\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
=\,
& r(s^*, a^*) \\
& + \gamma \sum_{s'} \lambda(s') \cdot P(s'|s^*, a^*)
- \lambda(s^*) \\
& + \eta(s^*, a^*)
\end{align*}
$$


In solving for $\ggg = 0$, we notice that $\eta(s,a)$ is a just a slack
variable!  Thus, we can treat this equation as an inequality constraint,

$$
\begin{eqnarray*}
\nabla_{\delta(s^*) \pi(a^* \mid s^*)} \mathcal{L} &=& 0  \\
&\Leftrightarrow& \\
\lambda(s^*)
&\ge& r(s^*, a^*) + \gamma \sum_{s'} \lambda(s') \cdot P(s'|s^*, a^*)\quad\text{for all }s^*, a^*.
\end{eqnarray*}
$$

This is precisely the set of constraints in the optimal value function problem.


The dual problem is min $\lambda$


As usual, setting the derivative with respect to the a Lagrange multiplier&mdash;in our case $\lambda&mdash;to zero results in a condition which says that the original equallity constraint should be satisfied, that is

$$
\begin{eqnarray*}
\nabla_{\lambda(s^*)} \mathcal{L} &=& 0  \\
&\Leftrightarrow& \\
\sum_{a'} \delta(s^*) \pi(a' \mid s^*)
&=& (1-\gamma) p_0(s^*) + \gamma \sum_{s,a} \delta(s) \pi(a \mid s) \cdot P(s^*|s,a)
\end{eqnarray*}
$$

**TODO** saddle-point problems and optimization instability, best-response and
  positive response; regularization.


### The Lagrangian dual problem

The Lagrange dual problem simplifies to

$$
\begin{cases}
\underset{\lambda}{\textrm{minimize }} & \sum_{s} p_0(s) \cdot \lambda(s) \\
\text{subject to } &
\lambda(s) \ge r(s, a)
+ \gamma \sum_{s'} P(s'|s,a) \lambda(s') \quad\text{for all }s \in \mathcal{S}, a \in \mathcal{A}
\end{cases}
$$


**TODO** The S-procedure might fit in here.  The Lagrange dual of the QP
  formulation has a no duality gap, which is the main consequence of the
  S-procedure in control theory.  The reason why this is true is because once we
  take the Lagrange dual the problem can be simplified into another LP.  This is
  pretty trivial in our case because the original problem is sort of an LP too.


### Implicit functions

Much like in consumer theory of economics, the optimization formulation of the
MDP problem has rich connections between variables in the primal and dual.
Theses connections are meaningful (e.g., as expectations and/or partial
derivatives) and interconnected (e.g., as implicit functions of one another).
In the case of consumer theory, almost every connection is associated with a
theorem named after a Nobel Laureate.

In this section, we sketch a number of connections in the MDP setting here.

**TODO** how do implicit functions work in te the primal-dual case?  I think the
answer is that any partial optimization create constraints (as usual),
regardless of whether or not those constraints are in the primal or the dual,
they are coupled by the Lagrangian. The generally the primal-dual coupling is
"modulated" by Lagrange multiplier estimates.

#### Occupancy measure: $\delta$ as an implicit function of $\pi$:

If we fix $\pi$ and solve for a primal feasible $\delta$, then we can interpret
$\delta$ as $\pi$'s occupancy measure.[^occupancy-note]

$$
\delta_\pi
$$

[^occupancy-note]:
  For the average reward case, we use $P(s' \mid s, a) = p(s' \mid s, a)$.
  We may run into issues with feasibility if the transition function is not unichain (ergodic for all policies).
  For discounted case use $P(s' \mid s, a) = (1-\gamma) \cdot p_0(s') + \gamma \cdot p(s' \mid s,a)$.
  In the discounted cases, we can think of $\delta$ as $\pi$'s stationary distribution if we regard $(1-\gamma)$ as the probability of restarting the Markov chain (i.e., sampling the next state from a mixture of $p_0(s')$ with probability $(1-\gamma)$ and from $p(s' \mid s,a)$ with probability $\gamma$).  This is an equivalent in expected reward, but not higher-order moments of reward, i.e., it's not equivalent in distribution.


#### Successor representation?

```python
def successor_representation(self):
    "Dayan's successor representation."
    return linalg.solve(np.eye(self.S) - self.gamma * self.P,
                        np.eye(self.S))
```

**TODO** featurized version of SR.



#### Value function: $V$ as an implicit function of $\pi$:

For a fixed policy, if we set $\nabla_{???} \mathcal{L} = 0$,
the Lagrange mulipliers associated with constraint $(\ref{eq:balance})$ become
another implicit function $V_\pi$ of the policy

$$
V_\pi
$$



## TODO

**TODO** Formulate the Lagrangian.  Take its derivatives.  Talk about all the
  implicit functions and connections.

**TODO** Take the LP dual, note that it is exactly the optimal value function
  problem.

**TODO** featurized equilibrium distribution

**TODO** S-procedure and nonconvexity.

**TODO** [Mates of costate](http://www.argmin.net/2016/05/18/mates-of-costate/)

**TODO** if we write an equivalent set of constraints, what happen to the dual
  variables (e.g., do we get Q or A under different variations)?  Or are these
  other things just different implicit functions?

**TODO** Block-coordinate method - is policy iteration.  Just like my backprop
  post, we can take any fixed policy and estimate the Lagrange mulipliers by
  setting the gradient of the Lagrangian wrt $V$ equal to zero given what every
  $\pi$ is.

**TODO** The successor representation is the dual of this connection -- They are
  Lagrange multiplier estimates of the in the primal problem.

**TODO** Value function polytope
  https://arxiv.org/pdf/1901.11524.pdf

**TODO** Consider adding entropic regularization to the discussion.  At the very
  least add references to Belousov & Peters. (2017)[^BP17]

**TODO** The averge-reward formulation of V/Q functions are a little
  different.  Figure out why.

  Q(s,a) = R(s,a) - Rbar + \sum_{s'} T(s'|s,a) V(s')

  (Is the Rbar because of the sum-to-one constraint?)

**TODO** the diagram in this consumer theory tutorial is great.
  https://policonomics.com/marshallian-hicksian-demand-curves/

**TODO** Optimization-based view of MDP [^KBP13] [^BP17] [^W08]

**TODO** the "line theorem" in VF-polytope extends to all factored probability
   model (PGMs) and case-factor diagrams (CFDs) more generally.  The laws of
   probability are such that all probabilistic statements are multi-linear
   polynomials (MPs): the basic operations are chain-rule decompositions
   (times), marginalization (sum), and Bayes rule (div); additionally, it is
   never ok to multiply p(A, ... | ...) * p(A, ... | ...).  I am not sure how
   Bayes rule works yet, but it is definitely the case that restricting to the
   semiring (times-sum) part results in MPs.  I think that CFDs don't allow
   division.  They do allow case statements unlike PGMs.

## Extensions

### Average reward

<div style="border: thin solid black; padding: 10px; background-color: #ffffcc;
margin-bottom: 1.5em;">

The average-reward formulation
$$
U(\pi) \overset{\text{def}}{=} \lim_{T \rightarrow \infty} \frac{1}{T} \mathbb{E}\left[  \sum_{t=0}^T r(s,a) \right]
$$

The follow mathematical program formalizes the average reward optimization
problem.  The main difference is the balance equation is slightly different and
we now require an explicit sum-to-one constraint that we didn't need in the
discounted case.  Also the $1/(1-\gamma)$ is gone from the objective function.

$$
\begin{cases}
\underset{\pi, \delta}{\textrm{maximize }} & \sum_{s,a} r(s,a) \cdot \delta(s) \pi(a|s) \\
\text{subject to } &
\sum_{a'} \delta(s') \pi(a' \mid s') = \sum_{s,a} \delta(s) \pi(a|s) \cdot P(s'|s,a)\quad\text{for all }s' \in \mathcal{S} \\
& \sum_{s,a} \delta(s) \pi(a \mid s) = 1 \\
& \delta(s) \pi(a \mid s) \ge 0 \quad\text{for } s \in \mathcal{S}, a \in \mathcal{A}
\end{cases}
$$

Note that all of the implicit functions for the average reward case are slightly
different.  The reason why is the extra sum-to-one constraint!

</div>


## Dynamic programming inference in graphical models

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
generally Darwiche (2003).[^D03]


[^P13]:
  Kober, Bagnell, & Peters. 2013
  [Reinforcement learning in robotics: A survey](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.910.7004&rep=rep1&type=pdf)

[^BP17]:
  Belousov & Peters. 2017.
  [f-Divergence constrained policy improvement](https://arxiv.org/abs/1801.00056)

[^W08]:
  Wang, Lizotte, Bowling, & Schuurmans. 2008.
  [Dual Representations for Dynamic Programming](https://webdocs.cs.ualberta.ca/~dale/papers/dualdp.pdf)

[^LM04]:
  Jonathan Levin and Paul Milgrom. 2004.
  [Consumer Theory](https://web.stanford.edu/~jdlevin/Econ%20202/Consumer%20Theory.pdf)

[^D03]:
  Adnan Darwiche.
  [A differential approach to inference in Bayesian networks](https://dl.acm.org/citation.cfm?id=765570).
  Journal of the Association for Computing Machinery, 50(3):280–305, 2003.
