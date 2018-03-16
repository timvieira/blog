title: Value functions are also Lagrange multipliers
date: 2018-03-16
comments: true
tags: rl, calculus
status: draft

Value functions are an essential concept in reinforce learning and optimial
control. However, we take value functions to simply be "useful definitions." In
this post, I show how value functions arise mathematically in the policy search
problem as Lagrange multiplers. The connection is pretty interesting and is
closely related to my previous post on
[the connection between backpropagation and the chain rule](http://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/).

I will also point out value functions are not limited to RL, they are arise in
virtually all problems that have dynamic programming solutions.

The policy search problem:

$$
\underset{\pi}{\textrm{maximize }} \sum_{s,a} r(s,a) \mu(s) \pi(a|s)
$$

subject to

 - $\pi$ is a valid conditional probability distribution

$$
\begin{cases}
\textstyle\sum_a \pi(a | s) = 1 \quad\text{for all } s \in S \\
\pi(a | s) \ge 0 \quad\text{for all } s \in S, a \in A
\end{cases}
$$

 - $\mu$ is a valid stationary distribution for $\pi$
$$
\mu(s') = \sum_{s,a} \mu(s) \pi(a|s) p(s'|s,a)\quad\text{for all }s' \in S
$$

In matrix form, this is

$$
\mu = P_\pi \mu
$$

We can write out the Lagrangian for this optimization problem

$$
\mathcal{L}(\mu, \pi, \lambda, \sigma) =
\sum_{s,a} r(s,a) \mu(s) \pi(a|s)
+ \sum_{s'} \lambda(s') (\mu(s') - \sum_{s,a} \mu(s) \pi(a|s) p(s'|s,a))
+ \sum_s \sigma(s) (1 - \sum_a \pi(a | s))
$$

TODO: what about the position pi inequality?

Solving for $\lambda$ such that $\frac{\partial \mathcal{L}}{\partial \mu} = 0$
we get

$$
\begin{eqnarray}
\frac{\partial \mathcal{L}}{\partial \mu(s^*)} &=&
\sum_a r(s^*,a) \pi(a|s^*)
+ D_{\mu(s^*)}\left[ \sum_{s'} \lambda(s') (\mu(s') - \sum_{s,a} \mu(s) \pi(a|s) p(s'|s,a)) \right] \\
\end{eqnarray}
$$
