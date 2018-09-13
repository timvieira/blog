title: Value functions are Lagrange multiplier estimates
date: 2018-03-16
comments: true
tags: rl, calculus
status: draft

Value functions are an essential concept in reinforcement learning and optimal
control.

In this post, I show how value functions arise mathematically in the policy
search problem as Lagrange multipler estimates. The connection is pretty cool
and is closely related to my previous post on
[the connection between backpropagation and the chain rule](http://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/).

The interpretation of "value functions as Lagrange multiplier estimates" should
be more productive than "value functions are just useful definitions" view.

I will also point out that the concept of a "value function" is not limited to
RL, they are arise in virtually all problems that have dynamic programming
solutions.

**The policy search problem:**

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

TODO: Show that this isa linear program... even though it might not look like
one. I think the crux is to realize that $\mu$ can be "collapsed out"&mdash;for
any setting of $\pi$ we can get by $\mu=(I - P \pi)^{-1}$. Where $P
\underset{sas',sa\rightarrow s s'}{\otimes} \pi$ is effectively just a
transition matrix.

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

TODO: what about the positive-$\pi$ inequality? (Well, we know for LP's we don't
need to worry about it. But we probably do for the basic Lagrangian.)

TODO: What does it mean to be an *estimate* -- it could mean a few things. I'm
going to take it to mean any consistent setting of V, given a fixed setting to
the policy. I.e., it satisfies the first Bellman equation but not the second.

(First and foremost, people (including myself) we often misuse the term
"Lagrange multiplier" when we often mean "Lagrange multiper *estimates*". A
Lagrange multiper estimate is only a Lagrange multiplier at convergence. CHECK.)


Solving for $\lambda$ such that $\frac{\partial \mathcal{L}}{\partial \mu} = 0$
we get

$$
\begin{eqnarray}
\frac{\partial \mathcal{L}}{\partial \mu(s^*)} &=&
\sum_a r(s^*,a) \pi(a|s^*)
+ D_{\mu(s^*)}\left[ \sum_{s'} \lambda(s') (\mu(s') - \sum_{s,a} \mu(s) \pi(a|s) p(s'|s,a)) \right] \\
\end{eqnarray}
$$
