title: Gradient-based hyperparameter optimization and the implicit function theorem
date: 2016-03-05
comments: true
tags: calculus, hyperparameter-optimization

The most approaches to hyperparameter optimization can be viewed as a bi-level
optimization&mdash;the "inner" optimization optimizes training loss (wrt $\theta$),
while the "outer" optimizes hyperparameters ($\lambda$).

$$
\lambda^* = \underset{\lambda}{\textbf{argmin}}\
\mathcal{L}_{\text{dev}}\left(
\underset{\theta}{\textbf{argmin}}\
\mathcal{L}_{\text{train}}(\theta, \lambda) \right)
$$

Can we estimate $\frac{\partial \mathcal{L}_{\text{dev}}}{\partial \lambda}$ so
that we can run gradient-based optimization over $\lambda$?

Well, what does it mean to have an $\textbf{argmin}$ inside a function?

Well, it means that there is a $\theta^*$ that gets passed to
$\mathcal{L}_{\text{dev}}$. And, $\theta^*$ is a function of $\lambda$, denoted
$\theta(\lambda)$. Furthermore, $\textbf{argmin}$ must set the derivative of the
inner optimization is zero in order to be a local optimum of the inner
function. So we can rephrase the problem as

$$
\lambda^* = \underset{\lambda}{\textbf{argmin}}\
\mathcal{L}_{\text{dev}}\left(\theta(\lambda) \right),
$$
where $\theta(\lambda)$ is the solution to,
$$
\frac{\partial \mathcal{L}_{\text{train}}(\theta, \lambda)}{\partial \theta} = 0.
$$

Now how does $\theta$ change as the result of an infinitesimal change to
$\lambda$?

The constraint on the derivative implies a type of "equilibrium"&mdash;the inner
optimization process will continue to optimize regardless of how we change
$\lambda$. Assuming we don't change $\lambda$ too much, then the inner
optimization shouldn't change $\theta$ too much and it will change in a
predictable way.

To do this, we'll appeal to the implicit function theorem. Let's looking the
general case to simplify notation. Suppose $x$ and $y$ are related through a
function $g$ as follows,

$$g(x,y) = 0.$$

Assuming $g$ is a smooth function in $x$ and $y$, we can perturb either
argument, say $x$ by a small amount $\Delta_x$ and $y$ by $\Delta_y$. Because
system preserves the constraint, i.e.,

$$
g(x + \Delta_x, y + \Delta_y) = 0.
$$

We can solve for the change of $x$ as a result of an infinitesimal change in
$y$. We take the first-order expansion,

$$
g(x, y) + \Delta_x \frac{\partial g}{\partial x} + \Delta_y \frac{\partial g}{\partial y} = 0.
$$

Since $g(x,y)$ is already zero,

$$
\Delta_x \frac{\partial g}{\partial x} + \Delta_y \frac{\partial g}{\partial y} = 0.
$$

Next, we solve for $\frac{\Delta_x}{\Delta_y}$.

$$
\Delta_x \frac{\partial g}{\partial x} = - \Delta_y \frac{\partial g}{\partial y}.
$$


$$
\frac{\Delta_x}{\Delta_y}  = -\left( \frac{\partial g}{\partial y} \right)^{-1} \frac{\partial g}{\partial x}.
$$

Back to the original problem: Now we can use the implicit function theorem to
estimate how $\theta$ varies in $\lambda$ by plugging in $g \mapsto
\frac{\partial \mathcal{L}_{\text{train}}}{\partial \theta}$, $x \mapsto \theta$
and $y \mapsto \lambda$:

$$
\frac{\partial \theta}{\partial \lambda} = - \left( \frac{ \partial^2 \mathcal{L}_{\text{train}} }{ \partial \theta\, \partial \theta^\top } \right)^{-1} \frac{ \partial^2 \mathcal{L}_{\text{train}} }{ \partial \theta\, \partial \lambda^\top}
$$

This tells us how $\theta$ changes with respect to an infinitesimal change to
$\lambda$. Now, we can apply the chain rule to get the gradient of the whole
optimization problem wrt $\lambda$,

$$
\frac{\partial \mathcal{L}_{\text{dev}}}{\partial \lambda}
= \frac{\partial \mathcal{L}_{\text{dev}}}{\partial \theta} \left( - \left( \frac{ \partial^2 \mathcal{L}_{\text{train}} }{ \partial \theta\, \partial \theta^\top } \right)^{-1} \frac{ \partial^2 \mathcal{L}_{\text{train}} }{ \partial \theta\, \partial \lambda^\top} \right)
$$

Since we don't like (explicit) matrix inverses, we compute $- \left( \frac{
\partial^2 \mathcal{L}_{\text{train}} }{ \partial \theta\, \partial \theta^\top
} \right)^{-1} \frac{ \partial^2 \mathcal{L}_{\text{train}} }{ \partial \theta\,
\partial \lambda^\top}$ as the solution to $\left( \frac{ \partial^2
\mathcal{L}_{\text{train}} }{ \partial \theta\, \partial \theta^\top } \right) x
= -\frac{ \partial^2 \mathcal{L}_{\text{train}}}{ \partial \theta\, \partial
\lambda^\top}$. When the Hessian is positive definite, the linear system can be
solved with conjugate gradient, which conveniently only requires matrix-vector
products&mdash;i.e., you never have to materialize the Hessian. (Apparently,
[matrix-free linear algebra](https://en.wikipedia.org/wiki/Matrix-free_methods)
is a thing.) In fact, you don't even have to implement the Hessian-vector and
Jacobian-vector products because they are accurately and efficiently
approximated with centered differences (see
[earlier post](/blog/post/2014/02/10/gradient-vector-product/)).

At the end of the day, this is an easy algorithm to implement! However, the
estimate of the gradient can be temperamental if the linear system is
ill-conditioned.

In a later post, I'll describe a more-robust algorithms based on automatic
differentiation through the inner optimization algorithm, which make fewer and
less-brittle assumptions about the inner optimization.

**Further reading**:

 - [Truncated Bi-Level Optimization](https://justindomke.wordpress.com/2014/02/03/truncated-bi-level-optimization/)

 - [Efficient multiple hyperparameter learning for log-linear models](http://ai.stanford.edu/~chuongdo/papers/learn_reg.pdf)

 - [Gradient-based Hyperparameter Optimization through Reversible Learning](http://arxiv.org/abs/1502.03492)

 - [Hyperparameter optimization with approximate gradient](http://fa.bianp.net/blog/2016/hyperparameter-optimization-with-approximate-gradient/)
   ([paper](https://arxiv.org/pdf/1602.02355.pdf)): This paper looks at the implicit
   differentiation approach where you have an *approximate*
   solution to the inner optimization problem. They are able to provide error bounds and
   convergence guarantees under some reasonable conditions.
