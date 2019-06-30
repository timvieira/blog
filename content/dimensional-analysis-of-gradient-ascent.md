title: Dimensional analysis of gradient ascent
date: 2016-05-27
comments: true
tags: optimization, calculus

In physical sciences, numbers are paired with units and called quantities. In
this augmented number system, dimensional analysis provides a crucial sanity
check, much like type checking in a programming language. There are simple rules
for building up units and constraints on what operations are allowed. For
example, you can't multiply quantities which are not conformable or add
quantities with different units. Also, we generally know the units of the input
and desired output, which allows us to check that our computations at least
produce the right units.

In this post, we'll discuss the dimensional analysis of gradient ascent, which
will hopefully help us understand why the "step size" is parameter so finicky
and why it even exists.

Gradient ascent is an iterative procedure for (locally) maximizing a function,
$f: \mathbb{R}^d \mapsto \mathbb{R}$.

$$
x_{t+1} = x_t + \alpha \frac{\partial f(x_t)}{\partial x}
$$

In general, $\alpha$ is a $d \times d$ matrix, but often we constrain the matrix
to be simple, e.g., $a\cdot I$ for some scalar $a$ or $\text{diag}(a)$ for some
vector $a$.

Now, let's look at the units of the change in $\Delta x=x_{t+1} - x_t$,
$$
(\textbf{units }\Delta x) = \left(\textbf{units }\alpha\cdot \frac{\partial f(x_t)}{\partial x}\right) = (\textbf{units }\alpha) \frac{(\textbf{units }f)}{(\textbf{units }x)}.
$$

The units of $\Delta x$ must be $(\textbf{units }x)$. However, if we assume $f$
is unit free, we're happy with $(\textbf{units }x) / (\textbf{units }f)$.

Solving for the units of $\alpha$ we get,
$$
(\textbf{units }\alpha) = \frac{(\textbf{units }x)^2}{(\textbf{units }f)}.
$$

This gives us an idea for what $\alpha$ should be.

For example, the inverse Hessian passes the unit check (if we assume $f$ unit
free). The disadvantages of the Hessian is that it needs to be positive-definite
(or at least invertible) in order to be a valid "step size" (i.e., we need
step sizes to be $> 0$).

Another method for handling step sizes is line search. However, line search
won't let us run online. Furthermore, line search would be too slow in the case
where we want a step size for each dimension.

In machine learning, we've become fond of online methods, which adapt the step
size as they go. The general idea is to estimate a step size matrix that passes
the unit check (for each dimension of $x$). Furthermore, we want do as little
extra work as possible to get this estimate (e.g., we want to avoid computing a
Hessian because that would be extra work). So, the step size should be based
only iterates and gradients up to time $t$.

- [AdaGrad](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf) doesn't doesn't
  pass the unit check. This motivated AdaDelta.

- [AdaDelta](https://arxiv.org/abs/1212.5701) uses the ratio of (running
  estimates of) the root-mean-squares of $\Delta x$ and $\partial f / \partial
  x$. The mean is taken using an exponentially weighted moving average. See
  paper for actual implementation.

- [Adam](http://arxiv.org/abs/1412.6980) came later and made some tweaks to
  remove (unintended) bias in the AdaDelta estimates of the numerator and
  denominator.

In summary, it's important/useful to analyze the units of numerical algorithms
in order to get a sanity check (i.e., catch mistakes) as well as to develop an
understanding of why certain parameters exist and how properties of a problem
affect the values we should use for them.
