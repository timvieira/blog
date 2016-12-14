title: Gradient-vector product
date: 2014-02-10
comments: true
tags: calculus


We've all written the following test for our gradient code (known as the
finite-difference approximation).

$$
\frac{\partial}{\partial x_i} f(\boldsymbol{x}) \approx
 \frac{1}{2 \varepsilon} \Big(
   f(\boldsymbol{x} + \varepsilon \cdot \boldsymbol{e_i})
 - f(\boldsymbol{x} - \varepsilon \cdot \boldsymbol{e_i})
 \Big)
$$

where $\varepsilon > 0$ and $\boldsymbol{e_i}$ is a vector of zeros except at
$i$ where it is $1$. This approximation is exact in the limit, and accurate to
$o(\varepsilon^2)$ additive error.

This is a specific instance of a more general approximation! The dot product of
the gradient and any (conformable) vector $\boldsymbol{d}$ can be approximated
with the following formula,

$$
\nabla f(\boldsymbol{x})^{\top} \boldsymbol{d} \approx
\frac{1}{2 \varepsilon} \Big(
   f(\boldsymbol{x} + \varepsilon \cdot \boldsymbol{d})
 - f(\boldsymbol{x} - \varepsilon \cdot \boldsymbol{d})
 \Big)
$$

We get the special case above when $\boldsymbol{d}=\boldsymbol{e_i}$. This also
exact in the limit and just as accurate.


**Runtime?** Finite-difference approximation is probably too slow for
  approximating a high-dimensional gradient because the number of function
  evaluations required is $2 n$ where $n$ is the dimensionality of $x$. However,
  if the end goal is to approximate a gradient-vector product, a mere $2$
  function evaluations is probably faster than specialized code for computing
  the gradient.

**How to set $\varepsilon$?** The second approach is more sensitive to
  $\varepsilon$ because $\boldsymbol{d}$ is arbitrary, unlike
  $\boldsymbol{e_i}$, which is a simple unit-norm vector. Luckily some guidance
  is available. Andrei (2009) reccommends

$$
\varepsilon = \sqrt{\epsilon_{\text{mach}}} (1 + \|\boldsymbol{x} \|_{\infty}) / \| \boldsymbol{d} \|_{\infty}
$$


where $\epsilon_{\text{mach}}$ is
[machine epsilon](http://en.wikipedia.org/wiki/Machine_epsilon). (Numpy users:
``numpy.finfo(x.dtype).eps``).


Why do I care?
--------------

1. Well, I tend to work on sparse, but high-dimensional problems where
   finite-difference would be too slow. Thus, my usual solution is to only test
   several randomly selected dimensions$-$biasing samples toward dimensions
   which should be nonzero. With the new trick, I can effectively test more
   dimensions at once by taking random vectors $\boldsymbol{d}$. I recommend
   sampling $\boldsymbol{d}$ from a spherical Gaussian so that we're uniform on
   the angle of the vector.

2. Sometimes the gradient-vector dot product is the end goal. This is the case
   with Hessian-vector products, which arises in many optimization algorithms,
   such as stochastic meta descent. Hessian-vector products are an instance of
   the gradient-vector dot product because the Hessian is just the gradient of
   the gradient.


Hessian-vector product
----------------------

Hessian-vector products are an instance of the gradient-vector dot product
because since the Hessian is just the gradient of the gradient! Now you only
need to remember one formula!

$$
H(\boldsymbol{x})\, \boldsymbol{d} \approx
\frac{1}{2 \varepsilon} \Big(
  \nabla f(\boldsymbol{x} + \varepsilon \cdot \boldsymbol{d})
- \nabla f(\boldsymbol{x} - \varepsilon \cdot \boldsymbol{d})
\Big)
$$

With this trick you never have to actually compute the gnarly Hessian! More on
[Justin Domke's blog](http://justindomke.wordpress.com/2009/01/17/hessian-vector-products/)
