title: Gradient of a product
date: 2015-07-29
comments: true
tags: math, calculus, numerical

$$
\newcommand{\gradx}[1]{\grad{x}{ #1 }}
\newcommand{\grad}[2]{\nabla_{\! #1}\! \left[ #2 \right]}
\newcommand{\R}{\mathbb{R}}
\newcommand{\bigo}[0]{\mathcal{O}}
$$

In this post we'll look at the gradient of product. This is such a common
subroutine in machine learning that it's worth careful consideration. In a later
post, I'll describe the gradient of a sum-over-products, which is another
interesting and common pattern in machine learning (e.g., exponential families,
CRFs, context-free grammar, case-factor diagrams, semiring-weighted logic
programming).

Given a collection of functions with a common argument $f_1, \cdots, f_n \in \{
\R^d \mapsto \R \}$.

Define their product $p(x) = \prod_{i=1}^n f_i(x)$

Suppose, we'd like to compute the gradient of the product of these functions
with respect to their common argument, $x$.

$$
\begin{eqnarray*}
\gradx{ p(x) }
&=& \gradx{ \prod_{i=1}^n f_i(x) }
&=& \sum_{i=1}^n \left( \gradx{f_i(x)} \prod_{i \ne j} f_j(x)  \right)
\end{eqnarray*}
$$

As you can see in the equation above, the gradient takes the form of a
"leave-one-out product" sometimes called a "cavity."

A naive method for computing the gradient computes the leave-out-out products
from scratch for each $i$ (outer loop)---resulting in a overall runtime of
$O(n^2)$ to compute the gradient. Later, we'll see a dynamic program for
computing this efficiently.

**Division trick**: Before going down the dynamic programming rabbit hole, let's
consider the following relatively simple method for computing the gradient,
which uses division:

$$
\begin{eqnarray*}
\gradx{ p(x) }
&=& \sum_{i=1}^n \left( \frac{\gradx{f_i(x)} }{ f_i(x) } \prod_{j=1}^n f_j(x) \right)
&=& \left( \sum_{i=1}^n \frac{\gradx{f_i(x)} }{ f_i(x) } \right) \left( \prod_{j=1}^n f_j(x) \right)
\end{eqnarray*}
$$

Pro:

 - Runtime $\bigo(n)$ with space $\bigo(1)$.

Con:

 - Requires $f \ne 0$. No worries, we can handle zeros with three cases: (1) If
   No zeros: the division trick works fine. (2) Only one zero: implies that only
   one term in the sum will have a nonzero gradient, which we compute via
   leave-one-out product. (3) Two or more zeros: all gradients are zero and
   there is no work to be done.

 - Requires multiplicative inverse operator (division) *and*
   associative-commutative multiplication, which means it's not applicable to
   matrices.


**Log trick**: Suppose $f_i$ are very small numbers (e.g., probabilities), which
we'd rather not multiply together because we'll quickly lose precision (e.g.,
for large $n$). It's common practice (especially in machine learning) to replace
$f_i$ with $\log f_i$, which turns products into sums, $\prod_{j=1}^n f_j(x) =
\exp \left( \sum_{j=1}^n \log f_j(x) \right)$, and tiny numbers (like
$\texttt{3.72e-44}$) into large ones (like $\texttt{-100}$).

Furthermore, using the identity $(\nabla g) = g \cdot \nabla \log g$, we can
operate exclusively in the "$\log$-domain".

$$
\begin{eqnarray*}
\gradx{ p(x) }
&=& \left( \sum_{i=1}^n \gradx{ \log f_i(x) } \right) \exp\left( \sum_{j=1}^n \log f_j(x) \right)
\end{eqnarray*}
$$

Pro:

 - Numerically stable

 - Runtime $\bigo(n)$ with space $\bigo(1)$.

Con:

 - Requires $f > 0$. But, we can use
   [LogReal number class](http://timvieira.github.io/blog/post/2015/02/01/log-real-number-class/)
   to represent negative numbers in log-space, but we still need to be careful
   about zeros (like in the division trick).

 - Doesn't easily generalize to other notions of multiplication.


**Dynamic programming trick**: $\bigo(n)$ runtime and $\bigo(n)$ space. You may
recognize this as forward-backward algorithm for linear chain CRFs
(cf. [Wallach (2004)](http://www.inference.phy.cam.ac.uk/hmw26/papers/crf_intro.pdf),
section 7).

The trick is very straightforward when you think about it in isolation. Compute
the products of all prefixes and suffixes. Then, multiply them together.

Here are the equations:

$$
\begin{eqnarray*}
\alpha_0(x) &=& 1 \\
\alpha_t(x)
   &=& \prod_{i \le t} f_i(x)
   = \alpha_{t-1}(x) \cdot f_t(x) \\
\beta_{n+1}(x) &=& 1 \\
\beta_t(x)
  &=& \prod_{i \ge t} f_i(x) = f_t(x) \cdot \beta_{t+1}(x)\\
\gradx{ p(x) }
&=& \sum_{i=1}^n \left( \prod_{j < i} f_j(x) \right) \gradx{f_i(x)} \left( \prod_{j > i} f_j(x) \right) \\
&=& \sum_{i=1}^n \alpha_{i-1}(x) \cdot \gradx{f_i(x)} \cdot \beta_{i+1}(x)
\end{eqnarray*}
$$

Clearly, this requires $O(n)$ additional space.

Only requires an associative operator (i.e., Does not require it to be
commutative or invertible like earlier strategies).

Why do we care about the non-commutative multiplication? A common example is
matrix multiplication where $A B C \ne B C A$, even if all matrices have the
conformable dimensions.

**Connections to automatic differentiation**: The theory behind reverse-mode
automatic differentiation says that if you can compute a function, then you
*can* compute it's gradient with the same asymptotic complexity, *but* you might
need more space. That's exactly what we did here: We started with a naive
algorithm for computing the gradient with $\bigo(n^2)$ time and $\bigo(1)$ space
(other than the space to store the $n$ functions) and ended up with a $\bigo(n)$
time $\bigo(n)$ space algorithm with a little clever thinking. What I'm saying
is autodiff---even if you don't use a magical package---tells us that an
efficient algorithm for the gradient always exists. Furthermore, it tells you
how to derive it manually, if you are so inclined. The key is to reuse
intermediate quantities (hence the increase in space).

*Sketch*: In the gradient-of-a-product case, assuming we implemented
multiplication left-to-right (forward pass) that already defines the prefix
products ($\alpha$). It turns out that the backward pass gives us $\beta$ as
adjoints. Lastly, we'd propagate gradients through the $f$'s to get
$\frac{\partial p}{\partial x}$. Essentially, we end up with exactly the dynamic
programming algorithm we came up with.
