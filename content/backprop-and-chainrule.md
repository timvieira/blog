title: Backprop is not just the chain rule
date: 2017-08-15
comments: true
tags: automatic-differentiation

Almost everyone I know says that "backprop is just the chain rule." Although
that's basically true, there are some subtle and beautiful things about
automatic differentiation (including backprop) that will not be appreciated with
this simplistic view.

I have ranted before that people do no understand some simple facts about
autodiff:

1. Evaluating $f(x)$ is provably as fast as evaluating $\nabla f(x)$
([see previous post](http://timvieira.github.io/blog/post/2016/09/25/evaluating-fx-is-as-fast-as-fx/)). Let
that sink in. It is basically the best news in the word, computing the gradient
&mdash;an essential incredient to efficient optimization&mdash; is no slower to
compute than the function. Imagine if it were slower, e.g., in proportion to the
dimensionality of $x$ as in finite-difference approximation.

<!--
Caveat:
   space complexity may be much worse
   ([previous post](http://timvieira.github.io/blog/post/2016/09/25/evaluating-fx-is-as-fast-as-fx/)). Importantly,
   we don't get a slow down proportional to the dimensionality, like we do with
   the finite-difference approximation
   ([discussed here](http://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/)).
-->

2. Code for $\nabla f(x)$ can be derived by a rote program transformation, even
   if the code has control flow structures like loops and intermediate variables
   (as long as the control flow is independent of $x$, i.e., the function is
   actually diffentiable).

Let's try to understand the difference between autodiff and the type of
differentiation that you learned in calculus, which is called *symbolic*
differentiation.

I'm going to use an example from
[Justin Domke's course notes](https://people.cs.umass.edu/~domke/courses/sml2011/08autodiff_nnets.pdf).
$$
f(x) = \exp(\exp(x) + \exp(x)^2) + \sin(\exp(x) + \exp(x)^2).
$$

If we plug-and-chug with the chain rule, we get a correct expression for the
derivative,
$$\small
\frac{\partial f}{\partial x} = \exp(\exp(x)+\exp(x)^2)(\exp(x)+2 \exp(x)^2) + \cos(\exp(x)+\exp(x)^2)(\exp(x)+2 \exp(x)^2).
$$

However, this expression leaves something to be desired because it has lot of
repeated evaluations of the same function. This is clearly bad, if we want to
turn it into source code.

Going back to $f$ for a minute. If we were writing *a program* (e.g., in Python)
to compute $f$, we'd definitely take advantage of the fact that it has a lot of
duplicate function evaluations. Here's a program which defines $f(x)$:

$$
\begin{eqnarray*}
a &=& \exp(x) \\
b &=& a^2     \\
c &=& a + b   \\
d &=& \exp(c) \\
e &=& \sin(c) \\
f &=& d + e
\end{eqnarray*}
$$

The beautiful thing about differentiation is that *the program for the
derivative has exactly the same structure*! It's most efficient if we evaluate
it in reverse (so called "reverse-mode" or "backpropagation"). Having exactly
the same structure means that we get the same runtime (up to some constants
factors).

You might hope that something like common subexpression elimination would save
the symbolic approach. Indeed that could be leveraged to improve any chunk of
code, but to match efficiency it's not needed! (If we had to blow up
computations then shrink it down that would be much less efficient! The "flat"
version of a program can be exponentially larger than a version with reuse.)

So how the heck do we take derivatives of programs? Well, it's basically the
same as with ordinary gradients, we apply the chain rule *locally* to each edge
in the computation graph (instead of globally to the entire expression). There
is a simple rule to stich together gradients:

1. When you use a variable more than once in a program, its adjoints add.

2. When you work backwards you multiply&mdash;like a unit conversion!
  $\frac{\partial\, \text{output}}{\partial\, \text{edge.input}} \texttt{ += } \frac{\partial\, \text{output}}{\partial\, \text{edge.output}} \cdot \frac{\partial\, \text{edge.output}}{\partial\, \text{edge.input}}$

Why is this correct? Correctness can be checked inductively in a similar manner
to proving the correctness of a dynamic programming algorithm. I won't go
through that, instead I'll describe an interesting connection that I stubled
upon in a few places in the literature (CITATIONS?). So rather than a computer
science type of explanation, this is a mathematical explanation based on basic
calculus.

## Autodiff by the method of Lagrange multipliers

Simply take the intermediate variables in our program to be equality constraints
in an equivalent *constrained* optimization problem.

$$
\begin{align*}
\underset{x}{\text{argmax}}\ & f \\
\text{s.t.} & \\
a &= \exp(x) \\
b &= a^2     \\
c &= a + b   \\
d &= \exp(c) \\
e &= \sin(c) \\
f &= d + e
\end{align*}
$$

It turns out that the Lagrange multipliers are *exactly* the adjoints in the
backprop algorithm!  (This is an easy exercise to work out.)

Each constraint is easy to differentiate&mdash;we don't even need the chain rule to
do it!

$$
\begin{align*}
\frac{\partial a}{\partial x} &= \exp(x) \\
\frac{\partial b}{\partial a} &= 2 a     \\
\frac{\partial c}{\partial a} &= 1   \\
\frac{\partial c}{\partial b} &= 1   \\
\frac{\partial d}{\partial c} &= \exp(c) \\
\frac{\partial e}{\partial c} &= \cos(c) \\
\frac{\partial f}{\partial d} &= 1 \\
\frac{\partial f}{\partial e} &= 1 \\
\frac{\partial f}{\partial f} &= 1
\end{align*}
$$

Setting the gradient of the Lagrange multiplers equal to zero enforces the
constraints, which is easy to do by evaluating the function in forward
mode. (You can think of this as a block coordinate step on the Lagrangian.)

The Lagrange multipliers form a simple linear system of equations.


The solution in the case of a DAG-structured computation graph specially
efficient because we only need to run back-substitution.

$$
\begin{align*}
% line 0
\lambda_f &= \frac{\partial f}{\partial f} \\
% line -1
\lambda_d &= \frac{\partial f}{\partial d} \cdot\lambda_f \\
\lambda_e &= \frac{\partial f}{\partial e} \cdot\lambda_f \\
% line -2
\lambda_c &+= \frac{\partial e}{\partial c} \cdot\lambda_e \\
% line -3
\lambda_c &+= \frac{\partial d}{\partial c} \cdot\lambda_d \\
% line -4
\lambda_a &+= \frac{\partial c}{\partial a} \cdot\lambda_c \\
\lambda_b &= \frac{\partial c}{\partial b} \cdot\lambda_c \\
% line -5
\lambda_a &+= \frac{\partial b}{\partial a} \cdot\lambda_b \\
% line -6
\lambda_x &= \frac{\partial a}{\partial x} \cdot\lambda_a \\
\end{align*}
$$

Furthermore, this system is very efficient to solve by back-substitution, which
is *exactly* what the backpropagation algorithm is doing. The reason why the
system is easy is because there are no cyclic dependencies among the variables
because the variables are related by a DAG.

It's nice that we don't need something like Guassian elimination to solve that
linear system. But, this connection is interesting: It tells us that we could,
in fact, compute gradients with cyclic graphs; all we need is to run a linear
system solver to stich together our gradients (instead of the two rules from
before). Of course, cyclic gradients will increase runtime because solving a
*general* linear systems is cubic time, not linear.

You've probably see the implicit function theorem before in calculus. I've
mentioned it before to do
[gradient-based hyperparameter optimization](http://timvieira.github.io/blog/post/2016/03/05/gradient-based-hyperparameter-optimization-and-the-implicit-function-theorem/). This
extra expressivity in our constraint language is clearly powerful and it's
interesting that we can still efficiently compute gradients in this setting.

In addition to handling cycles, we can also imagine using more general
algorithms for optimizing our function. We can see immediately that we could run
optimization with adjoints set to values other than those that backprop would
set them to (we can optimize them like we'd do in other algorithms for
optimizing Langrangians).
