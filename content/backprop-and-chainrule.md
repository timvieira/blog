title: Automatic differentiation is not just the chain rule
date: 2017-08-15
comments: true
tags: automatic-differentiation

I've heard a million people say that "automatic differentiation is just the
chain rule." Although that's basically true, there are some subtle things about
autodiff that will not be appreciated with this simplistic view.

I have ranted before that people do no understand some simple facts about
autodiff:

1. Evaluating $f(x)$ is provably as fast as evaluating $\nabla f(x)$.

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
\frac{\partial f}{\partial x} = \exp(\exp(x)+\exp(x)^2)(\exp(x)+2 \exp(x)^2) + \cos(\exp(x)+\exp(x)^2)(\exp(x)+2\cdot\exp(x)^2).
$$

However, this expression leaves something to be desired because it has lot of
repeated evaluations of the same function. This is clearly bad, if we want to
turn it into source code.

Going back to $f$ for a minute. If we were writing *a program* to compute $f$,
we'd definitely take advantage of the fact that it has a lot of duplicate
function evaluations. Here's a program which defines $f(x)$:

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
derivative has exactly the same structure*, but in reverse. Having exactly the
same structure means that we get the same runtime (up to some constants
factors).

You might hope that something like common subexpression elimination would
salvage the symbolic approach. Indeed that could be leveraged to improve any
chunk of code, but to match efficiency it's not needed!

So how the heck do we take derivatives of programs? Well, it's basically the
same as with ordinary gradients, we apply the chain rule *locally* to each edge
in the computation graph (instead of globally to the entire expression). There
is a simple rule to stich together gradients:

1. When you use a variable more than once in a program, its adjoints add.

2. When you work backwards you multiply like a unit conversion (should be
  familiar from dimensional analysis, right?)

Why is this correct? Correctness can be checked inductively in a similar manner
to proving the correctness of a dynamic programming algorithm. I won't go
through that, instead I'll describe an interesting connection that I stubled
upon in a few places in the literature (CITATIONS?).

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


It turns out that the Lagrange multipliers will be exactly the ajoints in the
backprop algorithm!  (This is an easy exercise to work out.)

The Lagrange multipliers form a simple linear system of equations. Furthermore,
this system is very efficient to solve by back-substitution, which is *exactly*
what the backpropagation algorithm is doing. The reason why the system is easy
is because there are no cyclic dependencies among the variables because the
variables are related by a DAG.

It's nice that we don't need something like Guassian elimination to solve that
linear system. But, this connection is interesting: It tells us that we could,
in fact, compute gradients with cyclic graphs; all we need is to run a linear
system solver to stich together our gradients (instead of the two rules from
before). Of course, cyclic gradients will increase runtime because solving a
*general* linear systems is cubic time, not linear.

You've probably see the implicit function theorem before in calculus. I
mentioned it before on this blog for use in
[gradient based hyperparameter optimization](http://timvieira.github.io/blog/post/2016/03/05/gradient-based-hyperparameter-optimization-and-the-implicit-function-theorem/). This
extra expressivity in our constraint language is clearly powerful and it's
interesting that we can still efficiently compute gradients in this setting.

In addition to handling cycles, we can also imagine using more general
algorithms for optimizing our function. We can see immediately that we could run
optimization with adjoints set to values other than those that backprop would
set them to (we can optimize them like we'd do in other algorithms for
optimizing Langrangians).
