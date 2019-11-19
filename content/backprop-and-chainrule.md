title: Backprop is not just the chain rule
date: 2017-08-18
comments: true
tags: calculus, automatic-differentiation, implicit-function-theorem, Lagrange-multipliers

Almost everyone I know says that "backprop is just the chain rule." Although
that's *basically true*, there are some subtle and beautiful things about
automatic differentiation techniques (including backprop) that will not be
appreciated with this *dismissive* attitude.

This leads to a poor understanding. As
[I have ranted before](http://timvieira.github.io/blog/post/2016/09/25/evaluating-fx-is-as-fast-as-fx/):
people do not understand basic facts about autodiff.

* Evaluating $\nabla f(x)$ is provably as fast as evaluating $f(x)$.

<!-- Let that sink in. Computing the gradient&mdash;an essential ingredient to
efficient optimization&mdash;is no slower to compute than the function
itself. Contrast that with the finite-difference gradient approximation, which
is quite accurate, but its runtime is $\textrm{dim}(x)$ times slower than
evaluating $f$
([discussed here](http://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/))!
-->

* Code for $\nabla f(x)$ can be derived by a rote program transformation, even
  if the code has control flow structures like loops and intermediate variables
  (as long as the control flow is independent of $x$). You can even do this
  "automatic" transformation by hand!


### Autodiff $\ne$ what you learned in calculus

Let's try to understand the difference between autodiff and the type of
differentiation that you learned in calculus, which is called *symbolic*
differentiation.

I'm going to use an example from
[Justin Domke's notes](https://people.cs.umass.edu/~domke/courses/sml2011/08autodiff_nnets.pdf),
$$
f(x) = \exp(\exp(x) + \exp(x)^2) + \sin(\exp(x) + \exp(x)^2).
$$

<!--
If we plug-and-chug with the chain rule, we get a correct expression for the
derivative,
$$\small
\frac{\partial f}{\partial x} =
\exp(\exp(x) + \exp(x)^2) (\exp(x) + 2 \exp(x) \exp(x)) \\
\quad\quad\small+ \cos(\exp(x) + \exp(x)^2) (\exp(x) + 2 \exp(x) \exp(x)).
$$

However, this expression leaves something to be desired because it has a lot of
repeated evaluations of the same function. This is clearly bad, if we want to
turn it into source code.
-->

If we were writing *a program* (e.g., in Python) to compute $f$, we'd take
advantage of the fact that it has a lot of repeated evaluations for efficiency.

```python
def f(x):
    a = exp(x)
    b = a**2
    c = a + b
    d = exp(c)
    e = sin(c)
    return d + e
```

Symbolic differentiation would have to use the "flat" version of this function,
so no intermediate variable $\Rightarrow$ slow.

Automatic differentiation lets us differentiate a program with *intermediate*
variables.

* The rules for transforming the code for a function into code for the gradient
  are really minimal (fewer things to memorize!). Additionally, the rules are
  more general than in symbolic case because they handle as a superset of
  programs.

* Quite [beautifully](http://conal.net/papers/beautiful-differentiation/), the
  program for the gradient *has exactly the same structure* as the function,
  which implies that we get the same runtime (up to some constants factors).

I won't give the details of how to execute the backpropagation transform to the
program. You can get that from
[Justin Domke's notes](https://people.cs.umass.edu/~domke/courses/sml2011/08autodiff_nnets.pdf)
and many other good
resources. [Here's some code](https://gist.github.com/timvieira/39e27756e1226c2dbd6c36e83b648ec2)
that I wrote that accompanies to the ``f(x)`` example, which has a bunch of
comments describing the manual "automatic" differentiation process on ``f(x)``.


<!--
Caveat: You might have seen some *limited* cases where an input variable was
reused, but chances are that it was something really simple like multiplication
or division, e.g., $\nabla\! \left[ f(x) \cdot g(x) \right] = f(x) \cdot g'(x)
+ f'(x) \cdot g(x)$, and you just memorized a rule. The rules of autodiff are
simpler and actually explains why there is a sum in the product rule. You can
also rederive the quotient rule without a hitch. I'm all about having fewer
things to memorize!
-->

<!--
You might hope that something like common subexpression elimination would save
the symbolic approach. Indeed that could be leveraged to improve any chunk of
code, but to match efficiency it's not needed! If we had needed to blow up the
computation to then shrink it down that would be much less efficient! The "flat"
version of a program can be exponentially larger than a version with reuse.
-->

<!-- Only sort of related: think of the exponential blow up in converting a
Boolean expression from conjunctive normal form to and disjunction normal.  -->


## Autodiff by the method of Lagrange multipliers

Let's view the intermediate variables in our optimization problem as simple
equality constraints in an equivalent *constrained* optimization problem. It
turns out that the de facto method for handling constraints, the method Lagrange
multipliers, recovers *exactly* the adjoints (intermediate derivatives) in the
backprop algorithm!

Here's our example from earlier written in this constraint form:

$$
\begin{align*}
\underset{x}{\text{argmax}}\ & f \\
\text{s.t.} \quad
a &= \exp(x) \\
b &= a^2     \\
c &= a + b   \\
d &= \exp(c) \\
e &= \sin(c) \\
f &= d + e
\end{align*}
$$

#### The general formulation
\begin{align*}
  & \underset{\boldsymbol{x}}{\text{argmax}}\ z_n & \\
  & \text{s.t.}\quad z_i = x_i                          &\text{ for $1 \le i \le d$} \\
  & \phantom{\text{s.t.}}\quad z_i = f_i(z_{\alpha(i)}) &\text{ for $d < i \le n$} \\
  \end{align*}

The first set of constraint ($1, \ldots, d$) are a little silly. They are only
there to keep our formulation tidy. The variables in the program fall into three
categories:

* **input variables** ($\boldsymbol{x}$): $x_1, \ldots, x_d$

* **intermediate variables**: ($\boldsymbol{z}$): $z_i = f_i(z_{\alpha(i)})$ for
  $1 \le i \le n$, where $\alpha(i)$ is a list of indices from $\{1, \ldots,
  n-1\}$ and $z_{\alpha(i)}$ is the subvector of variables needed to evaluate
  $f_i(\cdot)$. Minor detail: take $f_{1:d}$ to be the identity function.

* **output variable** ($z_n$): We assume that our programs has a singled scalar
  output variable, $z_n$, which represents the quantity we'd like to maximize.

<!-- (It is possible to generalize this story to compute Jacobians of functions
with multivariate outputs by "scalarizing" the objective, e.g., multiply the
outputs by a vector. This Gives an efficient program for computing Jacobian
vector products that can be used to extra Jacobians.)  -->

The relation $\alpha$ is a
[dependency graph](https://en.wikipedia.org/wiki/Dependency_graph) among
variables. Thus, $\alpha(i)$ is the list of *incoming* edges to node $i$ and
$\beta(j) = \{ i: j \in \alpha(i) \}$ is the set of *outgoing* edges. For now,
we'll assume that the dependency graph given by $\alpha$ is ① acyclic: no $z_i$
can transitively depend on itself.  ② single-assignment: each $z_i$ appears on
the left-hand side of *exactly one* equation.  We'll discuss relaxing these
assumptions in <a href="#lagrange-backprop-generalization">§ Generalizations</a>.

The standard way to solve a constrained optimization is to use the method
Lagrange multipliers, which converts a *constrained* optimization problem into
an *unconstrained* problem with a few more variables $\boldsymbol{\lambda}$ (one
per $x_i$ constraint), called Lagrange multipliers.

#### The Lagrangian

To handle constraints, let's dig up a tool from our calculus class,
[the method of Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier),
which converts a *constrained* optimization problem into an *unconstrained*
one. The unconstrained version is called "the Lagrangian" of the constrained
problem. Here is its form for our task,

$$
\mathcal{L}\left(\boldsymbol{x}, \boldsymbol{z}, \boldsymbol{\lambda}\right)
= z_n - \sum_{i=1}^n \lambda_i \cdot \left( z_i - f_i(z_{\alpha(i)}) \right).
$$

Optimizing the Lagrangian amounts to solving the following nonlinear system of
equations, which give necessary, but not sufficient, conditions for optimality,

$$
\nabla \mathcal{L}\left(\boldsymbol{x}, \boldsymbol{z}, \boldsymbol{\lambda}\right) = 0.
$$

Let's look a little closer at the Lagrangian conditions by breaking up the
system of equations into salient parts, corresponding to which variable types
are affected.

**Intermediate variables** ($\boldsymbol{z}$): Optimizing the
multipliers&mdash;i.e., setting the gradient of Lagrangian
w.r.t. $\boldsymbol{\lambda}$ to zero&mdash;ensures that the constraints on
intermediate variables are satisfied.

$$
\begin{eqnarray*}
\nabla_{\! \lambda_i} \mathcal{L}
= z_i - f_i(z_{\alpha(i)}) = 0
\quad\Leftrightarrow\quad z_i = f_i(z_{\alpha(i)})
\end{eqnarray*}
$$

We can use forward propagation to satisfy these equations, which we may regard
as a block-coordinate step in the context of optimizing the $\mathcal{L}$.

<!--GENERALIZATION:
However, if they are cyclic dependencies we may need to
solve a nonlinear system of equations. (TODO: it's unclear what the more general
cyclic setting is. Perhaps I should having a running example of a cyclic program
and an acyclic program.)
-->

**Lagrange multipliers** ($\boldsymbol{\lambda}$, excluding $\lambda_n$):
  Setting the gradient of the $\mathcal{L}$ w.r.t. the intermediate variables
  equal to zeros tells us what to do with the intermediate multipliers.

\begin{eqnarray*}
0 &=& \nabla_{\! z_j} \mathcal{L} \\
&=& \nabla_{\! z_j}\! \left[ z_n - \sum_{i=1}^n \lambda_i \cdot \left( z_i - f_i(z_{\alpha(i)}) \right) \right] \\
&=& - \sum_{i=1}^n \lambda_i \nabla_{\! z_j}\! \left[ \left( z_i - f_i(z_{\alpha(i)}) \right) \right] \\
&=& - \left( \sum_{i=1}^n \lambda_i \nabla_{\! z_j}\! \left[ z_i \right] \right) + \left( \sum_{i=1}^n \lambda_i \nabla_{\! z_j}\! \left[ f_i(z_{\alpha(i)}) \right] \right) \\
&=& - \lambda_j + \sum_{i \in \beta(j)} \lambda_i \frac{\partial f_i(z_{\alpha(i)})}{\partial z_j} \\
&\Updownarrow& \\
\lambda_j &=& \sum_{i \in \beta(j)} \lambda_i \frac{\partial f_i(z_{\alpha(i)})}{\partial z_j} \\
\end{eqnarray*}

Clearly, $\frac{\partial f_i(z_{\alpha(i)})}{\partial z_j} = 0$ for $j \notin
\alpha(i)$, which is why the $\beta(j)$ notation came in handy. By assumption,
the local derivatives, $\frac{\partial f_i(z_{\alpha(i)})}{\partial z_j}$ for $j
\in \alpha(i)$, are easy to calculate&mdash;we don't even need the chain rule to
compute them because they are simple function applications without
composition. Similar to the equations for $\boldsymbol{z}$, solving this linear
system is another block-coordinate step.

*Key observation*: The last equation for $\lambda_j$ should look very familiar:
It is exactly the equation used in backpropagation! It says that we sum
$\lambda_i$ of nodes that immediately depend on $j$ where we scaled each
$\lambda_i$ by the derivative of the function that directly relates $i$ and
$j$. You should think of the scaling as a "unit conversion" from derivatives of
type $i$ to derivatives of type $j$.


**Output multiplier** ($\lambda_n$): Here we follow the same pattern as for
  intermediate multipliers.

$$
\begin{eqnarray*}
0 &=& \nabla_{\! z_n}\! \left[ z_n - \sum_{i=1}^n \lambda_i \cdot \left( z_i - f_i(z_{\alpha(i)}) \right) \right] &=& 1 - \lambda_n \\
 &\Updownarrow& \\
 \lambda_n &=& 1
\end{eqnarray*}
$$

**Input multipliers** $(\boldsymbol{\lambda}_{1:d})$: Our dummy constraints
  gives us $\boldsymbol{\lambda}_{1:d}$, which are conveniently equal to the
  gradient of the function we're optimizing:

$$
\nabla_{\!\boldsymbol{x}} f(\boldsymbol{x}) = \boldsymbol{\lambda}_{1:d}.
$$

Of course, this interpretation is only precise when ① the constraints are
satisfied ($\boldsymbol{z}$ equations) and ② the linear system on multipliers is
satisfied ($\boldsymbol{\lambda}$ equations).

**Input variables** ($\boldsymbol{x}$): Unfortunately, the there is no
  closed-form solution to how to set $\boldsymbol{x}$. For this we resort to
  something like gradient ascent. Conveniently, $\nabla_{\!\boldsymbol{x}}
  f(\boldsymbol{x}) = \boldsymbol{\lambda}_{1:d}$, which we can use to optimize
  $\boldsymbol{x}$!

<div id="lagrange-backprop-generalization"></div>
### Generalizations

We can think of these equations for $\boldsymbol{\lambda}$ as a simple *linear*
system of equations, which we are solving by back-substitution when we use the
backpropagation method. The reason why back-substitution is sufficient for the
linear system (i.e., we don't need a *full* linear system solver) is that the
dependency graph induced by the $\alpha$ relation is acyclic. If we had needed a
full linear system solver, the solution would take $\mathcal{O}(n^3)$ time
instead of linear time, seriously blowing-up our nice runtime!

This connection to linear systems is interesting: It tells us that we can
compute *global* gradients in cyclic graphs. All we'd need is to run a linear
system solver to stitch together *local* gradients! That is exactly what the
[implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem)
says!

Cyclic constraints add some expressive powerful to our "constraint language," and
it's interesting that we can still efficiently compute gradients in this
setting. An example of what a general type of cyclic constraint looks like is

$$
\begin{align*}
& \underset{\boldsymbol{x}}{\text{argmax}}\, z_n \\
& \text{s.t.}\quad g(\boldsymbol{z}) = \boldsymbol{0} \\
& \text{and}\quad \boldsymbol{z}_{1:d} = \boldsymbol{x}
\end{align*}
$$

where $g$ can be any smooth multivariate function of the intermediate variables!
Of course, allowing cyclic constraints comes at the cost of a more-difficult
analogue of "the forward pass" to satisfy the $\boldsymbol{z}$ equations (if we
want to keep it a block-coordinate step). The $\boldsymbol{\lambda}$ equations
are now a linear system that requires a linear solver (e.g., Gaussian
elimination).


Example use cases:

* Bi-level optimization: Solving an optimization problem with another one inside
  it. For example,
  [gradient-based hyperparameter optimization](http://timvieira.github.io/blog/post/2016/03/05/gradient-based-hyperparameter-optimization-and-the-implicit-function-theorem/)
  in machine learning. The implicit function theorem manages to get gradients of
  hyperparameters without needing to store any of the intermediate states of the
  optimization algorithm used in the inner optimization! This is a *huge* memory
  saver since direct backprop on the inner gradient descent algorithm would
  require caching all intermediate states. Yikes!

* Cyclic constraints are useful in many graph algorithms. For example, computing
  gradients of edge weights in a general finite-state machine or, similarly,
  computing the value function in a Markov decision process.


### Other methods for optimization?

The connection to Lagrangians brings tons of algorithms for constrained
optimization into the mix! We can imagine using more general algorithms for
optimizing our function and other ways of enforcing the constraints. We see
immediately that we could run optimization with adjoints set to values other
than those that backprop would set them to (i.e., we can optimize them like we'd
do in other algorithms for optimizing general Lagrangians).

## Summary

Backprop does not directly fall out of the rules for differentiation that you
learned in calculus (e.g., the chain rule).

* This is because it operates on a more general family of functions: *programs*
  which have *intermediate variables*. Supporting intermediate variables is
  crucial for implementing both functions and their gradients efficiently.

I described how we could use something we did learn from calculus 101, the
method of Lagrange multipliers, to support optimization with intermediate
variables.

* It turned out that backprop is a *particular instantiation* of the method of
  Lagrange multipliers, involving block-coordinate steps for solving for the
  intermediates and multipliers.

* I also described a neat generalization to support *cyclic* programs and I
  hinted at ideas for doing optimization a little differently, deviating from
  the de facto block-coordinate strategy.


<!--
<center>
![Levels of enlightenment](/blog/images/backprop-brain-meme.png)
</center>
-->

## Further reading

After working out the connection between backprop and the method of Lagrange
multipliers, I discovered following paper, which beat me to it. I don't think my
version is too redundant.

> Yann LeCun. (1988)
> [A Theoretical Framework from Back-Propagation](http://yann.lecun.com/exdb/publis/pdf/lecun-88.pdf).

Ben Recht has a great blog post that uses the implicit function theorem to
*derive* the method of Lagrange multipliers. He also touches on the connection
to backpropagation.

> Ben Recht. (2016)
> [Mechanics of Lagrangians](http://www.argmin.net/2016/05/31/mechanics-of-lagrangians/).

Tom Goldstein's group took the Lagrangian view of backprop and used it to design
an ADMM approach for optimizing neural nets. The ADMM approach
can run massively in parallel and can leverage highly optimized solvers for
subproblems. This work nicely demonstrates that understanding automatic
differentiation&mdash;in the broader sense that I described in this
post&mdash;facilitates the development of novel optimization algorithms. <!--
ADMM is based on a cool reformulation trick, which takes a *big* circuit and
breaks it up into several *small* circuits (subproblems), which are *decoupled*
from the big problem because each subproblem gets to freely tune its own *local*
version of the variables. There is, of course, a global equality constraint on
the decoupled variables so that we get a correct solution. The global equality
constraints iteratively bring the subproblems into agreement.-->

> Gavin Taylor, Ryan Burmeister, Zheng Xu, Bharat Singh, Ankit Patel, Tom Goldstein. (2018)
> [Training Neural Networks Without Gradients: A Scalable ADMM Approach](https://arxiv.org/abs/1605.02026).

The backpropagation algorithm can be cleanly generalized from values to
functionals!

> Alexander Grubb and J. Andrew Bagnell. (2010)
> [Boosted Backpropagation Learning for Training Deep Modular Networks](https://t.co/5OW5xBT4Y1).

## Code

I have coded up and tested the Lagrangian perspective on automatic
differentiation that I presented in this article. The code is available in this
[gist](https://gist.github.com/timvieira/8addcb81dd622b0108e0e7e06af74185).

<script src="https://gist.github.com/timvieira/8addcb81dd622b0108e0e7e06af74185.js"></script>
