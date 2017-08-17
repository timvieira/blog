title: Backprop is not just the chain rule
date: 2017-08-15
comments: true
tags: automatic-differentiation
status: draft

Almost everyone I know says that "backprop is just the chain rule." Although
that's *basically true*, there are some subtle and beautiful things about
automatic differentiation (including backprop) that will not be appreciated with
this almost *dismissive* view.

This leads to a poor understanding. As I have ranted before: people do not
understand basic facts about autodiff.

1. Evaluating $f(x)$ is provably as fast as evaluating $\nabla f(x)$
   ([see previous post](http://timvieira.github.io/blog/post/2016/09/25/evaluating-fx-is-as-fast-as-fx/)). Let
   that sink in. Computing the gradient&mdash;an essential incredient to
   efficient optimization&mdash;is no slower to compute than the function
   itself. Contrast that with the finite-difference gradient approximation,
   which is quite accurate, but its runtime scales with the dimensionality of
   $x$ slower than evaluating $f$
   ([discussed here](http://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/)).

2. Code for $\nabla f(x)$ can be derived by a rote program transformation, even
   if the code has control flow structures like loops and intermediate variables
   (as long as the control flow is independent of $x$). You can even do this
   program transformation by hand!


## Autodiff $\ne$ what you learned in calculus

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

However, this expression leaves something to be desired because it has lot of
repeated evaluations of the same function. This is clearly bad, if we want to
turn it into source code.
-->

If we were writing *a program* (e.g., in Python) to compute $f$, we'd definitely
take advantage of the fact that it has a lot of repeated evaluations for
efficiency.

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

Automatic differentiation let's us differentiate a program with *intermediate*
variables.

* The rules for transforming the code for a function into code for the gradient
  are really minimal (fewer things to memorize!). Additionally, the rules are
  more general than in symbolic case because they handle as superset of
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
$$
\begin{align*}
&\textbf{def }f(x): \\
&\quad a = \exp(x) \\
&\quad b = a^2     \\
&\quad c = a + b   \\
&\quad d = \exp(c) \\
&\quad e = \sin(c) \\
&\quad f = d + e   \\
&\quad \textbf{return } f
\end{align*}
$$
-->


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
$$
\begin{align*}
& \textbf{return } f & \Rightarrow & \frac{df}{df} \texttt{ += } 1 \\
& f = d + e     & \Rightarrow & \frac{df}{dd} \texttt{ += } \frac{df}{df} \cdot \frac{df}{dd} ; \frac{df}{de} \texttt{ += } \frac{df}{df} \cdot \frac{df}{de} \\
& e = \sin(c)   & \Rightarrow & \frac{df}{dc} \texttt{ += } \frac{df}{de} \cdot \frac{de}{dc} \\
& d = \exp(c)   & \Rightarrow & \frac{df}{dc} \texttt{ += } \frac{df}{dd} \cdot \frac{dd}{dc} \\
& c = a + b     & \Rightarrow & \frac{df}{da} \texttt{ += } \frac{df}{dc} \cdot \frac{dc}{da} ; \frac{df}{db} \texttt{ += } \frac{df}{dc} \cdot \frac{dc}{db} \\
& b = a^2       & \Rightarrow & \frac{df}{da} \texttt{ += } \frac{df}{db} \cdot \frac{db}{da} \\
& a = \exp(x)   & \Rightarrow & \frac{df}{dx} \texttt{ += } \frac{df}{da} \cdot \frac{da}{dx}
\end{align*}
$$
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

<!--
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
-->

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


We can describe our programs in a general form

* **input variables** $x_1, \ldots, x_d$

* **intermediate variables** $z_i = f_i(z_{\alpha(i)})$ for $d < i < n$, where
  $\alpha(i)$ is a subset of indices $\{1, \ldots, n-1\}$ and $z_{\alpha(i)}$ is
  the subset of variables needed to evaluate $f_i(\cdot)$.

* **output variable** $z_n$ represents the quantity we'd like to maximize.

We can regard the relationship given by $\alpha$ as a dependency graph among
variables. Thus, $\alpha(i)$ is the set of *incoming* edges to node $i$ and
$\beta(j) = \{ i: j \in \alpha(i) \}$$ is the set of *outgoing* edges.

For now, we'll assume that the dependency graph given by $\alpha$ is (1)
acyclic: no $x_i$ can transitively depend on itself. (2) single-assignment: each
$x_i$ for $i > d$ appears on the left-hand side of exactly one equation. We'll
discuss relaxing these assumptions in the section "generalizations."

As a mathematical program, our optimization problem looks like this:
\begin{align*}
  & \underset{\boldsymbol{x}}{\text{argmax}}\ z_n \\
  & \text{s.t.}\quad z_i = f_i(z_{\alpha(i)}) \text{ for $d < i \le n$} \\
  & \text{and }\quad z_i = x_i \text{ for $1 \le i \le d$}
  \end{align*}

The second set of constraints are a little silly, but help keep our formulation
nice n' tidy.

The standard way to solve a constrained optimization is to use the method
Lagrange multipliers, which converts a *constrained* optimization problem into
an *unconstrained* problem with a few more variables $\boldsymbol{\lambda}$ (one
per $x_i$ variable), called Lagrange multipliers.

**The Lagrangian**: To handle constaints, let's dig up a tool from our calculus
class,
[the method of Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier),
which converts a *constrained* optimization probelm into an *unconstrainted*
one. The unconstrained version is called "the Lagrangian" of the constrained
problem. Here is its form for our task,

$$
\mathcal{L}\left(\boldsymbol{x}, \boldsymbol{z}, \boldsymbol{\lambda}\right)
= z_n - \sum_{i=d+1}^n \lambda_i \left( z_i - f_i(z_{\alpha(i)}) \right).
$$

Optimizing the Lagrangian amounts to solving the following nonlinear system of
equations, which give necessary, but not sufficient, conditions for optimality,

$$
\nabla \mathcal{L}\left(\boldsymbol{x}, \boldsymbol{z}, \boldsymbol{\lambda}\right) = 0.
$$

Let's look a little closer at the gradient of the Lagrangian system. by breaking
up the system into salient parts, corresponding to the different types of
variables: multipliers, output, intermediate and input variables.

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
&=& \nabla_{\! z_j}\! \left[ z_n - \sum_{i=d+1}^n \lambda_i \cdot \left( z_i - f_i(z_{\alpha(i)}) \right) \right] \\
&=& - \sum_{i=d+1}^n \lambda_i \cdot \nabla_{\! z_j}\! \left[ \left( z_i - f_i(z_{\alpha(i)}) \right) \right] \\
&=& - \left( \sum_{i=d+1}^n \lambda_i \nabla_{\! z_j}\! \left[ z_i \right] \right) + \left( \sum_{i=d+1}^n \lambda_i \nabla_{\! z_j}\! \left[ f_i(z_{\alpha(i)}) \right] \right) \\
&=& - \lambda_j + \sum_{i \in \beta(j)} \lambda_i \cdot \frac{\partial f_i(z_{\alpha(i)})}{\partial z_j} \\
&\Updownarrow& \\
\lambda_j &=& \sum_{i \in \beta(j)} \lambda_i \frac{\partial f_i(z_{\alpha(i)})}{\partial z_j} \\
\end{eqnarray*}

Clearly, $\frac{\partial f_i(z_{\alpha(i)})}{\partial z_j} = 0$ for $j \notin
\alpha(i)$, which is why the $\beta(j)$ notation came in handy. By assumption,
the local derivatives, $\frac{\partial f_i(z_{\alpha(i)})}{\partial z_j}$ for $j
\in \alpha(i)$, are easy to calculate&mdash;we don't even need the chain rule to
compute them because they are simple function applications without composition.

This final above equation should look familiar: It's exactly the equation used
in backpropagation! It's says that we sum $\lambda_i$ of nodes that immediately
depend on $j$ where we scaled each $\lambda_i$ by the derivative of the function
that directly relates $i$ and $j$. You should think of the scaling as a "unit
conversion" from derivatives of type $i$ to derivatives of type $j$.


**Output mutliplier** ($\lambda_n$): Here we follow the same pattern as for
  intermediate multipliers.

$$
\begin{eqnarray*}
0 &=& \nabla_{\! z_n}\! \left[ z_n - \sum_{i=d+1}^n \lambda_i \cdot \left( z_i - f_i(z_{\alpha(i)}) \right) \right] &=& 1 - \lambda_n \\
 &\Updownarrow& \\
 \lambda_n &=& 1
\end{eqnarray*}
$$

**Input multipliers** $(\boldsymbol{\lambda}_{1:d})$: Our dummy constraints
  gives us $\boldsymbol{\lambda}_{1:d}$, which are conveniently equal to the
  gradient of the function we're optimizing, assuming the constraints are
  satisfied and multipliers are optimized.

**Input variables** ($\boldsymbol{x}$) Unforunately the there is no closed-form
  solution to how to set $\boldsymbol{x}$. For this we resort to something like
  gradient ascent. Conveniently, the gradient of $\boldsymbol{x}$ is equal to
  $\boldsymbol{\lambda}_{1:d}$ so we can use that in our optimization!


<!--
<div class="example">
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
</div>
-->

<!--
<div class="example">
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
</div>
-->

## Generalizations

We can think of these equations for $\boldsymbol{\lambda}$ as a simple *linear*
system of equations, which we are solving by back-substitution when we use the
backpropagation method. The reason why back-substitution is sufficient for the
linear system (i.e., we don't need a *full* linear system solver) is that the
dependency graph induced by the $\alpha$ relation is acyclic. If we had needed a
full linear system solver, the solution would take $\mathcal{O}(n^3)$ time
instead of linear time, seriously blowing-up our nice runtime!

This connection to linear systems is interesting: It tells us that we *could*
compute gradients with cyclic graphs. All we'd need is to run a linear system
solver to stich together our gradients! That is exactly what the
[implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem)
says!

Cyclic constraints add some expressive powerful to "constraint language" and
it's interesting that we can still efficiently compute gradients in this
setting. An example of what a general type of cyclic constraint looks like is

$$
\begin{align*}
& \underset{\boldsymbol{x}}{\text{argmax}}\, z_n \\
& \text{s.t.}\quad g(\boldsymbol{z}) = \boldsymbol{0} \\
& \text{and}\quad \boldsymbol{z}_{1:d} = \boldsymbol{x}
\end{align*}
$$

where $g$ can be an any smooth multivariate function of the intermediate
variables! Of course, allowing cyclic constraints come at the cost of a
more-difficult analogue of "the forward pass" to satisfy the $\boldsymbol{z}$
equations (if we want to keep it a block-coordinate step). The
$\boldsymbol{\lambda}$ equations are now a linear system that requres a linear
solver (e.g., Guassian elimination).

<!--
   IFT would give us
   0 = dL/dz = d/dz(f(z) - lambda*g(z))
   0 = df/dz - \boldsymbol{\lambda} * dg(\boldsymbol{z})
   df/dz * dg(\boldsymbol{z})^-1 = \boldsymbol{\lambda} = dz
-->


Use cases:

* Bi-level optimization: Solving an optimization problem with another on inside
  it. For example,
  [gradient-based hyperparameter optimization](http://timvieira.github.io/blog/post/2016/03/05/gradient-based-hyperparameter-optimization-and-the-implicit-function-theorem/)
  in machine learning. The implicit function theorem manages to get gradients of
  hyperparameters without needing to store any of intermediate states of the
  optimization algorithm used in the inner optimzation! This is a *huge* memory
  saver since direct backprop on the inner gradient decent algorithm would
  require caching all intermediate states. Yikes!

* Cyclic constraints are useful in many graph algorithms. For example, computing
  gradients of edge weights in a general finite-state machine or, similarly,
  computing the value function in a Markov decision process.


## Other methods for optimization?

The connection to Lagrangians brings tons of algorithms for constrained
optimization into the mix! We can imagine using more general algorithms for
optimizing our function and other ways of enforcing the constraints. We see
immediately that we could run optimization with adjoints set to values other
than those that backprop would set them to (i.e., we can optimize them like we'd
do in other algorithms for optimizing general Langrangians).


## Further reading

The theoretical view of backpropagation as an instance of the method of Lagrange
multipliers was first presented in

> Yann LeCun. (1988)
> [A Theoretical Framework from Back-Propagation](http://yann.lecun.com/exdb/publis/pdf/lecun-88.pdf).

The backpropagation algorithm can be cleanly generalized from values to
functionals!

> Alexander Grubb and J. Andrew Bagnell. (2010)
> [Boosted Backpropagation Learning for Training Deep Modular Networks](https://t.co/5OW5xBT4Y1).


A great blog post that uses the implicit function theorem to *derive* the method
of Lagrange multipliers. He also touches on the connection to backpropgation.

> Ben Recht. (2016)
> [Mechanics of Lagrangians](http://www.argmin.net/2016/05/31/mechanics-of-lagrangians/)/
