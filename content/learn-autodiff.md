title: Learn automatic differentiation
date: 2016-09-25
comments: true
status: draft
tags: calculus, autodiff

Here are some references for learning autodiff:

I learned automatic differentiation from section 2 of
[this document](http://users.cecs.anu.edu.au/~jdomke/courses/sml2011/08autodiff_nnets.pdf).

- It's very short and simple.

- I recommend doing the example on paper, including drawing the graph of the
  function so you can annotate it as you follow along.

- It's also useful to compare the automatic differentiation method to the (less
  efficient) method for taking gradients you learned in calculus (often called
  "symbolic differentiation").

  - The key: Just like intermediate quantities improve efficiency of function
    evaluation (i.e., "forward propagation"), the intermediate gradients (i.e.,
    partial derivatives of intermediate variables wrt the output node, often
    called "adjoints") allow for equally efficient gradient evaluation ("back
    propagation").

 - The graph structure of the backward algorithm is always the same as
   forward. (This is surprising.)

   We have:

   - adjoint of output node = 1

   - ``+=`` feeding into the adjoint value of all nodes (only necessary if
     an intermediate node is reused, otherwise you /can/ just assign the value
     with ``=``).

   - The ``+=`` values aggregated into a node always have the same form:
     adjoint of later node times the 'local gradient' at the edge.

   - The units of the quantities flowing along the edges always pass the unit
     test (just like the chain rule, but note that we add partials that reuse a
     node). $\frac{\partial out}{\partial v}$ ``+=`` $\frac{\partial
     out}{\partial v} \cdot \frac{\partial u}{\partial v}$ for $u \in
     \text{outgoing}(v)$

**Testing**

You can test your understanding by making up your own functions then testing
that you got the gradients correct using the
[finite-difference test](https://en.wikipedia.org/wiki/Numerical_differentiation)
(sometimes called "numerical differentiation")

- This is the main way that people test that the gradient implementation is
  correct (for the function its based on; remember though just because the
  fdcheck passes doesn't mean that the $f$ is implemented correctly, just that
  the gradient matches $f$, including any bugs that $f$ may have).

- You should always use the "two-point" version.

- Test the gradient at multiple inputs (maybe randomly chosen, or chosen to test
  corner/extreme cases).

- Compare the numerical gradient to your derived gradient using relative error,
  not absolute error. This is available under numpy.allclose.

- This "test suite" (described under these bullet points) is implemented in
  scipy.optimize.check_grad.
