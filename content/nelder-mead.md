Title: Black-box optimization
date: 2018-03-16
comments: true
tags: optimization calculus

Black-box optimization algorithms are a fantastic tool that everyone should be
aware of. I frequently use black-box optimization algorithms for prototyping and
when gradient-based algorithms fail,
e.g., because the function is not differentiable,
because the function is truly opaque (no gradients),
because the gradient would require too much memory to compute efficiently.

From a young age, we are taught to love gradients and hence become obsessed with
gradient descent. I believe this obsession has put us in a local optimum. I've
been amazed at how few people know about non-gradient algorithms of
optimization. This is slowly improving thanks to the prevalence of
hyperparameter optimization, so most people have used random search and (at
least) know of Bayesian optimization.

There are many ways to optimize a function! The gradient just happens to have a
[beautiful](/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/) and
[computationally efficient](/blog/post/2016/09/25/evaluating-fx-is-as-fast-as-fx/)
shortcut for finding *the direction of steepest descent* in Euclidean space.

**What is a descent direction anyway?** For minimizing an function $f:
\mathbb{R}^d \mapsto \mathbb{R}$, a descent direction for $f$ is a
$(d+1)$-dimensional hyperplane. The gradient gives a unique hyperplane that is
tangent to the surface of $f$ at the point $x$; the $(d+1)^{\text{th}}$
coordinate comes from the value $f(x)$&mdash;think of it like a first-order
Taylor approximation to $f$ at $x$.

**The baseline:** Without access to gradient code, approximating* the gradient
*takes $d+1$ function evaluations via the finite-difference approximation to the
*gradient,[^twosidedfd] which I've discussed a
*[few](http://timvieira.github.io/blog/post/2014/02/10/gradient-vector-product/)
*[times](http://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/). This
*shouldn't be surprising since that's the size of the object we're looking for
*anyways![^faster-but-noisy]

**Can we do better?** Suppose we had $(d+1)$ arbitrary points
$\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(d+1)}$ in $\mathbb{R}^n$ with
values $f^{(i)} = f(\boldsymbol{x}^{(i)}).$ Can we find efficiently find a
descent direction without extra $f$ evaluations?

**The Nelder-Mead trick:** Take the worst-performing point in this set
$\boldsymbol{x}^{(\text{worst})}$ and consider moving that point through the
center-of-mass of the $d$ remaining points. Call this the NM direction. At some
point along that direction (think line search) there will be a good place to put
that point, which will make it the new best point. We can now repeat this
process: pick the worst point, reflect it through the center of mass, etc.

 - The cost of finding the NM descent direction requires no additional function
   evaluations, which allows the method to be very frugal with function
   evaluations. Of course, stepping in the search direction should use line
   search, which will require additional function evaluations; gradient-based
   methods also benefit from line search.

 - Finding the worst point can be done in time $\mathcal{O}(\log d)$ using a
   [heap](https://en.wikipedia.org/wiki/Heap_(data_structure)).

 - This NM direction might is not the steepest descent direction&mdash;like the
   gradient&mdash;but it does give a reasonable descent direction to
   follow. Often, the NM direction is more useful than the gradient direction
   because it is not based on an infinitesimal ball around the current point
   like the gradient. NM often "works" on noisy and nonsmooth functions where
   gradients do not exist.

 - On high-dimensional problems, NM requires a significant number of "warm up"
 *function* evaluations before it can take its first informed step. Whereas,
 gradient descent could plausibly CONVERGE in fewer *gradient* evaluations
 (assuming sufficiently "nice" functions)! So, if you have high-dimensional
 problem and efficient gradients, use them.

 - In three dimensions, we can visualize this as a tetrahedron with corners that
   "stick" to the surface of the function. At each iterations, the highest
   (i.e., worst performing) point is the one most likely to be affected by
   "gravity" which causes it to flip through the middle of the blob, as the
   other points stay stuck.

   <center>
   ![Nelder-Mead animation](https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Nelder-Mead_Himmelblau.gif/320px-Nelder-Mead_Himmelblau.gif)
   <br/>*(animation source: Wikipedia page for Nelder-Mead)*
   </center>

 - This is exactly the descent direction used in the
   [Nelder-Mead algorithm](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
   (Nelder & Mead, 1965), which happens to be a great default algorithm for
   locally optimizing functions without access to gradients. Matlab and scipy
   users may know it better as
   [``fmin``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html).
   There are some additional "search moves" required to turn NM into a robust
   algorithm; these include shrinking and growing the set of points. I won't try
   to make yet another tutorial on the specifics of Nelder-Mead, as several
   already exist, but rather bring it to your attention as a plausible approach
   for efficiently finding descent directions. You can find a tutorial with
   plenty of visualization on its
   [Wikipedia page](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method).

### Summary

I described the Nelder-Mead search direction as an efficient way to leverage
past function evaluations to find a descent directions, which serves as a
reasonable alternative to gradients when they are unavailable (or not useful).


### Further reading

 - There are plenty of other black-box optimization algorithms out there. The
   wiki page on
   [derivative-free optimization](https://en.wikipedia.org/wiki/Derivative-free_optimization)
   is a good starting point for learning more.

[^twosidedfd]: Of course, it's better to use the two-sided difference
approximation to the gradient in practice, which requires $2 \cdot d$ function
evaluations, not $d+1$.

[^faster-but-noisy]: Note that we can get noisy, approximations with much fewer
than $\mathcal{O}(d)$ evaluations, e.g.,
[SPSA](https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation)
or even REINFORCE obtain gradients approximations with just $\mathcal{O}(1)$
evaluations per iteration.
